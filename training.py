import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from losses.dual_attention import attention_loss as dual_attention_loss
from losses.single_attention import attention_loss as single_attention_loss
from losses.complementary import complementary_loss
from losses.foreground import (
    background_suppression_loss, foreground_complementary_loss, foreground_anticorrelation_loss,
    normalized_dual_attention, region_attention_statistics, three_class_foreground_mask,
)
from losses.union import apply_union_margin, head_scores, patch_attention, union_alignment
from models import JointTransMIL


class JointTraining(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        config = deepcopy(config)
        self.complementary_mode = config['Model'].setdefault('complementary_mode', 'symmetric')
        if self.complementary_mode not in {'symmetric', 'asymmetric'}:
            raise ValueError('complementary_mode must be symmetric or asymmetric')
        for name in ('comp_grad_weight_0', 'comp_grad_weight_1'):
            value = float(config['Model'].setdefault(name, 1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must belong to [0, 1]')
            config['Model'][name] = value
            setattr(self, name, value)
        self.union_margin = float(config['Model']['union'].setdefault('margin', 0.0))
        if not np.isfinite(self.union_margin) or self.union_margin < 0:
            raise ValueError('Model.union.margin must be finite and nonnegative')
        config['Model']['union']['margin'] = self.union_margin
        self.complementary_loss_type = config['Model'].setdefault('complementary_loss_type', 'original')
        if self.complementary_loss_type not in {'original', 'foreground', 'anti-correlation'}:
            raise ValueError('complementary_loss_type must be original, foreground or anti-correlation')
        self.foreground_comp_type = config['Model'].setdefault('foreground_comp_type', 'product')
        if self.complementary_loss_type == 'anti-correlation':
            self.foreground_comp_type = config['Model']['foreground_comp_type'] = 'anticorrelation'
        if self.foreground_comp_type not in {'product', 'anticorrelation'}:
            raise ValueError('foreground_comp_type must be product or anticorrelation')
        self.foreground_comp_rho_target = float(config['Model'].setdefault('foreground_comp_rho_target', -.3))
        if not np.isfinite(self.foreground_comp_rho_target) or not -1 <= self.foreground_comp_rho_target <= 1:
            raise ValueError('foreground_comp_rho_target must belong to [-1, 1]')
        config['Model']['foreground_comp_rho_target'] = self.foreground_comp_rho_target
        for name, default in (('foreground_threshold', .4), ('background_attention_threshold', .2),
                              ('background_loss_weight', 0.)):
            value = float(config['Model'].setdefault(name, default))
            if not np.isfinite(value) or value < 0 or (name != 'background_loss_weight' and value > 1):
                raise ValueError(f'Invalid {name}: {value}')
            config['Model'][name] = value
            setattr(self, name, value)
        if self.complementary_loss_type == 'original' and self.background_loss_weight > 0:
            raise ValueError('background_loss_weight > 0 requires complementary_loss_type=foreground')
        self.save_hyperparameters({'config': config})
        self.config = config
        model_cfg = config['Model']
        self.model = JointTransMIL(
            use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
            gradient_checkpointing_min_patches=model_cfg.get('gradient_checkpointing_min_patches', 30000),
        )
        self.weights = model_cfg['loss_weights']
        if any(not np.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError('Loss weights must be finite and nonnegative')
        self.union_cfg = model_cfg['union']
        if self.union_cfg['normalization'] not in {'max', 'minmax', 'sum'}:
            raise ValueError('Union normalization must be max, minmax, or sum')
        self.validation_rows = []
        self.test_rows = []
        self.single_evaluation_losses = {
            stage: {'cls': [], 'attn': []} for stage in ('val', 'test')
        }

    def on_fit_start(self):
        self.print(
            f"Fold {self.config['Data']['fold']}: complementary_mode={self.complementary_mode}, "
            f"comp_grad_weight_0={self.comp_grad_weight_0}, comp_grad_weight_1={self.comp_grad_weight_1}, "
            f"effective_branch1_scale={0.0 if self.complementary_mode == 'asymmetric' else self.comp_grad_weight_1}"
        )
        self.print(f'Union KL tolerance margin={self.union_margin}')
        self.print(f'Complementary loss type={self.complementary_loss_type}, '
                   f'foreground_threshold={self.foreground_threshold}, '
                   f'background_attention_threshold={self.background_attention_threshold}, '
                   f'background_loss_weight={self.background_loss_weight}, '
                   f'foreground_comp_type={self.foreground_comp_type}, '
                   f'foreground_comp_rho_target={self.foreground_comp_rho_target}')

    def forward(self, data, return_attn=False):
        return self.model(data, return_attn)

    def _supervised_attention(self, tensors, h, classifier, class_idx, targets, flags, loss_fn):
        if not any(flags):
            return h.new_zeros((), dtype=torch.float32)
        for attn, target, flag in zip(tensors, targets, flags):
            if flag and target.numel() != attn.shape[-1]:
                raise ValueError('Precomputed target and student token counts differ')
        header = head_scores(h, classifier, class_idx, tensors[0].shape[1])
        return loss_fn(tensors, targets[0], targets[1], flags[0], flags[1], header).float()

    def calculate_losses(self, batch):
        data, label, label0, label1 = batch[:4]
        if data.ndim != 3 or data.shape[0] != 1 or data.shape[1] < 1:
            raise ValueError('Joint training requires one nonempty WSI bag [1, N, 1536]')
        y = int(label.item())
        expected = (int(y == 1), int(y != 0))
        if (int(label0.item()), int(label1.item())) != expected:
            raise ValueError('Unexpected dual labels for original CCA/HCC/TM label')
        outputs = self.model(data, return_attn=True)
        single, sa, sh = outputs['single']
        dual, da, dh = outputs['dual']
        losses = {
            'single_cls': F.cross_entropy(single['logits'], label.long()),
            'dual_cls': F.cross_entropy(dual['branch0']['logits'], label0.long())
                        + F.cross_entropy(dual['branch1']['logits'], label1.long()),
        }
        zero = data.new_zeros((), dtype=torch.float32)
        union_kl = zero
        losses.update(single_attn=zero, dual_attn=zero, complement=zero, union=zero)
        foreground_stats = {}
        if self.complementary_loss_type != 'original':
            losses['background'] = zero
        if y in (0, 1):
            flags0 = [int(batch[7].item()), int(batch[8].item())]
            flags1 = [int(batch[11].item()), int(batch[12].item())]
            targets0, targets1 = batch[5:7], batch[9:11]
            losses['single_attn'] = self._supervised_attention(
                sa, sh, self.model.single._fc2, y, targets0, flags0, single_attention_loss)
            losses['dual_attn'] = self._supervised_attention(
                da['branch0'], dh['branch0'], self.model.dual._fc2,
                expected[0], targets0, flags0, dual_attention_loss)
            losses['dual_attn'] = losses['dual_attn'] + self._supervised_attention(
                da['branch1'], dh['branch1'], self.model.dual._fc2_b1,
                expected[1], targets1, flags1, dual_attention_loss)
        else:
            if len(sa) != 2 or len(da['branch0']) != 2 or len(da['branch1']) != 2:
                raise ValueError('Joint TM losses require two corresponding transformer layers')
            overlaps, alignments = [], []
            backgrounds, region_stats, correlations = [], [], []
            if self.complementary_loss_type != 'original':
                mask = three_class_foreground_mask(sa, sh, self.model.single._fc2,
                                                   data.shape[1], self.foreground_threshold)
                foreground_stats['foreground_patches'] = mask.float().sum()
                foreground_stats['foreground_proportion'] = mask.float().mean()
            with torch.autocast(device_type=data.device.type, enabled=False):
                for layer in range(2):
                    p0 = patch_attention(da['branch0'][layer], dh['branch0'], self.model.dual._fc2, 0, data.shape[1])
                    p1 = patch_attention(da['branch1'][layer], dh['branch1'], self.model.dual._fc2_b1, 1, data.shape[1])
                    target = patch_attention(sa[layer], sh, self.model.single._fc2, 2, data.shape[1])
                    if self.complementary_loss_type != 'original':
                        a0, a1 = normalized_dual_attention(p0), normalized_dual_attention(p1)
                        if self.foreground_comp_type == 'anticorrelation':
                            overlap, rho = foreground_anticorrelation_loss(
                                a0, a1, mask, self.foreground_comp_rho_target,
                                mode=self.complementary_mode, grad_weight_0=self.comp_grad_weight_0,
                                grad_weight_1=self.comp_grad_weight_1,
                            )
                            overlaps.append(overlap)
                            correlations.append(rho)
                        else:
                            overlaps.append(foreground_complementary_loss(
                                a0, a1, mask, self.complementary_mode,
                                self.comp_grad_weight_0, self.comp_grad_weight_1,
                            ))
                        backgrounds.append(background_suppression_loss(
                            a0, a1, mask, self.background_attention_threshold,
                        ))
                        stats = region_attention_statistics(a0, a1, mask)
                        if self.foreground_comp_type == 'anticorrelation':
                            for branch, attention in enumerate((a0, a1)):
                                selected = attention[mask]
                                stats[f'fg_attention_std_branch{branch}'] = (
                                    selected.std(unbiased=False) if selected.numel() else attention.sum() * 0.)
                        region_stats.append(stats)
                    else:
                        overlaps.append(complementary_loss(
                            p0, p1, self.complementary_mode,
                            self.comp_grad_weight_0, self.comp_grad_weight_1,
                        ))
                    alignments.append(union_alignment(
                        p0, p1, target,
                        normalization=self.union_cfg['normalization'],
                        detach_target=self.union_cfg['detach_single_target'],
                    ))
                losses['complement'] = torch.stack(overlaps).mean()
                union_kl = torch.stack(alignments).mean()
                losses['union'] = apply_union_margin(union_kl, self.union_margin)
                if self.complementary_loss_type != 'original':
                    losses['background'] = torch.stack(backgrounds).mean()
                    foreground_stats['complement_fg'] = losses['complement']
                    for key in region_stats[0]:
                        foreground_stats[key] = torch.stack([stats[key] for stats in region_stats]).mean()
                    if self.foreground_comp_type == 'anticorrelation':
                        valid = [rho for rho in correlations if rho is not None]
                        foreground_stats['complement_fg_anticorrelation'] = losses['complement']
                        foreground_stats['background_tm'] = losses['background']
                        foreground_stats['union_tm'] = losses['union']
                        foreground_stats['foreground_correlation_undefined_count'] = zero + len(correlations) - len(valid)
                        foreground_stats['foreground_correlation_undefined_fraction'] = (
                            foreground_stats['foreground_correlation_undefined_count'] / len(correlations))
                        # Missing/constant layers are not assigned a misleading rho=0.
                        if valid:
                            foreground_stats['foreground_rho'] = torch.stack(valid).mean()
        outputs['union_kl'] = union_kl
        outputs['foreground_stats'] = foreground_stats
        union_active = int(self.current_epoch) >= int(self.union_cfg.get('start_epoch', 0))
        total = sum((self.background_loss_weight if key == 'background' else self.weights[key])
                    * value * (union_active if key == 'union' else 1)
                    for key, value in losses.items())
        if not torch.isfinite(total):
            raise FloatingPointError(f'Non-finite joint loss for {batch[4]}')
        return total, losses, outputs

    def _log_foreground_stats(self, stage, outputs):
        for key, value in outputs['foreground_stats'].items():
            self.log(f'{stage}_{key}', value.detach(), on_step=False, on_epoch=True, batch_size=1)

    def training_step(self, batch, batch_idx):
        total, losses, outputs = self.calculate_losses(batch)
        self.log('train_total', total, on_step=False, on_epoch=True, batch_size=1)
        self.log('train_union_kl', outputs['union_kl'], on_step=False, on_epoch=True, batch_size=1)
        self._log_foreground_stats('train', outputs)
        for key, value in losses.items():
            self.log(f'train_{key}', value, on_step=False, on_epoch=True, batch_size=1)
        return total

    def _evaluation_step(self, batch, stage):
        total, losses, outputs = self.calculate_losses(batch)
        tracked = self.single_evaluation_losses[stage]
        tracked['cls'].append(losses['single_cls'].detach())
        if int(batch[1].item()) in (0, 1) and (int(batch[7].item()) or int(batch[8].item())):
            tracked['attn'].append(losses['single_attn'].detach())
        self.log(f'{stage}_total', total, on_epoch=True, batch_size=1, prog_bar=True)
        self.log(f'{stage}_union_kl', outputs['union_kl'], on_epoch=True, batch_size=1)
        self._log_foreground_stats(stage, outputs)
        cls_loss = self.weights['single_cls'] * losses['single_cls'] + self.weights['dual_cls'] * losses['dual_cls']
        self.log(f'{stage}_cls_loss', cls_loss, on_epoch=True, batch_size=1)
        for key, value in losses.items():
            self.log(f'{stage}_{key}', value, on_epoch=True, batch_size=1)
        single = outputs['single'][0]
        dual = outputs['dual'][0]
        p0, p1 = dual['branch0']['Y_prob'][0], dual['branch1']['Y_prob'][0]
        hat0, hat1 = int(p0.argmax()), int(p1.argmax())
        # (0,0)=CCA, (1,1)=HCC, (0,1)=TM; (1,0) is inconsistent.
        dual_pred = {(0, 0): 0, (1, 1): 1, (0, 1): 2}.get((hat0, hat1), -1)
        row = dict(slide_id=batch[4][0], fold=self.config['Data']['fold'], true_label=int(batch[1].item()),
                   single_pred=int(single['Y_hat'].item()), dual_pred=dual_pred,
                   branch0_pred=hat0, branch1_pred=hat1)
        for i, name in enumerate(('CCA', 'HCC', 'TM')):
            row[f'single_prob_{name}'] = float(single['Y_prob'][0, i].detach().cpu())
        for name, prob in zip(('CCA', 'HCC', 'TM'), (torch.minimum(p0[0], p1[0]), torch.minimum(p0[1], p1[1]), torch.minimum(p0[0], p1[1]))):
            row[f'dual_prob_{name}'] = float(prob.detach().cpu())
        getattr(self, 'validation_rows' if stage == 'val' else 'test_rows').append(row)
        return total

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'test')

    def _finish_evaluation(self, rows, stage):
        tracked = self.single_evaluation_losses[stage]
        if not rows:
            for values in tracked.values():
                values.clear()
            return
        if self.trainer.sanity_checking:
            rows.clear()
            for values in tracked.values():
                values.clear()
            return
        # Match standalone task 2: CE over all slides, attention over only
        # supervised slides (not diluted by TM or missing reference maps).
        single_cls = torch.stack(tracked['cls']).mean()
        single_attn = torch.stack(tracked['attn']).mean() if tracked['attn'] else single_cls.new_zeros(())
        single_total = self.weights['single_cls'] * single_cls + self.weights['single_attn'] * single_attn
        self.log(f'{stage}_single_attn_supervised', single_attn, on_step=False, on_epoch=True)
        self.log(f'{stage}_single_total', single_total, on_step=False, on_epoch=True)
        for values in tracked.values():
            values.clear()
        y = np.array([r['true_label'] for r in rows])
        metric_rows = []
        for name in ('single', 'dual'):
            pred = np.array([r[f'{name}_pred'] for r in rows])
            metrics = dict(epoch=int(self.current_epoch), model=name, accuracy=accuracy_score(y, pred))
            # Count inconsistent dual decisions as errors without introducing a fourth true class.
            per_class_f1 = []
            for i, cls in enumerate(('CCA', 'HCC', 'TM')):
                truth, decision = y == i, pred == i
                per_class_f1.append(f1_score(truth, decision, zero_division=0))
                metrics[f'recall_{cls}'] = recall_score(truth, decision, zero_division=0)
                metrics[f'auc_{cls}'] = roc_auc_score(truth, [r[f'{name}_prob_{cls}'] for r in rows]) if len(np.unique(truth)) == 2 else float('nan')
            metrics['macro_f1'] = float(np.mean(per_class_f1))
            self.log(f'{stage}_{name}_accuracy', metrics['accuracy'])
            self.log(f'{stage}_{name}_macro_f1', metrics['macro_f1'])
            metric_rows.append(metrics)
        if self.trainer.is_global_zero:
            folder = Path(self.config['output_dir']) / f"fold{self.config['Data']['fold']}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f'{stage}_predictions.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            path = folder / f'{stage}_metrics.csv'
            exists = path.exists()
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
                if not exists:
                    writer.writeheader()
                writer.writerows(metric_rows)
        rows.clear()

    def on_validation_epoch_end(self):
        self._finish_evaluation(self.validation_rows, 'val')

    def on_test_epoch_end(self):
        self._finish_evaluation(self.test_rows, 'test')

    def configure_optimizers(self):
        cfg = self.config['Optimizer']
        groups = []
        for model, lr in ((self.model.single, cfg['single_lr']),
                          (self.model.dual, cfg['dual_lr'])):
            decay, no_decay = [], []
            skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else ()
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim == 1 or name.endswith('.bias') or name in skip:
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups.extend([
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.},
                {'params': decay, 'lr': lr, 'weight_decay': cfg.get('weight_decay', 1e-5)},
            ])
        return torch.optim.RAdam(groups, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.)
