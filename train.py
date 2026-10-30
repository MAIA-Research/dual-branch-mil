import argparse
import math
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets import DataInterface
from training import JointTraining


class LastOnlyCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_train_end(self, trainer, pl_module):
        # Preserve the final validation checkpoint's completed-epoch metadata.
        if not self.last_model_path:
            super().on_train_end(trainer, pl_module)


def checkpoint_callbacks(folder):
    return [
        ModelCheckpoint(
            dirpath=folder, monitor='val_total', mode='min', save_top_k=1,
            filename='epoch={epoch:02d}-best-total-val_total={val_total:.4f}',
            auto_insert_metric_name=False, save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=folder, monitor='val_single_cls', mode='min', save_top_k=1,
            filename='epoch={epoch:02d}-best-classification-val_single_cls={val_single_cls:.4f}',
            auto_insert_metric_name=False, save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=folder, monitor='val_single_total', mode='min', save_top_k=1,
            filename='epoch={epoch:02d}-best-single-total-val_single_total={val_single_total:.4f}',
            auto_insert_metric_name=False, save_on_train_epoch_end=False,
        ),
        LastOnlyCheckpoint(
            dirpath=folder, save_top_k=0, save_last=True, every_n_epochs=1,
            enable_version_counter=False, save_on_train_epoch_end=False,
        ),
    ]


def initialize_part(module, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    state = {key.removeprefix('model.'): value for key, value in state.items()}
    # Existing Lightning checkpoints can include metric buffers beside model parameters.
    expected = module.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    module.load_state_dict(filtered, strict=True)
    print(f'Initialized from {path}')


def run(config, fold, resume=None, stage='train'):
    config = deepcopy(config)
    config['Data']['fold'] = fold
    pl.seed_everything(config['General'].get('seed', 2021), workers=True)
    data = Dict(config['Data'])
    if not data.dual_branch or data.ablation_case != 5:
        raise ValueError('Data must use dual_branch: true and ablation_case: 5')
    if data.train_dataloader.batch_size != 1 or data.test_dataloader.batch_size != 1:
        raise ValueError('Use batch_size: 1 for variable-length WSI bags')
    dm = DataInterface(
        train_batch_size=1, train_num_workers=data.train_dataloader.num_workers,
        test_batch_size=1, test_num_workers=data.test_dataloader.num_workers,
        dataset_name=data.get('dataset_name', 'PLC_data'), dataset_cfg=data,
        loader_seed=int(config['General'].get('seed', 2021)) + int(fold),
    )
    model = JointTraining(config)
    for part in ('single', 'dual'):
        path = config.get('Initialization', {}).get(f'{part}_checkpoint')
        if path:
            if resume:
                raise ValueError('Use initialization or resume, not both')
            initialize_part(getattr(model.model, part), str(path).format(fold=fold))
    folder = Path(config['output_dir']) / f'fold{fold}'
    general = config['General']
    callbacks = checkpoint_callbacks(folder)
    callbacks.append(EarlyStopping(
        monitor='val_single_cls', mode='min',
        patience=int(general.get('patience', 20)),
        min_delta=float(general.get('min_delta', 0.001)),
        check_on_train_epoch_end=False,
    ))
    trainer = pl.Trainer(
        accelerator=general.get('accelerator', 'gpu'), devices=general.get('devices', [0]),
        precision=general.get('precision', 32), max_epochs=general['epochs'],
        accumulate_grad_batches=general.get('grad_acc', 2),
        gradient_clip_val=general.get('gradient_clip_val', 1.0),
        deterministic=True, num_sanity_val_steps=0,
        callbacks=callbacks, logger=CSVLogger(config['output_dir'], name=f'fold{fold}', version='history'),
        log_every_n_steps=1,
    )
    if stage == 'train':
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
    else:
        if not resume:
            raise ValueError('--stage test requires --resume_ckpt')
        trainer.test(model, datamodule=dm, ckpt_path=resume)


def main():
    parser = argparse.ArgumentParser(description='Joint task-2 and case-5 TransMIL training')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--stage', choices=['train', 'test'], default='train')
    parser.add_argument(
        '--complementary_mode',
        type=str,
        choices=['symmetric', 'asymmetric'],
        default=None,
        help='Gradient direction for the dual-branch complementary loss',
    )
    parser.add_argument(
        '--comp_grad_weight_0', type=float, default=None,
        help='Gradient scaling factor of complementary loss for dual Branch 0',
    )
    parser.add_argument(
        '--comp_grad_weight_1', type=float, default=None,
        help='Gradient scaling factor of complementary loss for dual Branch 1',
    )
    parser.add_argument('--complementary_loss_type', choices=['original', 'foreground', 'anti-correlation'], default=None)
    parser.add_argument('--foreground_comp_type', choices=['product', 'anticorrelation'], default=None)
    parser.add_argument('--foreground_comp_rho_target', type=float, default=None)
    parser.add_argument('--foreground_threshold', type=float, default=None)
    parser.add_argument('--background_attention_threshold', type=float, default=None)
    parser.add_argument('--background_loss_weight', type=float, default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    defaults = dict(complementary_mode='symmetric', comp_grad_weight_0=1., comp_grad_weight_1=1.,
                    complementary_loss_type='original', foreground_threshold=.4,
                    background_attention_threshold=.2, background_loss_weight=0.,
                    foreground_comp_type='product', foreground_comp_rho_target=-.3)
    # Explicit CLI options override YAML; omitted options retain YAML or legacy defaults.
    for name, default in defaults.items():
        override = getattr(args, name)
        value = override if override is not None else config['Model'].get(name, default)
        config['Model'][name] = value
        setattr(args, name, value)
    if args.complementary_mode not in {'symmetric', 'asymmetric'}:
        parser.error('Invalid complementary_mode')
    if args.complementary_loss_type not in {'original', 'foreground', 'anti-correlation'}:
        parser.error('Invalid complementary_loss_type')
    if args.complementary_loss_type == 'anti-correlation':
        config['Model']['foreground_comp_type'] = args.foreground_comp_type = 'anticorrelation'
    if args.foreground_comp_type not in {'product', 'anticorrelation'}:
        parser.error('Invalid foreground_comp_type')
    if not math.isfinite(args.foreground_comp_rho_target) or not -1 <= args.foreground_comp_rho_target <= 1:
        parser.error('--foreground_comp_rho_target must belong to [-1, 1]')
    for name in ('comp_grad_weight_0', 'comp_grad_weight_1'):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f'--{name} must belong to [0, 1]')
    for name in ('foreground_threshold', 'background_attention_threshold', 'background_loss_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != 'background_loss_weight' and value > 1):
            parser.error(f'Invalid --{name}: {value}')
    if args.complementary_loss_type == 'original' and args.background_loss_weight > 0:
        parser.error('--background_loss_weight > 0 requires --complementary_loss_type foreground')
    folds = [args.fold] if args.fold is not None else config.get('folds', [config['Data']['fold']])
    if args.resume_ckpt and len(folds) != 1:
        raise ValueError('Specify one --fold when resuming/testing a checkpoint')
    for fold in folds:
        run(config, fold, args.resume_ckpt, args.stage)


if __name__ == '__main__':
    main()
