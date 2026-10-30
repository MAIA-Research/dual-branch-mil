import torch
import pandas as pd
from pathlib import Path
import numpy as np
import numpy as _np
import os
import h5py
import math

import torch.utils.data as data
from torch.utils.data import dataloader


class PLCData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.cca_1_stain_feature_dir = None
        if self.dataset_cfg.get('CCA_1_stain_norm', False):
            self.cca_1_stain_feature_dir = Path(self.dataset_cfg.get(
                'CCA_1_stain_norm_feature_dir',
                '/media/local-admin/Crucial X92/MAIA/MIL_10/CCA_1/features_CCA_1_stain_slide/pt_files',
            ))
            if not self.cca_1_stain_feature_dir.is_dir():
                raise FileNotFoundError(
                    f'CCA_1_stain_norm feature directory not found: {self.cca_1_stain_feature_dir}'
                )
        precomputed_root = self.dataset_cfg.get('precomputed_attention_root', '')
        if precomputed_root:
            fold_root = Path(precomputed_root) / f'fold_{self.fold}'
            block_0_dir = fold_root / 'attns_save_1'
            block_1_dir = fold_root / 'attns_save_2'
            self.heatmap_dir_label_0_0 = block_0_dir
            self.heatmap_dir_label_1_0 = block_0_dir
            self.heatmap_dir_label_0_1 = block_1_dir
            self.heatmap_dir_label_1_1 = block_1_dir
        else:
            self.heatmap_dir_label_0_0 = Path(self.dataset_cfg.get('attn_dir_label_0_0', ''))
            self.heatmap_dir_label_1_0 = Path(self.dataset_cfg.get('attn_dir_label_1_0', ''))
            self.heatmap_dir_label_0_1 = Path(self.dataset_cfg.get('attn_dir_label_0_1', ''))
            self.heatmap_dir_label_1_1 = Path(self.dataset_cfg.get('attn_dir_label_1_1', ''))
        self.filtered_coord_dir = self.dataset_cfg.get('filtered_coord_dir', '')
        self.filtered_coord_dir_branch1 = self.dataset_cfg.get(
            'filtered_coord_dir_branch1',
            self.filtered_coord_dir,
        )
        self.cluster_coord_dir = self.dataset_cfg.get('cluster_coord_dir', '')
        self.use_precomputed_attention_targets = bool(
            self.dataset_cfg.get('use_precomputed_attention_targets', True)
        )
        self.use_segmentation_annotations = bool(
            self.dataset_cfg.get('use_segmentation_annotations', True)
        )
        self.use_cluster_attention_targets = bool(
            self.dataset_cfg.get('use_cluster_attention_targets', False)
        )
        self.annotation_layers = self.dataset_cfg.get('annotation_layers', [1, 1])
        self.min_annotation_patches = int(self.dataset_cfg.get('min_annotation_patches', 10))
        self.feature_h5_dir = self.dataset_cfg.get(
            'feature_h5_dir',
            str(Path(self.feature_dir).parent / 'h5_files'),
        )
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        self.attn_code = self.dataset_cfg.attn_code
        self.dual_branch = bool(getattr(self.dataset_cfg, 'dual_branch', False))
        self.ablation_case = int(self.dataset_cfg.get('ablation_case', 0) or 0)
        self.max_patches_per_slide = int(getattr(self.dataset_cfg, 'max_patches_per_slide', 0) or 0)
        self.max_patches_per_tm_slide = int(getattr(self.dataset_cfg, 'max_patches_per_tm_slide', 0) or 0)
        self.l2_normalize_features = bool(self.dataset_cfg.get('l2_normalize_features', False))
        self.smooth_annotation_targets = bool(self.dataset_cfg.get('smooth_annotation_targets', False))
        self.annotation_smoothing_sigma = float(self.dataset_cfg.get('annotation_smoothing_sigma', 1.0))
        self.annotation_smoothing_radius = int(self.dataset_cfg.get('annotation_smoothing_radius', 2))
        self._printed_patch_cap_slides = set()

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()


    def __len__(self):
        return len(self.data)

    def _feature_path_for_slide(self, slide_id):
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        if self.cca_1_stain_feature_dir is not None:
            stain_path = self.cca_1_stain_feature_dir / f'{slide_id}.pt'
            if stain_path.is_file():
                full_path = stain_path
        return full_path

    def _coord_steps(self, coords):
        steps = []
        for axis in range(coords.shape[1]):
            values = _np.unique(coords[:, axis])
            diffs = _np.diff(_np.sort(values))
            diffs = diffs[diffs > 0]
            steps.append(float(_np.median(diffs)) if diffs.size > 0 else 1.0)
        return _np.asarray(steps, dtype=_np.float32)

    def _annotation_patch_target(self, full_coords, annotated_coords):
        annotated_set = set(map(tuple, annotated_coords.tolist()))
        if not self.smooth_annotation_targets:
            return torch.tensor(
                [tuple(coord) in annotated_set for coord in full_coords.tolist()],
                dtype=torch.float32,
            )

        radius = max(0, self.annotation_smoothing_radius)
        sigma = max(float(self.annotation_smoothing_sigma), 1e-6)
        steps = self._coord_steps(full_coords).tolist()
        coord_to_index = {
            tuple(coord): idx
            for idx, coord in enumerate(full_coords.tolist())
        }
        patch_target = torch.zeros(full_coords.shape[0], dtype=torch.float32)

        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist2 = dx * dx + dy * dy
                if dist2 > radius * radius:
                    continue
                weight = math.exp(-dist2 / (2.0 * sigma * sigma))
                offsets.append((dx, dy, float(weight)))

        for coord in annotated_coords.tolist():
            base_x, base_y = coord[:2]
            for dx, dy, weight in offsets:
                candidate = (
                    int(round(base_x + dx * steps[0])),
                    int(round(base_y + dy * steps[1])),
                )
                idx = coord_to_index.get(candidate)
                if idx is not None and float(patch_target[idx]) < weight:
                    patch_target[idx] = weight

        return patch_target

    def _build_annotation_heatmap(self, slide_id, n_patches, filtered_coord_dir=None):
        filtered_coord_dir = self.filtered_coord_dir if filtered_coord_dir is None else filtered_coord_dir
        if not filtered_coord_dir:
            return None

        filtered_h5_path = Path(filtered_coord_dir) / f'{slide_id}.h5'
        if not filtered_h5_path.exists():
            return None

        full_h5_path = Path(self.feature_h5_dir) / f'{slide_id}.h5'
        if not full_h5_path.exists():
            raise FileNotFoundError(
                f'Found filtered coords for {slide_id}, but missing full feature coords: {full_h5_path}'
            )

        with h5py.File(full_h5_path, 'r') as f:
            full_coords = f['coords'][:]
        with h5py.File(filtered_h5_path, 'r') as f:
            annotated_coords = f['coords'][:]

        if full_coords.shape[0] != n_patches:
            raise ValueError(
                f'Coordinate/feature length mismatch for {slide_id}: '
                f'{full_coords.shape[0]} coords for {n_patches} features'
            )

        patch_target = self._annotation_patch_target(full_coords, annotated_coords)
        return torch.cat([torch.zeros(1, dtype=torch.float32), patch_target], dim=0)

    def _build_cluster_heatmap(self, slide_id, n_patches):
        if not self.cluster_coord_dir:
            return None

        selected_h5_path = Path(self.cluster_coord_dir) / f'{slide_id}.h5'
        if not selected_h5_path.exists():
            return None

        with h5py.File(selected_h5_path, 'r') as f:
            selected_coords = f['coords'][:]
        if selected_coords.shape[0] < self.min_annotation_patches:
            return None

        full_h5_path = Path(self.feature_h5_dir) / f'{slide_id}.h5'
        if not full_h5_path.exists():
            raise FileNotFoundError(
                f'Found cluster coords for {slide_id}, but missing full feature coords: {full_h5_path}'
            )
        with h5py.File(full_h5_path, 'r') as f:
            full_coords = f['coords'][:]
        if full_coords.shape[0] != n_patches:
            raise ValueError(
                f'Coordinate/feature length mismatch for {slide_id}: '
                f'{full_coords.shape[0]} coords for {n_patches} features'
            )

        patch_target = self._annotation_patch_target(full_coords, selected_coords)
        if patch_target.sum() <= 0:
            return None
        return torch.cat([torch.zeros(1, dtype=torch.float32), patch_target], dim=0)

    def _apply_annotation_heatmap(self, label, annotation_heatmap):
        heatmap_0 = torch.zeros_like(annotation_heatmap)
        heatmap_1 = torch.zeros_like(annotation_heatmap)
        has_heatmap_0 = 0
        has_heatmap_1 = 0
        label_code = label + 1

        if label == 0:
            if self.attn_code[0] == 1:
                heatmap_0 = annotation_heatmap.clone()
                has_heatmap_0 = label_code
            if self.attn_code[1] == 1:
                heatmap_1 = annotation_heatmap.clone()
                has_heatmap_1 = label_code
        elif label == 1:
            if self.attn_code[2] == 1:
                heatmap_0 = annotation_heatmap.clone()
                has_heatmap_0 = label_code
            if self.attn_code[3] == 1:
                heatmap_1 = annotation_heatmap.clone()
                has_heatmap_1 = label_code

        return heatmap_0, heatmap_1, has_heatmap_0, has_heatmap_1

    def _dual_labels(self, label):
        # Original 3-class labels are expected to be CCA=0, HCC=1, TM=2.
        label_branch0 = 1 if label == 1 else 0  # CCA+TM=0 vs HCC=1
        label_branch1 = 0 if label == 0 else 1  # CCA=0 vs HCC+TM=1
        return label_branch0, label_branch1

    def _load_heatmap(self, path, features, has_value):
        heatmap = torch.zeros(features.shape[0] + 1)
        has_heatmap = 0
        if os.path.exists(path):
            heatmap = np.load(path)
            heatmap = torch.from_numpy(heatmap).float().squeeze(0)
            has_heatmap = has_value
        return heatmap, has_heatmap

    def _dual_branch_uses_heatmap(self, branch_name, label):
        if self.ablation_case == 1:
            return False
        if self.ablation_case in (2, 4):
            return (branch_name == 'branch0' and label == 0) or (branch_name == 'branch1' and label == 1)
        if self.ablation_case in (3, 5, 6, 7):
            return label in (0, 1)
        return label in (0, 1)

    def _shuffle_heatmap(self, heatmap, perm, slide_id, heatmap_name):
        cls_score = heatmap[0:1]
        patch_scores = heatmap[1:]
        if perm.numel() > 0 and int(perm.max()) >= patch_scores.shape[0]:
            raise ValueError(
                f'{heatmap_name} length mismatch for {slide_id}: '
                f'{patch_scores.shape[0]} scores, max selected index {int(perm.max())}'
            )
        patch_scores = patch_scores.index_select(0, perm)
        return torch.cat([cls_score, patch_scores], dim=0)

    def _patch_selection(self, n_patches, label):
        if self.shuffle == True:
            perm = torch.randperm(n_patches)
        else:
            perm = torch.arange(n_patches)
        if self.max_patches_per_slide > 0:
            perm = perm[:self.max_patches_per_slide]
        if self.dual_branch and label == 2 and self.max_patches_per_tm_slide > 0:
            perm = perm[:self.max_patches_per_tm_slide]
        return perm

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = self._feature_path_for_slide(slide_id)
        features = torch.load(full_path, map_location='cpu')
        if not isinstance(features, torch.Tensor):
            raise TypeError(f'Expected tensor features for {slide_id}, got {type(features)}')
        if features.dim() != 2:
            raise ValueError(f'Expected 2D features for {slide_id}, got shape {tuple(features.shape)}')
        if features.shape[1] != 1536:
            raise ValueError(f'Expected feature dim 1536 for {slide_id}, got shape {tuple(features.shape)}')
        features = features.float().contiguous()
        if self.l2_normalize_features:
            features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
        heatmap_path_label_0_0 = Path(self.heatmap_dir_label_0_0) / f'{slide_id}.npy'
        heatmap_path_label_0_1 = Path(self.heatmap_dir_label_0_1) / f'{slide_id}.npy'
        heatmap_path_label_1_0 = Path(self.heatmap_dir_label_1_0) / f'{slide_id}.npy'
        heatmap_path_label_1_1 = Path(self.heatmap_dir_label_1_1) / f'{slide_id}.npy'
        # print('heatmap_path:', heatmap_path)

        if self.dual_branch:
            label_branch0, label_branch1 = self._dual_labels(label)

            heatmap_b0_0 = torch.zeros(features.shape[0]+1)
            heatmap_b0_1 = torch.zeros(features.shape[0]+1)
            heatmap_b1_0 = torch.zeros(features.shape[0]+1)
            heatmap_b1_1 = torch.zeros(features.shape[0]+1)
            has_b0_0 = 0
            has_b0_1 = 0
            has_b1_0 = 0
            has_b1_1 = 0
            has_annotation_b0 = 0
            has_annotation_b1 = 0
            tm_non_tumor_heatmap = torch.zeros(features.shape[0]+1)
            has_tm_non_tumor_heatmap = 0

            cluster_heatmap = None
            if self.use_cluster_attention_targets:
                cluster_heatmap = self._build_cluster_heatmap(slide_id, features.shape[0])
            if cluster_heatmap is not None:
                if label == 2:
                    # TM coordinates mark non-tumor, not positive-attention patches.
                    tm_non_tumor_heatmap = cluster_heatmap.clone()
                    has_tm_non_tumor_heatmap = 1
                else:
                    branch0_code = label_branch0 + 1
                    branch1_code = label_branch1 + 1
                    if len(self.annotation_layers) > 0 and int(self.annotation_layers[0]) == 1:
                        heatmap_b0_0 = cluster_heatmap.clone()
                        heatmap_b1_0 = cluster_heatmap.clone()
                        has_b0_0 = branch0_code
                        has_b1_0 = branch1_code
                    if len(self.annotation_layers) > 1 and int(self.annotation_layers[1]) == 1:
                        heatmap_b0_1 = cluster_heatmap.clone()
                        heatmap_b1_1 = cluster_heatmap.clone()
                        has_b0_1 = branch0_code
                        has_b1_1 = branch1_code

            if cluster_heatmap is None and self.use_precomputed_attention_targets and label == 0:
                if self.attn_code[0] == 1:
                    heatmap, has_heatmap = self._load_heatmap(heatmap_path_label_0_0, features, 1)
                    if self._dual_branch_uses_heatmap('branch0', label):
                        heatmap_b0_0 = heatmap.clone()
                        has_b0_0 = has_heatmap
                    if self._dual_branch_uses_heatmap('branch1', label):
                        heatmap_b1_0 = heatmap.clone()
                        has_b1_0 = has_heatmap
                if self.attn_code[1] == 1:
                    heatmap, has_heatmap = self._load_heatmap(heatmap_path_label_0_1, features, 1)
                    if self._dual_branch_uses_heatmap('branch0', label):
                        heatmap_b0_1 = heatmap.clone()
                        has_b0_1 = has_heatmap
                    if self._dual_branch_uses_heatmap('branch1', label):
                        heatmap_b1_1 = heatmap.clone()
                        has_b1_1 = has_heatmap

            if cluster_heatmap is None and label == 1:
                annotation_heatmap_hcc = None
                if self.use_segmentation_annotations:
                    annotation_heatmap_hcc = self._build_annotation_heatmap(
                        slide_id,
                        features.shape[0],
                        self.filtered_coord_dir_branch1,
                    )
                if annotation_heatmap_hcc is not None:
                    if self.attn_code[2] == 1:
                        if self._dual_branch_uses_heatmap('branch0', label):
                            heatmap_b0_0 = annotation_heatmap_hcc.clone()
                            has_b0_0 = 2
                            has_annotation_b0 = 1
                        if self._dual_branch_uses_heatmap('branch1', label):
                            heatmap_b1_0 = annotation_heatmap_hcc.clone()
                            has_b1_0 = 2
                            has_annotation_b1 = 1
                    if self.attn_code[3] == 1:
                        if self._dual_branch_uses_heatmap('branch0', label):
                            heatmap_b0_1 = annotation_heatmap_hcc.clone()
                            has_b0_1 = 2
                            has_annotation_b0 = 1
                        if self._dual_branch_uses_heatmap('branch1', label):
                            heatmap_b1_1 = annotation_heatmap_hcc.clone()
                            has_b1_1 = 2
                            has_annotation_b1 = 1
                elif self.use_precomputed_attention_targets:
                    if self.attn_code[2] == 1:
                        heatmap, has_heatmap = self._load_heatmap(heatmap_path_label_1_0, features, 2)
                        if self._dual_branch_uses_heatmap('branch0', label):
                            heatmap_b0_0 = heatmap.clone()
                            has_b0_0 = has_heatmap
                        if self._dual_branch_uses_heatmap('branch1', label):
                            heatmap_b1_0 = heatmap.clone()
                            has_b1_0 = has_heatmap
                    if self.attn_code[3] == 1:
                        heatmap, has_heatmap = self._load_heatmap(heatmap_path_label_1_1, features, 2)
                        if self._dual_branch_uses_heatmap('branch0', label):
                            heatmap_b0_1 = heatmap.clone()
                            has_b0_1 = has_heatmap
                        if self._dual_branch_uses_heatmap('branch1', label):
                            heatmap_b1_1 = heatmap.clone()
                            has_b1_1 = has_heatmap

            perm = self._patch_selection(features.shape[0], label)
            if perm.shape[0] != features.shape[0]:
                original_n_patches = features.shape[0]
                if slide_id not in self._printed_patch_cap_slides:
                    print(
                        f'[PLCData] patch cap for {slide_id}: '
                        f'{original_n_patches} -> {perm.shape[0]} patches'
                    )
                    self._printed_patch_cap_slides.add(slide_id)
                features = features.index_select(0, perm)
                if has_b0_0 != 0:
                    heatmap_b0_0 = self._shuffle_heatmap(heatmap_b0_0, perm, slide_id, 'Branch 0 heatmap 0')
                else:
                    heatmap_b0_0 = torch.zeros(features.shape[0]+1)
                if has_b0_1 != 0:
                    heatmap_b0_1 = self._shuffle_heatmap(heatmap_b0_1, perm, slide_id, 'Branch 0 heatmap 1')
                else:
                    heatmap_b0_1 = torch.zeros(features.shape[0]+1)
                if has_b1_0 != 0:
                    heatmap_b1_0 = self._shuffle_heatmap(heatmap_b1_0, perm, slide_id, 'Branch 1 heatmap 0')
                else:
                    heatmap_b1_0 = torch.zeros(features.shape[0]+1)
                if has_b1_1 != 0:
                    heatmap_b1_1 = self._shuffle_heatmap(heatmap_b1_1, perm, slide_id, 'Branch 1 heatmap 1')
                else:
                    heatmap_b1_1 = torch.zeros(features.shape[0]+1)
                if has_tm_non_tumor_heatmap != 0:
                    tm_non_tumor_heatmap = self._shuffle_heatmap(
                        tm_non_tumor_heatmap, perm, slide_id, 'TM non-tumor heatmap'
                    )
                else:
                    tm_non_tumor_heatmap = torch.zeros(features.shape[0]+1)
            elif self.shuffle == True:
                features = features.index_select(0, perm)
                if has_b0_0 != 0:
                    heatmap_b0_0 = self._shuffle_heatmap(heatmap_b0_0, perm, slide_id, 'Branch 0 heatmap 0')
                if has_b0_1 != 0:
                    heatmap_b0_1 = self._shuffle_heatmap(heatmap_b0_1, perm, slide_id, 'Branch 0 heatmap 1')
                if has_b1_0 != 0:
                    heatmap_b1_0 = self._shuffle_heatmap(heatmap_b1_0, perm, slide_id, 'Branch 1 heatmap 0')
                if has_b1_1 != 0:
                    heatmap_b1_1 = self._shuffle_heatmap(heatmap_b1_1, perm, slide_id, 'Branch 1 heatmap 1')
                if has_tm_non_tumor_heatmap != 0:
                    tm_non_tumor_heatmap = self._shuffle_heatmap(
                        tm_non_tumor_heatmap, perm, slide_id, 'TM non-tumor heatmap'
                    )

            return (
                features,
                label,
                label_branch0,
                label_branch1,
                slide_id,
                heatmap_b0_0,
                heatmap_b0_1,
                has_b0_0,
                has_b0_1,
                heatmap_b1_0,
                heatmap_b1_1,
                has_b1_0,
                has_b1_1,
                has_annotation_b1,
                has_annotation_b0,
                tm_non_tumor_heatmap,
                has_tm_non_tumor_heatmap,
            )

        heatmap_0 = torch.zeros(features.shape[0]+1)
        heatmap_1 = torch.zeros(features.shape[0]+1)
        has_heatmap_0 = 0
        has_heatmap_1 = 0
        annotation_heatmap = self._build_annotation_heatmap(slide_id, features.shape[0])
        if annotation_heatmap is not None:
            heatmap_0, heatmap_1, has_heatmap_0, has_heatmap_1 = self._apply_annotation_heatmap(
                label, annotation_heatmap
            )
        elif self.attn_code[0] == 1 and label == 0 and os.path.exists(heatmap_path_label_0_0) :
            # print(f"Loading heatmap for slide_id: {slide_id}")
            heatmap_0 = np.load(heatmap_path_label_0_0)
            heatmap_0 = torch.from_numpy(heatmap_0).float().squeeze(0)   # [H+1]
            cls_score_0 = heatmap_0[0:1]     # [1]
            patch_scores_0 = heatmap_0[1:]   # [H]                      
            has_heatmap_0 = 1

        if annotation_heatmap is None and self.attn_code[1] == 1 and label == 0 and os.path.exists(heatmap_path_label_0_1):
            heatmap_1 = np.load(heatmap_path_label_0_1)
            heatmap_1 = torch.from_numpy(heatmap_1).float().squeeze(0)   # [H+1]
            cls_score_1 = heatmap_1[0:1]     # [1]
            patch_scores_1 = heatmap_1[1:]   # [H]                      
            has_heatmap_1 = 1

        if annotation_heatmap is None and self.attn_code[2] == 1 and label == 1 and os.path.exists(heatmap_path_label_1_0) :
            # print(f"Loading heatmap for slide_id: {slide_id}")
            heatmap_0 = np.load(heatmap_path_label_1_0)
            heatmap_0 = torch.from_numpy(heatmap_0).float().squeeze(0)   # [H+1]
            cls_score_0 = heatmap_0[0:1]     # [1]
            patch_scores_0 = heatmap_0[1:]   # [H]                      
            has_heatmap_0 = 2

        if annotation_heatmap is None and self.attn_code[3] == 1 and label == 1 and os.path.exists(heatmap_path_label_1_1):
            heatmap_1 = np.load(heatmap_path_label_1_1)
            heatmap_1 = torch.from_numpy(heatmap_1).float().squeeze(0)   # [H+1]
            cls_score_1 = heatmap_1[0:1]     # [1]
            patch_scores_1 = heatmap_1[1:]   # [H]                      
            has_heatmap_1 = 2

        #----> shuffle
        if self.shuffle == True:
            perm = torch.randperm(features.shape[0])
            features = features.index_select(0, perm)
            if has_heatmap_0 != 0:
                cls_score_0 = heatmap_0[0:1]
                patch_scores_0 = heatmap_0[1:]
                if patch_scores_0.shape[0] != perm.shape[0]:
                    raise ValueError(
                        f'Heatmap 0 length mismatch for {slide_id}: '
                        f'{patch_scores_0.shape[0]} scores for {perm.shape[0]} patches'
                    )
                patch_scores_0 = patch_scores_0.index_select(0, perm)
                heatmap_0 = torch.cat([cls_score_0, patch_scores_0], dim=0)
            if has_heatmap_1 != 0:
                cls_score_1 = heatmap_1[0:1]
                patch_scores_1 = heatmap_1[1:]
                if patch_scores_1.shape[0] != perm.shape[0]:
                    raise ValueError(
                        f'Heatmap 1 length mismatch for {slide_id}: '
                        f'{patch_scores_1.shape[0]} scores for {perm.shape[0]} patches'
                    )
                patch_scores_1 = patch_scores_1.index_select(0, perm)
                heatmap_1 = torch.cat([cls_score_1, patch_scores_1], dim=0)

        return features, label, slide_id, heatmap_0, heatmap_1, has_heatmap_0, has_heatmap_1  ### addded slide_id, heatmap, has_heatmap 


