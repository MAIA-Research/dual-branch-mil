import torch.nn as nn

from .TransMIL import TransMIL


class JointTransMIL(nn.Module):
    def __init__(self, use_gradient_checkpointing=False, gradient_checkpointing_min_patches=30000):
        super().__init__()
        options = dict(
            use_attention_loss=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_min_patches=gradient_checkpointing_min_patches,
        )
        self.single = TransMIL(n_classes=3, dual_branch=False, **options)
        self.dual = TransMIL(n_classes=2, dual_branch=True, **options)

    def forward(self, data, return_attn=True):
        return {
            'single': self.single(data=data, return_attn=return_attn),
            'dual': self.dual(data=data, return_attn=return_attn),
        }
