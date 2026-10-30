import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from nystrom_attention_modif import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head_fusion='mean', return_attn=True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            return_attn= return_attn,
            head_fusion='max',
        )

    def forward(self, x, return_attn=None):
        attn_out = self.attn(self.norm(x), return_attn=return_attn)
        if isinstance(attn_out, tuple):
            out, attn = attn_out
        else:
            out, attn = attn_out, None
        x = x + out
        return x, attn
class TransLayer1(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head_fusion='mean'):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            # return_attn= True,
            # head_fusion='max',
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(
        self,
        n_classes,
        use_attention_loss=True,
        dual_branch=False,
        use_gradient_checkpointing=False,
        gradient_checkpointing_min_patches=30000,
    ):
        super(TransMIL, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1536, 512), nn.ReLU()) #### self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))   ##### class token randomly generated
        self.dual_branch = dual_branch
        self.n_classes = 2 if dual_branch else n_classes
        self.use_attention_loss = use_attention_loss
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_checkpointing_min_patches = int(gradient_checkpointing_min_patches or 0)
        self.pos_layer = PPEG(dim=512)
        self.layer1 = TransLayer(dim=512, return_attn=use_attention_loss)
        self.layer2 = TransLayer(dim=512, return_attn=use_attention_loss)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        if self.dual_branch:
            self.pos_layer_b1 = PPEG(dim=512)
            self.layer1_b1 = TransLayer(dim=512, return_attn=use_attention_loss)
            self.layer2_b1 = TransLayer(dim=512, return_attn=use_attention_loss)
            self.norm_b1 = nn.LayerNorm(512)
            self._fc2_b1 = nn.Linear(512, self.n_classes)

    def _add_cls_and_pad(self, h):
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1) #[B, N, 512]

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        return h, H, _H, _W

    def _should_checkpoint(self, h):
        return (
            self.training
            and self.use_gradient_checkpointing
            and self.gradient_checkpointing_min_patches > 0
            and h.shape[1] >= self.gradient_checkpointing_min_patches
            and h.requires_grad
        )

    def _run_layer(self, layer, h, return_attn):
        if not self._should_checkpoint(h):
            return layer(h, return_attn=return_attn)
        if return_attn:
            return checkpoint(lambda x: layer(x, return_attn=True), h, use_reentrant=True)
        h = checkpoint(lambda x: layer(x, return_attn=False)[0], h, use_reentrant=True)
        return h, None

    def _run_pos_layer(self, pos_layer, h, grid_h, grid_w):
        if not self._should_checkpoint(h):
            return pos_layer(h, grid_h, grid_w)
        return checkpoint(lambda x: pos_layer(x, grid_h, grid_w), h, use_reentrant=True)

    def _forward_branch(self, h, H, grid_h, grid_w, layer1, pos_layer, layer2, norm, classifier, return_attn):
        h, attn0 = self._run_layer(layer1, h, return_attn) #[B, N, 512]
        h = self._run_pos_layer(pos_layer, h, grid_h, grid_w) #[B, N, 512]
        h, attn1 = self._run_layer(layer2, h, return_attn) #[B, N, 512]
        h = norm(h)[:, 0]

        logits = classifier(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        if return_attn:
            attns = [attn0[..., :H+1], attn1[..., :H+1]]
        else:
            attns = []
        return results_dict, attns, h

    def forward(self, **kwargs):
        return_attn = kwargs.get('return_attn', self.use_attention_loss)
        if isinstance(return_attn, dict):
            return_attn_b0 = bool(return_attn.get('branch0', False))
            return_attn_b1 = bool(return_attn.get('branch1', False))
        else:
            return_attn_b0 = bool(return_attn)
            return_attn_b1 = bool(return_attn)

        h = kwargs['data'].float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        h, H, _H, _W = self._add_cls_and_pad(h)

        branch0 = self._forward_branch(
            h.clone(), H, _H, _W,
            self.layer1, self.pos_layer, self.layer2, self.norm, self._fc2,
            return_attn_b0,
        )
        if not self.dual_branch:
            return branch0

        branch1 = self._forward_branch(
            h.clone(), H, _H, _W,
            self.layer1_b1, self.pos_layer_b1, self.layer2_b1, self.norm_b1, self._fc2_b1,
            return_attn_b1,
        )
        results_dict = {
            'branch0': branch0[0],
            'branch1': branch1[0],
            'logits': branch0[0]['logits'],
            'Y_prob': branch0[0]['Y_prob'],
            'Y_hat': branch0[0]['Y_hat'],
        }
        attns = {'branch0': branch0[1], 'branch1': branch1[1]}
        features = {'branch0': branch0[2], 'branch1': branch1[2]}
        return results_dict, attns, features

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
