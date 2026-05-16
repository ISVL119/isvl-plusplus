import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualSegHead(nn.Module):
    def __init__(self, in_dims, hidden_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim + 1, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for in_dim in in_dims
        ])

        fuse_dim = hidden_dim * len(in_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        up_dims = [
            hidden_dim,
            max(hidden_dim // 2, 64),
            max(hidden_dim // 4, 32),
            max(hidden_dim // 8, 16),
            max(hidden_dim // 16, 8),
        ]
        self.up_blocks = nn.ModuleList([
            UpConv(up_dims[i], up_dims[i + 1]) for i in range(len(up_dims) - 1)
        ])
        self.out_conv = nn.Conv2d(up_dims[-1], 1, kernel_size=1, stride=1)
        self.total_upscale = 2 ** len(self.up_blocks)

    def _validate_out_size(self, base_size, out_size):
        expected_h = base_size[0] * self.total_upscale
        expected_w = base_size[1] * self.total_upscale
        if out_size[0] != expected_h or out_size[1] != expected_w:
            raise ValueError(
                f"ResidualSegHead expects out_size={(expected_h, expected_w)} from base_size={base_size}, "
                f"but got out_size={out_size}."
            )

    def forward(self, residual_feats, out_size):
        base_size = residual_feats[0].shape[-2:]
        self._validate_out_size(base_size, out_size)

        processed = []
        for feat, block in zip(residual_feats, self.proj):
            if feat.shape[-2:] != base_size:
                raise ValueError(
                    f"All residual feature maps must share the same spatial size. "
                    f"Got {feat.shape[-2:]} vs base_size={base_size}."
                )
            processed.append(block(feat))

        x = torch.cat(processed, dim=1)
        x = self.fuse(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.out_conv(x)
        return x


class INP_Former(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        aggregation,
        decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        remove_class_token=False,
        encoder_require_grad_layer=[],
        prototype_token=None,
        residual_head=None,
        eval_seg_weight=0.35,
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        self.residual_head = residual_head
        self.eval_seg_weight = float(eval_seg_weight)
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def _forward_vit(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []

        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue

            if i in self.target_layers:
                en_list.append(x)

        feat_h = feat_w = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
        return x, en_list, feat_h, feat_w

    def _forward_dinov3_official(self, x):
        with torch.no_grad():
            en_list = list(
                self.encoder.get_intermediate_layers(
                    x,
                    n=self.target_layers,
                    reshape=False,
                    return_class_token=False,
                    return_extra_tokens=False,
                    norm=False,
                )
            )
        feat_h = x.shape[-2] // self.encoder.patch_size
        feat_w = x.shape[-1] // self.encoder.patch_size
        return en_list, feat_h, feat_w

    def _forward_dinov3_exact(self, x):
        x, (feat_h, feat_w) = self.encoder.prepare_tokens_with_masks(x)
        en_list = []
        num_storage_tokens = getattr(
            self.encoder,
            'n_storage_tokens',
            getattr(self.encoder, 'num_register_tokens', 0),
        )

        rope_sincos = None
        if getattr(self.encoder, 'rope_embed', None) is not None:
            rope_sincos = self.encoder.rope_embed(H=feat_h, W=feat_w)

        for i, blk in enumerate(self.encoder.blocks):
            if i > self.target_layers[-1]:
                break

            if i in self.encoder_require_grad_layer:
                x = blk(x, rope_sincos)
            else:
                with torch.no_grad():
                    x = blk(x, rope_sincos)

            if i in self.target_layers:
                out = self.encoder.norm(x)
                out = out[:, 1 + num_storage_tokens:, :]
                en_list.append(out)

        return en_list, feat_h, feat_w

    def _extract_tokens(self, x):
        use_dinov3 = getattr(self.encoder, 'is_dinov3', True)
        if use_dinov3:
            if len(self.encoder_require_grad_layer) == 0:
                en_list, feat_h, feat_w = self._forward_dinov3_official(x)
            else:
                en_list, feat_h, feat_w = self._forward_dinov3_exact(x)
        else:
            x, en_list, feat_h, feat_w = self._forward_vit(x)
            if self.remove_class_token:
                en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
        return en_list, feat_h, feat_w, use_dinov3

    def _decode_from_tokens(self, fused_tokens, en_list, feat_h, feat_w, use_dinov3):
        B = fused_tokens.shape[0]

        agg_prototype = self.prototype_token
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), fused_tokens)

        g_loss = self.gather_loss(fused_tokens, agg_prototype)

        x = fused_tokens
        for blk in self.bottleneck:
            x = blk(x)

        de_list = []
        for blk in self.decoder:
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if (not use_dinov3) and (not self.remove_class_token):
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([B, -1, feat_h, feat_w]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, feat_h, feat_w]).contiguous() for d in de]
        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def build_residual_features(self, en, de, detach_input=True):
        residual_feats = []
        for e, d in zip(en, de):
            if detach_input:
                e = e.detach()
                d = d.detach()
            abs_res = torch.abs(e - d)
            cos_res = 1 - F.cosine_similarity(e, d, dim=1).unsqueeze(1)
            residual_feats.append(torch.cat([abs_res, cos_res], dim=1))
        return residual_feats

    def forward(self, x, return_seg=False, seg_out_size=None, detach_seg_input=True):
        en_list, feat_h, feat_w, use_dinov3 = self._extract_tokens(x)
        fused_tokens = self.fuse_feature(en_list)
        en, de, g_loss = self._decode_from_tokens(fused_tokens, en_list, feat_h, feat_w, use_dinov3)

        seg_logits = None
        if return_seg and self.residual_head is not None:
            if seg_out_size is None:
                seg_out_size = x.shape[-2:]
            residual_feats = self.build_residual_features(en, de, detach_input=detach_seg_input)
            seg_logits = self.residual_head(residual_feats, out_size=seg_out_size)
        return en, de, g_loss, seg_logits

    def forward_seg(self, x, seg_out_size=None, freeze_backbone=True):
        if self.residual_head is None:
            return None
        if seg_out_size is None:
            seg_out_size = x.shape[-2:]
        if freeze_backbone:
            with torch.no_grad():
                en_list, feat_h, feat_w, use_dinov3 = self._extract_tokens(x)
                fused_tokens = self.fuse_feature(en_list)
                en, de, _ = self._decode_from_tokens(fused_tokens, en_list, feat_h, feat_w, use_dinov3)
            residual_feats = self.build_residual_features(en, de, detach_input=False)
        else:
            en, de, _, _ = self.forward(x, return_seg=False)
            residual_feats = self.build_residual_features(en, de, detach_input=False)
        return self.residual_head(residual_feats, out_size=seg_out_size)