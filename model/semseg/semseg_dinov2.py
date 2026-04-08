import torch
from torch import nn
import torch.nn.functional as F

import random
import numpy as np
from copy import deepcopy
from util.utils import obtain_cutmix, apply_cutmix

from model.backbone.dinov2 import DINOv2, DPTHead


dinov2_cfgs = {
    'small': {
        'features': 64,
        'intermediate_layer_idx': [2, 5, 8, 11],
        'out_channels': [48, 96, 192, 384]
    },
    'base': {
        'features': 128,
        'intermediate_layer_idx': [2, 5, 8, 11],
        'out_channels': [96, 192, 384, 768]
    },
    'large': {
        'features': 256,
        'intermediate_layer_idx': [4, 11, 17, 23],
        'out_channels': [256, 512, 1024, 1024]
    },
    'giant': {
        'features': 384,
        'intermediate_layer_idx': [9, 19, 29, 39],
        'out_channels': [1536, 1536, 1536, 1536]
    }
}




class RGB_HGT_Fusion(nn.Module):
    def __init__(self, dims, fusion_method='sum'):
        super(RGB_HGT_Fusion, self).__init__()
        self.fusion_method = fusion_method
        self.dims = dims

        if fusion_method == 'concat':
            self.fus_weight_layers = nn.ModuleList([
                nn.Linear(dims[i] + dims[i], dims[i])
                for i in range(len(dims))
            ])

        elif fusion_method == 'cross_attention':
            # Define cross-attention layers and final linear layers for each scale
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=dims[i], num_heads=1, batch_first=True)
                for i in range(len(dims))
            ])
            self.fus_weight_layers = nn.ModuleList([
                nn.Linear(dims[i], dims[i])
                for i in range(len(dims))
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(dims[i]) for i in range(len(dims))
            ])

    def forward(self, rgb_feats, hgt_feats):
        rgb_feats = [rgb.detach() for rgb in rgb_feats]

        fused_features = []
        for i, items in enumerate(zip(rgb_feats, hgt_feats)):
            rgb, hgt = items

            if self.fusion_method == 'sum':
                fused = rgb + hgt

            elif self.fusion_method == 'concat':
                fused = torch.cat([rgb, hgt], dim=2)  # B x N x 2C
                fused = self.fus_weight_layers[i](fused) 

            elif self.fusion_method == 'cross_attention':
                fused, _ = self.cross_attention_layers[i](query=rgb, key=hgt, value=hgt)
                fused = self.fus_weight_layers[i](fused)  
                fused = self.norm_layers[i](rgb + fused)

            fused_features.append(fused)
        return fused_features



class Build_RGB_HGT_DinoV2(nn.Module):
    def __init__(self, n_classes, encoder_type='small'):
        super(Build_RGB_HGT_DinoV2, self).__init__()
    
        checkpoint = torch.load(
            f'./pretrained/dinov2_{encoder_type}.pth', 
            weights_only=True
        )        
        self.use_bn = False
        self.features = dinov2_cfgs[encoder_type]['features']
        self.intermediate_layer_idx = dinov2_cfgs[encoder_type]['intermediate_layer_idx']
        self.out_channels = [dinov2_cfgs[encoder_type]['out_channels'][-1] \
                            for i in range(len(dinov2_cfgs[encoder_type]['out_channels']))]
        self.multiplier = 14

        encoder = DINOv2(encoder_type)
        decoder = DPTHead(
            n_classes,
            encoder.embed_dim,
            self.features,
            self.use_bn,
            out_channels=self.out_channels
        )

        self.teacher = nn.ModuleDict({
            'encoder_rgb': deepcopy(encoder),
            'decoder_rgb': deepcopy(decoder),
            'encoder_hgt': deepcopy(encoder),
            'feature_fus': RGB_HGT_Fusion(
                                self.out_channels, 
                                fusion_method='sum'
                            ),
            'decoder_fus': deepcopy(decoder),
        })
        self.student = nn.ModuleDict({
            key: deepcopy(module) for key, module in self.teacher.items()
        })
        self.teacher['encoder_rgb'].load_state_dict(checkpoint)
        self.teacher['encoder_hgt'].load_state_dict(checkpoint)
        self.student['encoder_rgb'].load_state_dict(checkpoint)
        self.student['encoder_hgt'].load_state_dict(checkpoint)

    def forward(self, rgb=None, hgt=None, mode='val'):
        if mode == 'val':
            if rgb is not None and hgt is None:
                h, w = rgb.shape[-2:]
                feat_rgb = self.get_feat(self.teacher['encoder_rgb'], rgb)
                pred_rgb = self.get_pred(self.teacher['decoder_rgb'], feat_rgb, size=[h, w])
                return pred_rgb

            if rgb is None and hgt is not None:
                h, w = hgt.shape[-2:]
                feat_hgt = self.get_feat(self.teacher['encoder_hgt'], hgt)
                pred_hgt = self.get_pred(self.teacher['decoder_fus'], feat_hgt, size=[h, w])
                return pred_hgt

            if rgb is not None and hgt is not None:
                h, w = rgb.shape[-2:]
                feat_rgb = self.get_feat(self.teacher['encoder_rgb'], rgb)
                feat_hgt = self.get_feat(self.teacher['encoder_hgt'], hgt)
                feat_fus = self.fus_feat(self.teacher['feature_fus'], feat_rgb, feat_hgt)
                pred_fus = self.get_pred(self.teacher['decoder_fus'], feat_fus, size=[h, w])
                return pred_fus

        elif mode == 'train':
            h, w = rgb.size()[-2:]

            rgb_l, rgb_w, rgb_s = rgb.chunk(3)
            hgt_l, hgt_w, hgt_s = hgt.chunk(3)

            # --------- labeled prediction ----------#
            feat_l_rgb = self.get_feat(self.student['encoder_rgb'], rgb_l)
            pred_l_rgb = self.get_pred(self.student['decoder_rgb'], feat_l_rgb, size=[h, w])

            feat_l_hgt = self.get_feat(self.student['encoder_hgt'], hgt_l)
            feat_l_fus = self.fus_feat(self.student['feature_fus'], feat_l_rgb, feat_l_hgt)
            pred_l_fus = self.get_pred(self.student['decoder_fus'], feat_l_fus, size=[h, w])

            # --------- unlabeled prediction --------#
            coord_mix, indice_mix = obtain_cutmix(rgb_w)

            # ----- weakly-unlabeled prediction -----#
            self.eval()
            with torch.no_grad():
                feat_w_rgb = self.get_feat(self.teacher['encoder_rgb'], rgb_w)
                pred_w_rgb = self.get_pred(self.teacher['decoder_rgb'], feat_w_rgb, size=[h, w])

                feat_w_hgt = self.get_feat(self.teacher['encoder_hgt'], hgt_w)
                feat_w_fus = self.fus_feat(self.teacher['feature_fus'], feat_w_rgb, feat_w_hgt)
                pred_w_fus = self.get_pred(self.teacher['decoder_fus'], feat_w_fus, size=[h, w])

                pred_w_rgb_mixed = apply_cutmix(pred_w_rgb, coord_mix, indice_mix).detach()         
                pred_w_fus_mixed = apply_cutmix(pred_w_fus, coord_mix, indice_mix).detach()         
            self.train()

            # ---- strongly-unlabeled prediction ----#
            rgb_s_mixed = apply_cutmix(rgb_s, coord_mix, indice_mix)
            hgt_s_mixed = apply_cutmix(hgt_s, coord_mix, indice_mix)

            feat_s_rgb_mixed = self.get_feat(self.student['encoder_rgb'], rgb_s_mixed)
            pred_s_rgb_mixed = self.get_pred(self.student['decoder_rgb'], feat_s_rgb_mixed, size=[h, w])

            feat_s_hgt_mixed = self.get_feat(self.student['encoder_hgt'], hgt_s_mixed)
            feat_s_fus_mixed = self.fus_feat(self.student['feature_fus'], feat_s_rgb_mixed, feat_s_hgt_mixed)
            pred_s_fus_mixed = self.get_pred(self.student['decoder_fus'], feat_s_fus_mixed, size=[h, w])

            return pred_l_rgb, pred_l_fus, \
                   pred_w_rgb_mixed, pred_w_fus_mixed, \
                   pred_s_rgb_mixed, pred_s_fus_mixed

    def get_feat(self, encoder, rgb):
        rgb_h, rgb_w = rgb.shape[-2:]
        new_h = int(rgb_h / self.multiplier + 0.5) * self.multiplier 
        new_w = int(rgb_w / self.multiplier + 0.5) * self.multiplier
        rgb = self.upsample(rgb, (new_h, new_w))
        feats = encoder.get_intermediate_layers(
            rgb, self.intermediate_layer_idx
        )
        return feats

    def fus_feat(self, fusion, rgb_feats, hgt_feats):
        fused_feats = fusion(rgb_feats, hgt_feats)
        return fused_feats

    def get_pred(self, decoder, feats, size):
        rgb_h, rgb_w = size
        pad_h = int(rgb_h / self.multiplier + 0.5) 
        pad_w = int(rgb_w / self.multiplier + 0.5)
        pred = decoder(feats, pad_h, pad_w)
        pred = self.upsample(pred, size=(rgb_h, rgb_w))   
        return pred

    def upsample(self, x, size, mode='bilinear'):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            x = F.interpolate(x, size=size, mode=mode, align_corners=(mode == 'bilinear'))
            return x.squeeze(1)
        elif len(x.shape) == 4:
            return F.interpolate(x, size=size, mode=mode, align_corners=(mode == 'bilinear'))
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    def get_1x_lr_params(self):
        params = nn.ParameterList()
        for key in self.student.keys():
            if 'encoder' in key:
                params.extend(self.student[key].parameters())
        return params

    def get_mx_lr_params(self):
        params = nn.ParameterList()
        for key in self.student.keys():
            if 'encoder' not in key:
                params.extend(self.student[key].parameters())
        return params

    def update_ema(self, decay=0.995):
        if set(self.teacher.keys()) != set(self.student.keys()):
            raise ValueError("Mismatch between keys in teacher and student modules.")

        for key in self.teacher.keys():  
            for stu_param, tec_param in zip(self.student[key].parameters(), self.teacher[key].parameters()):
                tec_param.data = decay * tec_param.data + (1.0 - decay) * stu_param.data



class Build_RGB_DinoV2(nn.Module):
    def __init__(self, n_classes, encoder_type='small'):
        super(Build_RGB_DinoV2, self).__init__()

        checkpoint = torch.load(
            f'./pretrained/dinov2_{encoder_type}.pth', 
            weights_only=True
        )

        self.features = dinov2_cfgs[encoder_type]['features']    
        self.intermediate_layer_idx = dinov2_cfgs[encoder_type]['intermediate_layer_idx']    
        self.out_channels = dinov2_cfgs[encoder_type]['out_channels']    
        self.use_bn = False
        self.multiplier = 14

        encoder = DINOv2(encoder_type)
        embed_dim =encoder.embed_dim
        decoder = DPTHead(
            n_classes, 
            embed_dim, 
            self.features, 
            self.use_bn, 
            out_channels=self.out_channels
        )

        self.stu_encoder = deepcopy(encoder)
        self.stu_encoder.load_state_dict(checkpoint)
        self.stu_decoder = deepcopy(decoder)

        self.tec_encoder = deepcopy(encoder)
        self.tec_encoder.load_state_dict(checkpoint)
        self.tec_decoder = deepcopy(decoder)

    def forward(self, img=None, mode='val'):
        if mode == 'val':
            h, w = img.shape[-2:]
            feat = self.get_feat(self.tec_encoder, img)
            pred = self.get_pred(self.tec_decoder, feat, size=[h, w])
            return pred

        elif mode == 'supervised':
            h, w = img.shape[-2:]
            feat = self.get_feat(self.stu_encoder, img)
            pred = self.get_pred(self.stu_decoder, feat, size=[h, w])
            return pred

        elif mode == 'train':
            h, w = img.size()[-2:]
            img_l, img_w, img_s = img.chunk(3)

            coord_mix, indice_mix = obtain_cutmix(img_w)
            img_s_mixed = apply_cutmix(img_s, coord_mix, indice_mix)

            feat_l = self.get_feat(self.stu_encoder, img_l)
            pred_l = self.get_pred(self.stu_decoder, feat_l, [h, w])

            self.eval()
            with torch.no_grad():
                feat_w = self.get_feat(self.tec_encoder, img_w)
                pred_w = self.get_pred(self.tec_decoder, feat_w, [h, w]).detach()
                pred_w_mixed = apply_cutmix(pred_w, coord_mix, indice_mix).detach()
            self.train()

            feat_s_mixed = self.get_feat(self.stu_encoder, img_s_mixed)
            pred_s_mixed = self.get_pred(self.stu_decoder, feat_s_mixed, [h, w]) 
            return pred_l, pred_w_mixed, pred_s_mixed

    def get_feat(self, encoder, img):
        multiplier = 14
        img_h, img_w = img.shape[-2:]
        new_h = int(img_h / self.multiplier + 0.5) * self.multiplier 
        new_w = int(img_w / self.multiplier + 0.5) * self.multiplier
        img = self.upsample(img, (new_h, new_w))
        feats = encoder.get_intermediate_layers(
            img, self.intermediate_layer_idx
        )
        return feats

    def get_pred(self, decoder, feats, size):
        img_h, img_w = size
        pad_h = int(img_h / self.multiplier + 0.5) 
        pad_w = int(img_w / self.multiplier + 0.5)
        pred = decoder(feats, pad_h, pad_w)
        pred = self.upsample(pred, size=(img_h, img_w))        
        return pred

    def upsample(self, x, size, mode='bilinear'):
        if mode == 'bilinear':
            return F.interpolate(x, size=size, mode=mode, align_corners=True)
        elif mode == 'nearest':
            return F.interpolate(x, size=size, mode='nearest')

    def get_1x_lr_params(self):
        params = nn.ParameterList()
        params.extend(self.stu_encoder.parameters())
        return params

    def get_mx_lr_params(self):
        params = nn.ParameterList()
        params.extend(self.stu_decoder.parameters())
        return params

    def update_ema(self, decay=0.995):
        with torch.no_grad(): 
            for stu_param, tec_param in zip(self.stu_encoder.parameters(), self.tec_encoder.parameters()):
                tec_param.data = decay * tec_param.data + (1.0 - decay) * stu_param.data
            for stu_param, tec_param in zip(self.stu_decoder.parameters(), self.tec_decoder.parameters()):
                tec_param.data = decay * tec_param.data + (1.0 - decay) * stu_param.data







