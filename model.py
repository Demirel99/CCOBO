#model.py
"""
Model definitions including VGG19 Encoder, FPN Decoder, ASPP, and PSF Head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import from config
from config import PSF_HEAD_TEMP, MODEL_INPUT_SIZE # MODEL_INPUT_SIZE is imported but not directly used for layer definitions

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        # 1x1 conv
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        # Atrous convs
        for rate in rates[1:]:
             self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                        padding=rate, dilation=rate, bias=False))

        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True) # ReLU after GAP conv
        )

        # Batch norm for each branch
        self.bn_ops = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(len(self.convs) + 1)]) # +1 for GAP

        # Final 1x1 conv and dropout
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.convs) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), # BN after final projection
            nn.ReLU(inplace=True),
            nn.Dropout(0.2) # Consider placing dropout after ReLU
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []
        # Parallel convolutions
        for i, conv in enumerate(self.convs):
            features.append(F.relu(self.bn_ops[i](conv(x)))) # ReLU after BN
        # Global pooling
        gap_feat = self.global_pool(x)
        gap_feat = F.interpolate(gap_feat, size=size, mode='bilinear', align_corners=False)
        features.append(self.bn_ops[-1](gap_feat)) # BN for GAP feature

        # Concatenate and project
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class VGG19Encoder(nn.Module):
    """Encodes an image using VGG19 features at multiple scales."""
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg19.features)
        # Indices corresponding to VGG stages output (after ReLU/MaxPool)
        # VGG19: C1(64)@idx3, C2(128)@idx8, C3(256)@idx17, C4(512)@idx26, C5(512)@idx35
        # Spatial sizes for 224x224 input:
        # C1: 224x224, C2: 112x112, C3: 56x56, C4: 28x28, C5: 14x14 (before 5th pool)
        self.feature_layers = nn.ModuleList(features)
        self.capture_indices = {3, 8, 17, 26, 35}

    def forward(self, x):
        results = {}
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.capture_indices:
                 # Assign based on index to standard names C1-C5
                 if i == 3: results['C1'] = x
                 elif i == 8: results['C2'] = x
                 elif i == 17: results['C3'] = x
                 elif i == 26: results['C4'] = x
                 elif i == 35: results['C5'] = x
        # Return features C1..C5 in a list, shallow to deep
        return [results['C1'], results['C2'], results['C3'], results['C4'], results['C5']]

class SmallPSFEncoder(nn.Module):
    """Encodes the 1-channel input PSF mask."""
    def __init__(self):
        super(SmallPSFEncoder, self).__init__()
        # For input 224x224:
        # 224->112 (pool1)
        # 112->56  (pool2)
        # 56->28   (pool3)
        # 28->14   (pool4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(inplace=True) # Keep size (e.g., 14x14 for 224 input)
        )
        # Output: (B, 64, H_in/16, W_in/16)

    def forward(self, x):
        return self.encoder(x)

class FPNDecoder(nn.Module):
    """Feature Pyramid Network (FPN) decoder."""
    def __init__(self, encoder_channels=[64, 128, 256, 512, 512], fpn_channels=256, out_channels=64):
        super(FPNDecoder, self).__init__()
        # Assuming C1, C2, C3, C4, C5_effective channels are passed in encoder_channels
        assert len(encoder_channels) == 5, "Expected 5 encoder channel numbers for C1 to C5_effective."

        self.lateral_convs = nn.ModuleList()
        # Lateral connections for C5_eff, C4, C3, C2, C1 (reversed order processing)
        for enc_ch in reversed(encoder_channels): # Process from deep (C5_eff) to shallow (C1)
            self.lateral_convs.append(nn.Conv2d(enc_ch, fpn_channels, kernel_size=1))

        self.smooth_convs = nn.ModuleList()
        # Smoothing layers for P5, P4, P3, P2, P1
        for _ in range(len(encoder_channels)): # One smooth layer per pyramid level
             self.smooth_convs.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1))

        # Final convolution to get desired output channels from the finest level (P1)
        # P1 should be at the same resolution as C1 (i.e., MODEL_INPUT_SIZE)
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, top_down_feat, lateral_feat):
        """Upsamples top_down_feat and adds lateral_feat."""
        _, _, H, W = lateral_feat.shape
        # Upsample the feature map from the coarser layer (top_down_feat)
        upsampled_feat = F.interpolate(top_down_feat, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_feat + lateral_feat

    def forward(self, x_top, encoder_features_c1_c4):
        """
        Args:
            x_top (Tensor): Feature map from the bottleneck (e.g., after ASPP, corresponds to C5_effective scale).
            encoder_features_c1_c4 (list[Tensor]): List [C1, C2, C3, C4]. Order: shallow to deep.
        """
        C1, C2, C3, C4 = encoder_features_c1_c4
        # Combine all features in C1..C5_effective order for easier indexing later
        all_features = [C1, C2, C3, C4, x_top] # Indices 0..4 correspond to C1..C5_eff

        pyramid_features = [] # This will store [P5, P4, P3, P2, P1]

        # P5 (from x_top/C5_effective) - Index -1 or 4 in all_features
        p = self.lateral_convs[0](all_features[-1]) # lateral_convs[0] corresponds to C5_eff
        p = self.smooth_convs[0](p)                # smooth_convs[0] corresponds to P5
        pyramid_features.append(p) # P5 added

        # P4 to P1 (Indices i=1, 2, 3, 4 for lateral_convs/smooth_convs)
        for i in range(1, len(self.lateral_convs)):
            # Index into all_features for C4, C3, C2, C1
            # i=1 -> lateral_idx=3 (C4)
            # i=2 -> lateral_idx=2 (C3)
            # i=3 -> lateral_idx=1 (C2)
            # i=4 -> lateral_idx=0 (C1)
            lateral_idx = len(all_features) - 1 - i
            lateral_feat = self.lateral_convs[i](all_features[lateral_idx])

            # Get the output from the previous P-level (the one just added to pyramid_features)
            p_prev = pyramid_features[-1]
            top_down_feat = self._upsample_add(p_prev, lateral_feat) # Upsample p_prev and add lateral
            p = self.smooth_convs[i](top_down_feat) # Apply smoothing to the sum
            pyramid_features.append(p) # Add P4, P3, P2, P1

        # The pyramid features are [P5, P4, P3, P2, P1]
        p1_output = pyramid_features[-1] # P1 is the last element (finest resolution)

        # Final 3x3 conv on P1 to get desired output channels
        out = F.relu(self.final_conv(p1_output)) # Add ReLU. Shape: (B, out_channels, H, W), where H,W = MODEL_INPUT_SIZE

        return out # Output shape should be (B, out_channels, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


class PSFHead(nn.Module):
    """Predicts the PSF map from the final decoder features."""
    def __init__(self, in_channels, temperature=PSF_HEAD_TEMP):
        super(PSFHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1) # Output 1 channel
        )
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1) # Softmax over spatial dimensions (H*W)

    def forward(self, x):
        x = self.head(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1) # Reshape for spatial softmax: (B, C, H*W)
        if self.temperature > 1e-6:
             x = x / self.temperature # Apply temperature sharpening
        x = self.softmax(x) # Apply softmax over the spatial dimension
        return x.view(b, c, h, w) # Reshape back


class VGG19FPNASPP(nn.Module):
    """The main model combining VGG19, ASPP, FPN, and PSF Head."""
    def __init__(self):
        super(VGG19FPNASPP, self).__init__()
        # Encoders
        # For MODEL_INPUT_SIZE = 224x224:
        # VGG19Encoder outputs [C1(1/1), C2(1/2), C3(1/4), C4(1/8), C5(1/16)]
        # e.g., C5 is (B, 512, 14, 14) for 224x224 input.
        self.image_encoder = VGG19Encoder()
        # SmallPSFEncoder outputs features at 1/16 scale (e.g., 14x14 for 224x224 input)
        self.mask_encoder = SmallPSFEncoder()

        # VGG channel info
        vgg_c1_channels = 64
        vgg_c2_channels = 128
        vgg_c3_channels = 256
        vgg_c4_channels = 512
        vgg_c5_channels = 512
        mask_features_channels = 64 # Output of SmallPSFEncoder

        # Fusion layer (fuse C5 and mask_features directly as they have the same 1/16 scale)
        fusion_in_channels_c5 = vgg_c5_channels + mask_features_channels # 512 + 64 = 576
        fusion_out_channels_c5 = 512 # Project back to 512 for consistency
        self.fusion_conv_c5 = nn.Sequential(
            nn.Conv2d(fusion_in_channels_c5, fusion_out_channels_c5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_channels_c5),
            nn.ReLU(inplace=True)
        )

        # ASPP at C5 level (operates on 1/16 scale features)
        self.aspp_c5 = ASPP(in_channels=fusion_out_channels_c5, out_channels=fusion_out_channels_c5) # 512 -> 512

        # FPN Decoder
        fpn_encoder_channels = [vgg_c1_channels, vgg_c2_channels, vgg_c3_channels, vgg_c4_channels, fusion_out_channels_c5]
        self.fpn_decoder = FPNDecoder(
             encoder_channels=fpn_encoder_channels,
             fpn_channels=256, # Internal FPN channels
             out_channels=64   # Channels before PSF head
         )

        # Final PSF Head
        self.psf_head = PSFHead(in_channels=64, temperature=PSF_HEAD_TEMP)


    def forward(self, image, mask):
        # Image normalization happens in dataset.py

        # Ensure mask has channel dimension
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # 1. Encode Image and Mask
        encoder_features = self.image_encoder(image) # [C1, C2, C3, C4, C5]
        C1, C2, C3, C4, C5 = encoder_features
        # For 224x224 input:
        # C5 shape (B, 512, H/16, W/16) e.g., (B, 512, 14, 14)
        mask_features = self.mask_encoder(mask)
        # mask_features shape (B, 64, H/16, W/16) e.g., (B, 64, 14, 14)

        # 2. Fuse C5 and Mask Features (at 1/16 scale)
        fused_features = torch.cat([C5, mask_features], dim=1) # Shape: (B, 512+64, 14, 14) for 224 input
        fused_c5 = self.fusion_conv_c5(fused_features)        # Shape: (B, 512, 14, 14) for 224 input

        # 3. Apply ASPP to Fused Bottleneck Features
        aspp_output = self.aspp_c5(fused_c5) # Effective C5 for FPN. Shape: (B, 512, 14, 14) for 224 input

        # 4. Decode using FPN
        # Decoder output will be at MODEL_INPUT_SIZE (e.g., 224x224)
        decoder_output = self.fpn_decoder(aspp_output, [C1, C2, C3, C4])

        # 5. Predict PSF Map
        psf_map = self.psf_head(decoder_output) # Output shape (B, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        return psf_map
