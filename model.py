import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ASPP module: applies parallel 5x5 atrous (dilated) convolutions with different dilation rates
# to capture multi-scale context, and then fuses the results via a 1x1 convolution.
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            # For a 5x5 kernel, use padding = dilation * 2 to preserve spatial dimensions.
            padding = dilation * 2  
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=padding, dilation=dilation)
            )
        # Fuse multi-scale features by reducing the concatenated channels back to out_channels.
        self.project = nn.Conv2d(len(dilations) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply each atrous convolution in parallel.
        outs = [conv(x) for conv in self.convs]
        # Concatenate features along the channel dimension.
        x = torch.cat(outs, dim=1)
        # Project concatenated features to the desired number of channels.
        x = self.project(x)
        return x

# PixelShuffle upsampling module: learnable upsampling that rearranges feature maps
# to increase spatial resolution while preserving details.
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleUpsample, self).__init__()
        # The convolution expands channels by (upscale_factor^2) before pixel shuffling.
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

# PretrainedVGG16Saliency: a saliency prediction network that leverages a pre-trained VGG16
# for feature extraction, applies multi-scale context via ASPP, and progressively upsamples
# while incorporating a Gaussian center bias.
class PretrainedVGG16Saliency(nn.Module):
    def __init__(self, output_size=224, sigma_scale=0.5, n_bias=3):
        """
        Args:
            output_size: The desired size of the final saliency map (e.g., 224 for a 224x224 map).
            sigma_scale: Scale factor used in generating the Gaussian center bias.
            n_bias: Number of bias feature copies to concatenate (each with randomly weighted contributions).
        """
        super(PretrainedVGG16Saliency, self).__init__()
        self.output_size = output_size
        self.n_bias = n_bias

        # 1) Load a pre-trained VGG16 model as the feature extractor.
        #    Input is 224x224 and the output feature map is approximately (512, 7, 7).
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        for param in self.features.parameters():
            param.requires_grad = False

        # Pre-compute a fixed Gaussian center bias map of size (output_size x output_size).
        # This map will be resized later using bilinear interpolation.
        self.register_buffer("center_bias_map_full", self.compute_center_bias(output_size, output_size, sigma_scale))

        # 2) Apply an ASPP module to the VGG16 features to capture multi-scale context.
        #    The ASPP module outputs a feature map with 512 channels.
        self.aspp = ASPP(in_channels=512, out_channels=512, dilations=[1, 2, 4])

        # Block 1 (at 7x7 resolution):
        # - Resize the center bias map and generate n_bias randomly weighted bias features.
        # - Concatenate the ASPP output (512 channels) with these bias features, resulting in (512 + n_bias) channels.
        # - Apply a 5x5 atrous convolution (with dilation=2 and padding=4), ReLU activation, and BatchNorm.
        # - Upsample from 7x7 to 14x14 using PixelShuffle (the number of channels remains 512).
        self.atrous1 = nn.Conv2d(in_channels=512 + n_bias, out_channels=512, kernel_size=5, padding=4, dilation=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.upsample1 = PixelShuffleUpsample(in_channels=512, out_channels=512, upscale_factor=2)
        self.bn2 = nn.BatchNorm2d(512)

        # Block 2 (at 14x14 resolution):
        # - Resize the center bias map and generate another set of n_bias bias features.
        # - Concatenate the upsampled features (512 channels) with these bias features,
        #   giving an input of (512 + n_bias) channels.
        # - Apply a 5x5 atrous convolution (with dilation=2 and padding=4), ReLU, and BatchNorm.
        # - Upsample from 14x14 to 28x28 using PixelShuffle (channels remain 512).
        self.atrous2 = nn.Conv2d(in_channels=512 + n_bias, out_channels=512, kernel_size=5, padding=4, dilation=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.upsample2 = PixelShuffleUpsample(in_channels=512, out_channels=512, upscale_factor=2)
        self.bn4 = nn.BatchNorm2d(512)

        # 3) Final processing:
        # - Apply a 1x1 convolution to reduce the channel dimension to 1.
        # - Upsample the result to the final output size (e.g., 224x224) using bilinear interpolation.
        # - Apply a sigmoid function to normalize the saliency map values to [0, 1].
        self.conv_final = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

    def compute_center_bias(self, H, W, sigma_scale):
        """
        Generate a fixed Gaussian center bias map of size (H, W).
        """
        center_x, center_y = W / 2.0, H / 2.0
        sigma_x = W * sigma_scale / 2.0
        sigma_y = H * sigma_scale / 2.0
        xs = torch.arange(0, W).view(1, W).expand(H, W).float()
        ys = torch.arange(0, H).view(H, 1).expand(H, W).float()
        bias = torch.exp(-(((xs - center_x) ** 2) / (2 * sigma_x ** 2) +
                           ((ys - center_y) ** 2) / (2 * sigma_y ** 2)))
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        return bias.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    def forward(self, x):
        # Input x: (batch, 3, 224, 224)
        # 1) Extract features from the input image using the pre-trained VGG16.
        x_vgg = self.features(x)  # Expected shape: (batch, 512, 7, 7)
        
        # Apply the ASPP module to capture multi-scale context.
        x_aspp = self.aspp(x_vgg)  # Output shape: (batch, 512, 7, 7)

        # At 7x7 resolution:
        # - Resize the center bias map to 7x7.
        # - Generate n_bias randomly weighted bias features.
        bias_feature = F.interpolate(self.center_bias_map_full, size=x_aspp.shape[2:], mode='bilinear', align_corners=False)  # (1, 1, 7, 7)
        random_weights = torch.rand(x_aspp.size(0), self.n_bias, 1, 1, device=x.device)
        bias_feature = bias_feature.repeat(1, self.n_bias, 1, 1) * random_weights  # (batch, n_bias, 7, 7)
        # Concatenate ASPP features (512 channels) with bias features, resulting in (512 + n_bias) channels.
        x_cat = torch.cat([x_aspp, bias_feature], dim=1)  # (batch, 512 + n_bias, 7, 7)

        # Block 1:
        # - Apply an atrous convolution (5x5 kernel, dilation=2, padding=4), followed by ReLU and BatchNorm.
        x1 = F.relu(self.bn1(self.atrous1(x_cat)))  # Shape: (batch, 512, 7, 7)
        # - Upsample from 7x7 to 14x14 using PixelShuffle.
        x1 = self.upsample1(x1)                       # Shape: (batch, 512, 14, 14)
        x1 = F.relu(self.bn2(x1))

        # At 14x14 resolution:
        # - Resize the center bias map to 14x14.
        # - Generate another set of n_bias randomly weighted bias features.
        bias_feature2 = F.interpolate(self.center_bias_map_full, size=x1.shape[2:], mode='bilinear', align_corners=False)  # (1, 1, 14, 14)
        random_weights2 = torch.rand(x1.size(0), self.n_bias, 1, 1, device=x.device)
        bias_feature2 = bias_feature2.repeat(1, self.n_bias, 1, 1) * random_weights2  # (batch, n_bias, 14, 14)
        # Concatenate the upsampled features (512 channels) with the bias features, resulting in (512 + n_bias) channels.
        x1_cat = torch.cat([x1, bias_feature2], dim=1)  # (batch, 512 + n_bias, 14, 14)

        # Block 2:
        # - Apply an atrous convolution (5x5 kernel, dilation=2, padding=4) with ReLU and BatchNorm.
        x2 = F.relu(self.bn3(self.atrous2(x1_cat)))  # Shape: (batch, 512, 14, 14)
        # - Upsample from 14x14 to 28x28 using PixelShuffle.
        x2 = self.upsample2(x2)                        # Shape: (batch, 512, 28, 28)
        x2 = F.relu(self.bn4(x2))

        # Final processing:
        # - Apply a 1x1 convolution to reduce channel dimension to 1.
        x_out = self.conv_final(x2)  # Shape: (batch, 1, 28, 28)
        # - Upsample the saliency map to the final output size (e.g., 224x224) using bilinear interpolation.
        x_out = F.interpolate(x_out, size=(self.output_size, self.output_size),
                              mode='bilinear', align_corners=False)
        # - Apply sigmoid activation to obtain values in the range [0, 1].
        x_out = torch.sigmoid(x_out)
        return x_out
