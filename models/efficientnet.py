import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Constants for EfficientNet versions
# (width_factor, depth_factor, resolution, dropout_rate)
EFFICIENTNET_PARAMS = {
    # B0 to B7 standard models
    0: (1.0, 1.0, 224, 0.2),
    1: (1.0, 1.1, 240, 0.2),
    2: (1.1, 1.2, 260, 0.3),
    3: (1.2, 1.4, 300, 0.3),
    4: (1.4, 1.8, 380, 0.4),
    5: (1.6, 2.2, 456, 0.4),
    6: (1.8, 2.6, 528, 0.5),
    7: (2.0, 3.1, 600, 0.5),
}

class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block for EfficientNet"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, dropout_rate=None):
        super(MBConvBlock, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.use_residual = (in_channels == out_channels and stride == 1)
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv3d(in_channels, expanded_channels, 
                                        kernel_size=1, stride=1, padding=0, bias=False)
            self.bn0 = nn.BatchNorm3d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv3d(expanded_channels, expanded_channels, 
                                       kernel_size=kernel_size, stride=stride,
                                       padding=(kernel_size-1)//2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm3d(expanded_channels)
        
        # Squeeze and Excitation
        self.se = SEModule(expanded_channels, reduction=int(in_channels * se_ratio))
        
        # Output phase
        self.project_conv = nn.Conv3d(expanded_channels, out_channels, 
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.swish = Swish()
        
    def forward(self, inputs):
        x = inputs
        
        # Expansion
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        # Squeeze and Excitation
        x = self.se(x)
        
        # Output
        x = self.project_conv(x)
        x = self.bn2(x)
        
        # Skip connection
        if self.use_residual:
            if self.dropout_rate is not None and self.dropout_rate > 0 and self.training:
                x = F.dropout3d(x, p=self.dropout_rate, training=self.training)
            x = x + inputs
            
        return x

class EfficientNet3D(nn.Module):
    """
    3D implementation of EfficientNet
    
    Args:
        width_factor: Width factor for scaling model width
        depth_factor: Depth factor for scaling number of layers
        dropout_rate: Dropout rate
        num_classes: Number of output classes
        n_input_channels: Number of input channels (3 for RGB)
    """
    def __init__(self, 
                 width_factor=1.0, 
                 depth_factor=1.0, 
                 dropout_rate=0.2, 
                 num_classes=400,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False):
        super(EfficientNet3D, self).__init__()
        
        # Base EfficientNet architecture parameters
        # (in_channels, out_channels, kernel_size, stride, expand_ratio, repeats)
        base_architecture = [
            # stage 1
            [32, 16, 3, 1, 1, 1],
            # stage 2
            [16, 24, 3, 2, 6, 2],
            # stage 3
            [24, 40, 5, 2, 6, 2],
            # stage 4
            [40, 80, 3, 2, 6, 3],
            # stage 5
            [80, 112, 5, 1, 6, 3],
            # stage 6
            [112, 192, 5, 2, 6, 4],
            # stage 7
            [192, 320, 3, 1, 6, 1]
        ]
        
        # Adjust channels based on width_factor
        input_channels = self._round_filters(32, width_factor)
        
        # First convolution layer
        self.conv_stem = nn.Conv3d(n_input_channels, input_channels, 
                                  kernel_size=(conv1_t_size, 3, 3),
                                  stride=(conv1_t_stride, 2, 2),
                                  padding=((conv1_t_size - 1) // 2, 1, 1),
                                  bias=False)
        self.bn0 = nn.BatchNorm3d(input_channels)
        self.swish = Swish()
        
        # Build blocks
        self.blocks = nn.ModuleList([])
        
        # For each stage in the base architecture
        for stage_params in base_architecture:
            in_chs, out_chs, kernel_size, stride, exp_ratio, repeats = stage_params
            
            # Adjust filters and repeats based on width and depth factors
            out_chs = self._round_filters(out_chs, width_factor)
            repeats = self._round_repeats(repeats, depth_factor)
            
            # Create blocks for this stage
            for i in range(repeats):
                # Only the first block in each stage uses the specified stride
                block_stride = stride if i == 0 else 1
                block_in_chs = input_channels if i == 0 else out_chs
                
                self.blocks.append(MBConvBlock(
                    in_channels=block_in_chs,
                    out_channels=out_chs,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    expand_ratio=exp_ratio,
                    dropout_rate=dropout_rate
                ))
                
            input_channels = out_chs
        
        # Head
        out_channels = self._round_filters(1280, width_factor)
        self.conv_head = nn.Conv3d(input_channels, out_channels,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Final classifier
        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def _round_filters(self, filters, width_factor):
        """Round number of filters based on depth multiplier."""
        multiplier = width_factor
        divisor = 8
        filters *= multiplier
        min_depth = divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, depth_factor):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_factor * repeats))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.swish(x)
        
        # Blocks
        for block in self.blocks:
            x = block(x)
            
        # Head
        x = self.conv_head(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
def get_efficientnet_params(version):
    """Get efficientnet parameters based on version"""
    if version in EFFICIENTNET_PARAMS:
        width_factor, depth_factor, resolution, dropout_rate = EFFICIENTNET_PARAMS[version]
        return width_factor, depth_factor, resolution, dropout_rate
    else:
        raise ValueError(f"EfficientNet version {version} not supported")

def generate_model(model_depth=0, n_classes=400, n_input_channels=3, 
                  dropout_rate=None, conv1_t_size=7, 
                  conv1_t_stride=1, no_max_pool=False):
    """
    Generate EfficientNet model
    
    Args:
        model_depth: EfficientNet version (0-7 for B0-B7)
        n_classes: Number of classes
        n_input_channels: Number of input channels
        dropout_rate: Override default dropout rate
        conv1_t_size: Size of temporal kernel in first conv layer
        conv1_t_stride: Stride of temporal kernel in first conv layer
        no_max_pool: If true, max pooling after first conv is removed
    """
    width_factor, depth_factor, resolution, default_dropout_rate = get_efficientnet_params(model_depth)
    
    # Use provided dropout rate or the default one from model definition
    if dropout_rate is None:
        dropout_rate = default_dropout_rate
    
    model = EfficientNet3D(
        width_factor=width_factor,
        depth_factor=depth_factor,
        dropout_rate=dropout_rate,
        num_classes=n_classes,
        n_input_channels=n_input_channels,
        conv1_t_size=conv1_t_size,
        conv1_t_stride=conv1_t_stride,
        no_max_pool=no_max_pool
    )
    
    return model 