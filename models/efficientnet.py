import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
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
    def __init__(self, in_planes, planes, expansion, stride=1, kernel_size=3, reduction_ratio=4, dropout_rate=0.2, use_residual=True):
        super(MBConvBlock, self).__init__()
        self.use_residual = use_residual and stride == 1 and in_planes == planes
        self.dropout_rate = dropout_rate
        expand_planes = in_planes * expansion
        
        # Expansion phase
        if expansion != 1:
            self.expand_conv = nn.Conv3d(in_planes, expand_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm3d(expand_planes)
            self.expand_swish = Swish()
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv3d(
            expand_planes, expand_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=expand_planes, bias=False
        )
        self.depthwise_bn = nn.BatchNorm3d(expand_planes)
        self.depthwise_swish = Swish()
        
        # Squeeze and Excitation
        self.se = SEModule(expand_planes, reduction=reduction_ratio)
        
        # Pointwise convolution
        self.project_conv = nn.Conv3d(expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm3d(planes)
        
        # Dropout
        if self.use_residual and dropout_rate > 0:
            self.dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_swish(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_swish(x)
        
        # Squeeze and Excitation
        x = self.se(x)
        
        # Pointwise
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.use_residual:
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x += identity
            
        return x


class EfficientNet3D(nn.Module):
    def __init__(self, 
                 width_multiplier=1.0,
                 depth_multiplier=1.0,
                 dropout_rate=0.2,
                 num_classes=400,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False):
        super(EfficientNet3D, self).__init__()
        
        # EfficientNet-B0 baseline parameters for blocks
        block_params = [
            # expansion, channels, layers, kernel_size, stride
            [1, 16, 1, 3, 1],
            [6, 24, 2, 3, 2],
            [6, 40, 2, 5, 2],
            [6, 80, 3, 3, 2],
            [6, 112, 3, 5, 1],
            [6, 192, 4, 5, 2],
            [6, 320, 1, 3, 1]
        ]
        
        # Adjust channels based on width multiplier
        self.input_channels = int(32 * width_multiplier)
        
        # Initial stem
        self.conv1 = nn.Conv3d(n_input_channels, self.input_channels,
                              kernel_size=(conv1_t_size, 3, 3),
                              stride=(conv1_t_stride, 2, 2),
                              padding=(conv1_t_size // 2, 1, 1),
                              bias=False)
        self.bn1 = nn.BatchNorm3d(self.input_channels)
        self.swish = Swish()
        self.no_max_pool = no_max_pool
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Build MBConv blocks
        self.blocks = nn.Sequential()
        layer_idx = 0
        
        for block_idx, (expansion, channels, layers, kernel_size, stride) in enumerate(block_params):
            # Adjust output channels based on width multiplier
            output_channels = int(channels * width_multiplier)
            
            # Adjust the number of layers per block based on depth multiplier
            repeats = int(math.ceil(layers * depth_multiplier))
            
            # Create block
            for i in range(repeats):
                # Use stride only for the first layer of each block
                block_stride = stride if i == 0 else 1
                
                self.blocks.add_module(
                    f'block{layer_idx}',
                    MBConvBlock(
                        self.input_channels, output_channels, expansion, 
                        stride=block_stride, kernel_size=kernel_size, 
                        dropout_rate=dropout_rate
                    )
                )
                self.input_channels = output_channels
                layer_idx += 1
        
        # Head
        self.head_channels = int(1280 * width_multiplier)
        self.head_conv = nn.Conv3d(self.input_channels, self.head_channels, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm3d(self.head_channels)
        self.head_swish = Swish()
        
        # Final FC layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.head_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
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
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        
        # MBConv blocks
        x = self.blocks(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        
        # Pooling and FC
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_efficientnet_params(model_name):
    """Get efficientnet parameters based on model name."""
    params_dict = {
        # width_mult, depth_mult, resolution, dropout_rate
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def generate_model(model_name='efficientnet-b0', **kwargs):
    """Generate EfficientNet model."""
    if model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 
                      'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 
                      'efficientnet-b6', 'efficientnet-b7']:
        width_mult, depth_mult, _, dropout_rate = get_efficientnet_params(model_name)
    else:
        # Default to b0 parameters
        width_mult, depth_mult, _, dropout_rate = 1.0, 1.0, 224, 0.2
    
    model = EfficientNet3D(
        width_multiplier=width_mult,
        depth_multiplier=depth_mult,
        dropout_rate=dropout_rate,
        **kwargs
    )
    
    return model 