"""
ACDNet Model for UrbanSound8K Classification
Adapted from: Environmental Sound Classification on the Edge (arXiv:2103.03483)

Architecture consists of:
- SFEB (Spatial Feature Extraction Block): Convolutional filter bank
- TFEB (Temporal Feature Extraction Block): Temporal feature processing
"""

import torch
import torch.nn as nn


class ACDNetV2(nn.Module):
    """
    ACDNet V2 for raw waveform audio classification
    
    Args:
        input_length: Length of input audio in samples (30000 for 1.5s at 20kHz)
        n_class: Number of output classes (10 for UrbanSound8K)
        sr: Sampling rate (20000 Hz)
        ch_conf: Optional channel configuration for custom model sizes
    """
    
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ACDNetV2, self).__init__()
        self.input_length = input_length
        self.ch_config = ch_conf
        
        stride1 = 2
        stride2 = 2
        channels = 8
        k_size = (3, 3)
        n_frames = (sr / 1000) * 10  # No of frames per 10ms
        
        sfeb_pool_size = int(n_frames / (stride1 * stride2))
        
        # Default channel configuration if not provided
        if self.ch_config is None:
            self.ch_config = [
                channels, 
                channels*8,   # 64
                channels*4,   # 32
                channels*8,   # 64
                channels*8,   # 64
                channels*16,  # 128
                channels*16,  # 128
                channels*32,  # 256
                channels*32,  # 256
                channels*64,  # 512
                channels*64,  # 512
                n_class       # 10
            ]
        
        fcn_no_of_inputs = self.ch_config[-1]
        
        # Create convolutional layers
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1))
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2))
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1)
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1)
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1)
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1)
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1)
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1)
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1)
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1)
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1)
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1))
        
        fcn = nn.Linear(fcn_no_of_inputs, n_class)
        # Kaiming initialization with sigmoid is equivalent to lecun_normal in Keras
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid')
        
        # SFEB: Spatial Feature Extraction Block (Filter bank)
        self.sfeb = nn.Sequential(
            conv1, bn1, nn.ReLU(),
            conv2, bn2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )
        
        # TFEB: Temporal Feature Extraction Block
        tfeb_modules = []
        self.tfeb_width = int(((self.input_length / sr) * 1000) / 10)  # 10ms frames
        tfeb_pool_sizes = self.get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width)
        p_index = 0
        
        for i in [3, 4, 6, 8, 10]:
            tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()])
            
            if i != 3:
                tfeb_modules.extend([eval('conv{}'.format(i+1)), eval('bn{}'.format(i+1)), nn.ReLU()])
            
            h, w = tfeb_pool_sizes[p_index]
            if h > 1 or w > 1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size=(h, w)))
            p_index += 1
        
        tfeb_modules.append(nn.Dropout(0.2))
        tfeb_modules.extend([conv12, bn12, nn.ReLU()])
        
        h, w = tfeb_pool_sizes[-1]
        if h > 1 or w > 1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size=(h, w)))
        
        tfeb_modules.extend([nn.Flatten(), fcn])
        
        self.tfeb = nn.Sequential(*tfeb_modules)
        
        # Output layer with softmax
        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 1, 1, length)
        
        Returns:
            y: Output probabilities of shape (batch, n_class)
        """
        x = self.sfeb(x)
        # Swap axes for temporal processing
        x = x.permute((0, 2, 1, 3))
        x = self.tfeb(x)
        y = self.output[0](x)
        return y
    
    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        """Create convolutional layer with batch normalization"""
        conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        # Kaiming initialization with relu is equivalent to he_normal in Keras
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn
    
    def get_tfeb_pool_sizes(self, con2_ch, width):
        """Calculate pooling sizes for TFEB"""
        h = self.get_tfeb_pool_size_component(con2_ch)
        w = self.get_tfeb_pool_size_component(width)
        pool_size = []
        for (h1, w1) in zip(h, w):
            pool_size.append((h1, w1))
        return pool_size
    
    def get_tfeb_pool_size_component(self, length):
        """Calculate pooling size component"""
        c = []
        index = 1
        while index <= 6:
            if length >= 2:
                if index == 6:
                    c.append(length)
                else:
                    c.append(2)
                    length = length // 2
            else:
                c.append(1)
            index += 1
        return c


def GetACDNetModel(input_len=30000, nclass=10, sr=20000, channel_config=None):
    """
    Create ACDNet model for UrbanSound8K
    
    Args:
        input_len: Length of input audio (30000 = 1.5s at 20kHz)
        nclass: Number of classes (10 for UrbanSound8K)
        sr: Sampling rate (20000 Hz)
        channel_config: Optional custom channel configuration
    
    Returns:
        ACDNetV2 model instance
    """
    net = ACDNetV2(input_len, nclass, sr, ch_conf=channel_config)
    return net


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_shape=(1, 1, 1, 30000)):
    """
    Get a summary of the model
    
    Args:
        model: ACDNet model
        input_shape: Shape of input tensor (batch, channels, height, width)
    
    Returns:
        Dictionary with model statistics
    """
    total_params = count_parameters(model)
    
    # Estimate model size in MB
    param_size = total_params * 4  # 4 bytes per float32 parameter
    model_size_mb = param_size / (1024 ** 2)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': total_params,
        'model_size_mb': model_size_mb,
        'input_shape': input_shape
    }
    
    return summary
