import torch.nn as nn
from functools import partial
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
    """
    Calculates the output dimension of a pooling layer.
    """
    # If stride is 0, set it to the kernel size
    if stride == 0:
        stride = kernel_size

    # Calculate the output dimension
    num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
    outDim = int(np.floor(num/stride + 1))

    return outDim
class DoubleConv(nn.Module):
    """
    Double convolution layer

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        encoder (bool, optional): Whether to use this layer as an encoder. Defaults to True.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        conv_padding (int, optional): The amount of padding to use in the convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, encoder=True, conv_kernel_size=3, conv_padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels//2
            conv2_in_channels = out_channels//2
            conv2_out_channels = out_channels
        else:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            conv2_in_channels = out_channels
            conv2_out_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv3d(conv1_in_channels, conv1_out_channels, conv_kernel_size, padding=conv_padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(conv1_out_channels),
            nn.Conv3d(conv2_in_channels, conv2_out_channels, conv_kernel_size, padding=conv_padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(conv2_out_channels),
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.double_conv(x)
    
            
class EncoderModule(nn.Module):
    """
    Encoder module of the UNet.

    Args:
        in_channels (int): The number of input channels.
        f_maps (int, optional): The number of feature maps in the first layer. Defaults to 64.
        num_levels (int, optional): The number of levels in the UNet. Defaults to 4.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        pool_kernel_size (int, optional): The size of the pooling kernel. Defaults to 2.
        conv_padding (int, optional): The amount of padding to use in the convolution. Defaults to 1.
    """
    def __init__(self, in_channels, f_maps=64, num_levels=4, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1):
        super(EncoderModule, self).__init__()
        self.encoder_steps = nn.ModuleList()
        for k in range(num_levels):
            features = f_maps*(2**k)
            self.encoder_steps.append(DoubleConv(in_channels=in_channels, out_channels=features))
            in_channels = features
            
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
            list: A list of tensors with the output of each level.
        """
        encoders_features = []
        for down in self.encoder_steps:
            x = down(x)
            #print(x.shape)
            if down != self.encoder_steps[-1]:
                encoders_features.insert(0, x)
                x = self.pooling(x)
                #print(x.shape)
        return x, encoders_features


        
class DecoderModule(nn.Module):
    """
    Decoder module of the UNet.

    Args:
        out_channels (int): The number of output channels.
        f_maps (int, optional): The number of feature maps in the first layer. Defaults to 64.
        num_levels (int, optional): The number of levels in the UNet. Defaults to 4.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        pool_kernel_size (int, optional): The size of the pooling kernel. Defaults to 2.
        conv_padding (int, optional): The amount of padding to use in the convolution. Defaults to 1.
    """
    def __init__(self, out_channels, f_maps=64, num_levels=4, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1):
        super(DecoderModule, self).__init__()
        self.decoder_steps = nn.ModuleList()
        self.upconv = nn.ModuleList()

        for k in reversed(range(num_levels)):
            features = int(f_maps*(2**(k-1)))
            self.upconv.append(nn.ConvTranspose3d(features*2, features*2, kernel_size=pool_kernel_size, stride=2))
            self.decoder_steps.append(DoubleConv(in_channels=features*2+features, out_channels=features, encoder=False))
        self.final_layer = DoubleConv(in_channels=features*2, out_channels=out_channels, encoder=False)
        
        
    def forward(self, x, encoders_features):
        """
        Args:
            x (torch.Tensor): The input tensor.
            encoders_features (list): A list of tensors with the output of each level of the encoder.

        Returns:
            torch.Tensor: The output tensor.
        """
        for i in range(len(encoders_features)):
            x = self.upconv[i](x)
            concat_x = torch.cat((encoders_features[i], x), dim=1)
            x = self.decoder_steps[i](concat_x)
        x=self.final_layer(x)
        return x        

class UNetModule(nn.Module):
    """
    UNet

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        img_size (tuple): The size of the input image.
        f_maps (int, optional): The number of feature maps in the first layer. Defaults to 64.
        num_levels (int, optional): The number of levels in the UNet. Defaults to 4.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        pool_kernel_size (int, optional): The size of the pooling kernel. Defaults to 2.
        conv_padding (int, optional): The amount of padding to use in the convolution. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, img_size, f_maps=64, 
                 num_levels=4, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1):
        super(UNetModule, self).__init__()
        self.encoder = EncoderModule(in_channels=in_channels, f_maps=f_maps, num_levels=num_levels) 
        self.decoder = DecoderModule(out_channels=out_channels, f_maps=f_maps, num_levels=num_levels)
        self.l_encoder_out = int(np.prod(np.array(img_size)/(2**(num_levels-1)))*f_maps*(2**(num_levels-1)))
        self.final = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x, encoders_features = self.encoder(x)
        x = self.decoder(x, encoders_features)
        return self.final(x)


class LocalizationModule(nn.Module):
    """
    Spatial Transformer for Localization

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        img_size (tuple): The size of the input image.
        f_maps (int, optional): The number of feature maps in the first layer. Defaults to 16.
        num_levels (int, optional): The number of levels in the UNet. Defaults to 4.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        pool_kernel_size (int, optional): The size of the pooling kernel. Defaults to 2.
        conv_padding (int, optional): The amount of padding to use in the convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, img_size, f_maps=16, 
                 num_levels=4, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1):
        super(LocalizationModule, self).__init__()
        self.flow_unet = EncoderModule(2, f_maps=f_maps, num_levels=num_levels) 
        self.l_encoder_out = int(np.prod(np.array(img_size)/(2**(num_levels-1)))*f_maps*(2**(num_levels-1)))
        self.fc_loc = nn.Sequential(
            nn.Linear(self.l_encoder_out, 64),
            nn.PReLU(),
            nn.Linear(64, 3 * 4)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        
        
    def forward(self, x, xm, x_seg):
        """
        Args:
            x (torch.Tensor): The input tensor.
            xm (torch.Tensor): The input tensor.
            x_seg (torch.Tensor): The input tensor.

        Returns:
            x_seg: Tranformed segmentation
            x_stn: Tranformed image
            theta: Transformation matrix
        """
        xinp = torch.cat([x_seg, xm], dim=1)
        xs, _ = self.flow_unet(xinp)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)
        grid = F.affine_grid(theta, (1, 1, 64, 64, 64), align_corners=False)
        x_stn = F.grid_sample(x, grid, align_corners=False)
        x_seg = F.grid_sample(x_seg, grid, align_corners=False)
        return x_seg, x_stn, theta

        

## For debug purpose only   
if __name__ == "__main__":
    import nrrd
    DEVICE = "cpu"
    encoder = LocalizationModule(in_channels=1, out_channels=1, img_size=[128, 128, 128], f_maps=32, num_levels=6).to(DEVICE)
    float_tensor = torch.tensor(np.zeros((128, 128, 128))).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(float_tensor.shape)
    output = encoder(float_tensor) 
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape) 
