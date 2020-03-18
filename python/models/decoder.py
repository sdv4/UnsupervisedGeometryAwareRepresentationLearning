"""
    Module defining the abstract Decoder object
"""
import torch.nn as nn


class Decoder(nn.Module):
    """
        Abstract class for object that will take a latent variable (and optionally an image background)
        and output an image.
    Args:
        upper_bilinear (bool): if True do upscalling via bilinear interpolation instead of deconvolution
        lower_bilinear (bool): if True do down-scalling via bilinear interpolation instead of convolutions
        s_deconv (bool): if True, increase dimensions in decoder via deconv, else via bilinear interpolation
        num_decoding_layers (int): number of stacked 'hourglass' layers to use in going from latent to decoded output image
        filters (list): number of channels per within each hourglass layer
        bottleneck_resolution (int): TODO
    """
    def __init__(self,
                 upper_bilinear,
                 lower_bilinear,
                 is_deconv,
                 num_decoding_layers,
                 out_channels,
                 filters,
                 bottleneck_resolution,
                 skip_background,
                 dimension_fg):
        super().__init__()
        self.upper_bilinear = upper_bilinear
        self.lower_biliner = lower_bilinear
        self.is_deconv = is_deconv
        self.num_decoding_layers = num_decoding_layers
        self.out_channels = out_channels
        self.filters = filters
        self.bottleneck_resolution = bottleneck_resolution
        self.skip_background = skip_background
        self.dimension_fg = dimension_fg


    def forward(self, latent_3d_rotated, latent_fg_shuffled, batch_size):
        pass
