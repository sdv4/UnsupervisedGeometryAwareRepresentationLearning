"""

"""
import torch
import torch.nn as nn
from models.unet_utils import unetUpNoSKip
from models.unet_utils import unetUp
from models.decoder import Decoder


class UnetDecoder(Decoder):
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
        super().__init__(upper_bilinear,
                         lower_bilinear,
                         is_deconv,
                         num_decoding_layers,
                         out_channels,
                         filters,
                         bottleneck_resolution,
                         skip_background,
                         dimension_fg)

        upper_conv = is_deconv and not upper_bilinear
        lower_conv = is_deconv and not lower_bilinear

        for li in range(1, num_decoding_layers - 1):
            setattr(self,
                    'upconv_' + str(li) + '_stage0',
                    unetUpNoSKip(self.filters[num_decoding_layers - li],
                                 self.filters[num_decoding_layers - li - 1],
                                 upper_conv,
                                     padding=1))


        setattr(self,
                'upconv_' + str(num_decoding_layers - 1) + '_stage0',
                unetUp(self.filters[1], self.filters[0], lower_conv, padding=1))

        setattr(self, 'final_stage0', nn.Conv2d(self.filters[0], out_channels, 1))


    def forward(self, map_from_3d, latent_fg_shuffled, conv1_bg_shuffled, batch_size):
        """

        :param latent_3d_rotated:
        :param latent_fg_shuffled:
        :param batch_size:
        :return: output_img_shuffled
        """
        map_width = self.bottleneck_resolution  # out_enc_conv.size()[2]
        map_channels = self.filters[self.num_decoding_layers - 1]  # out_enc_conv.size()[1]
        latent_fg_shuffled_replicated = latent_fg_shuffled.view(batch_size,
                                                                self.dimension_fg, 1,
                                                                1).expand(batch_size,
                                                                          self.dimension_fg,
                                                                          map_width,
                                                                          map_width)
        latent_shuffled = torch.cat([latent_fg_shuffled_replicated,
                                     map_from_3d.view(batch_size,
                                                      map_channels - self.dimension_fg,
                                                      map_width, map_width)], dim=1)



        out_deconv = latent_shuffled
        for li in range(1, self.num_decoding_layers - 1):
            out_deconv = getattr(self, 'upconv_' + str(li) + '_stage0')(out_deconv)

        if self.skip_background:
            out_deconv = getattr(self, 'upconv_' + \
                                 str(self.num_decoding_layers - 1) + \
                                 '_stage0')(conv1_bg_shuffled, out_deconv)
        else:
            out_deconv = getattr(self, 'upconv_' + \
                                 str(self.num_decoding_layers - 1) + \
                                 '_stage0')(out_deconv)

        output_img_shuffled = getattr(self, 'final_stage0')(out_deconv)

        return output_img_shuffled
