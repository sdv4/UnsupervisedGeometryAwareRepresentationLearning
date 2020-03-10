"""
    Module containing the Encoder

    TODO: decide if this should be a abstract class for seperate resnet and unet encoders
"""
import torch.nn as nn
from models import resnet_VNECT_3Donly
from models.unet_utils import unetConv2



class Encoder(nn.Module):
    def __init__(self,
                 encoder_type,
                 in_resolution,
                 dimension_fg,
                 dimension_3d,
                 in_channels,
                 filters,
                 is_batchnorm,
                 num_encoding_layers,
                 from_latent_hidden_layers):
        super().__init__()
        self.encoder_type = encoder_type
        self.in_resolution = in_resolution
        self.dimension_fg = dimension_fg
        self.dimension_3d = dimension_3d
        self.in_channels = in_channels
        self.filters = filters
        self.is_batchnorm = is_batchnorm
        self.num_encoding_layers = num_encoding_layers
        self.from_latent_hidden_layers = from_latent_hidden_layers
        #  Initialize encoder  ####################################################################
        if self.encoder_type == "ResNet":
            self.encoder = resnet_VNECT_3Donly.resnet50(pretrained=True,
                                                               input_key='img_crop',
                                                               output_keys=['latent_3d', '2D_heat'],
                                                               input_width=self.in_resolution,
                                                               num_classes=self.dimension_fg + self.dimension_3d)
        else:  # encoder_type is "UNet"
            setattr(self,
                    'conv_1_stage0',
                    unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage0', nn.MaxPool2d(kernel_size=2))
            # note, first layer(li==1) is already created,
            # last layer(li==num_encoding_layers) is created externally:
            for li in range(2, self.num_encoding_layers):
                setattr(self, 'conv_' + str(li) + '_stage0',
                        unetConv2(self.filters[li - 2], self.filters[li - 1], self.is_batchnorm, padding=1))
                setattr(self, 'pool_' + str(li) + '_stage0', nn.MaxPool2d(kernel_size=2))

            if self.from_latent_hidden_layers:
                setattr(self, 'conv_' + str(self.num_encoding_layers) + '_stage0',
                        nn.Sequential(unetConv2(self.filters[self.num_encoding_layers - 2],
                                                self.filters[self.num_encoding_layers - 1],
                                                self.is_batchnorm,
                                                padding=1),
                                      nn.MaxPool2d(kernel_size=2)))
            else:
                setattr(self, 'conv_' + str(self.num_encoding_layers) + '_stage0',
                        unetConv2(self.filters[self.num_encoding_layers - 2],
                                  self.filters[self.num_encoding_layers - 1],
                                  self.is_batchnorm,
                                  padding=1))


    def forward(self, has_fg, input_dict_cropped, batch_size):
        """
        TODO
        :param input_dict:
        :return:
        """
        if self.encoder_type == "ResNet":
            output = self.encoder(input_dict_cropped)['latent_3d']
            print("output shape: ", output.shape)
            if has_fg:
                latent_fg = output[:, :self.dimension_fg]
            latent_3d = output[:, self.dimension_fg:self.dimension_fg + \
                                                    self.dimension_3d].contiguous().view(batch_size, -1, 3)
        else:  # UNet encoder
            out_enc_conv = input_dict_cropped['img_crop']

            # note, first layer(li==1) is already created,
            # last layer(li==num_encoding_layers) is created externally
            for li in range(1, self.num_encoding_layers):
                out_enc_conv = getattr(self, 'conv_' + str(li) + '_stage0')(out_enc_conv)
                out_enc_conv = getattr(self, 'pool_' + str(li) + '_stage0')(out_enc_conv)
            out_enc_conv = getattr(self, 'conv_' + str(self.num_encoding_layers) + '_stage0')(out_enc_conv)

            # fully-connected
            center_flat = out_enc_conv.view(batch_size, -1)
            if has_fg:
                latent_fg = self.to_fg(center_flat)
            latent_3d = self.to_3d(center_flat).view(batch_size, -1, 3)
        return latent_3d, latent_fg
