"""
    Module containing the representation learning functionality.
"""
import random
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout

import numpy as np

from models.unet_utils import unetConv2
from models.unet_utils import unetUpNoSKip
from models.unet_utils import unetUp
from models import MLP
from models import encoder
from models import decoder


def _shuffle_segment(shuffled_appearance, start, end, training, num_cameras):
    """TODO
        Helper method for GeoAwareRepLearner forward function.
        :param shuffled_appearance:
        :param start:
        :param end:
        :param training:
        :param num_cameras:
    """
    selected = shuffled_appearance[start:end]
    if training:
        if 0 and end-start == 2:  # Not enabled in ECCV submission, disbled now too HACK
            prob = np.random.random([1])
            # assuming four cameras, make it more often that one of
            # the others is taken, rather than just autoencoding
            # (no flip, which would happen 50% otherwise):
            if prob[0] > 1/num_cameras:
                selected = selected[::-1]  # reverse
            else:
                pass  # let it as it is
        else:
            random.shuffle(selected)

    else:  # deterministic shuffling for testing
        selected = np.roll(selected, 1).tolist()
    shuffled_appearance[start:end] = selected


def _flip_segment(shuffled_appearance, start, width):
    """TODO
        Helper method for GeoAwareRepLearner forward function.
        :param shuffled_appearance:
        :param start:
        :param width:
    """
    selected = shuffled_appearance[start:start+width]
    shuffled_appearance[start:start+width] = shuffled_appearance[start+width:start+2*width]
    shuffled_appearance[start+width:start+2*width] = selected


class GeoAwareRepLearner(nn.Module):
    """ Class defining the representation learner, which has an encoder, decoder, and a MLP for predicting
        3D points.

        Args:
            feature_scale (int): reduce dimensionality by given factor
            in_resolution (int): resolution of input images
            out_channels (int): number of output channels from decoder
            is_deconv (bool): if True, increase dimensions in decoder via deconv, else via bilinear interpolation
            upper_bilinear (bool): decoder will increase dimensionality via bilinear interpolation
            lower_bilinear (bool): encoder will reduce dimensionality via bilinear interpolation
            in_channels (int): number of input channels to the decoder. 3 for RBG images
            is_batchnorm (bool): add batch-normalization operation after activation function in the encoder
            skip_background (bool): sends the background image to the decoder when True #TODO confirm correct
            num_joints (int): number of joints for pose estimator to predict
            nb_dims (int): number of dimensions in encoding transformation
            encoder_type (string): specifies type of encoder to use - UNet or ResNet based
            num_encoding_layers (int): number of conv layers to use in encoding (also determines num in decoding)
            dimension_bg (int): num pixels on one side of square background image
            dimension_fg (int): num pixels on one side of square foreground image
            dimension_3d (int): size of 3d geometry latent space. Must be divisible by 3
            latent_dropout (float): dropout rate between latent space and PoseEstimator
            shuffle_fg (bool): if True swap the appearance of the primary image with another TODO confirm correct
            shuffle_3d (bool): TODO: what is this? why shuffle latend variable ?
            from_latent_hidden_layers (bool): TODO: what is this?
            n_hidden_to3Dpose (int): number of fc layers between latent space and PoseEstimator
            subbatch_size (int): TODO: what is this?
            implicit_rotation (bool): TODO: what is this?
            output_types (tuple): types of output to produce from forward function
            num_cameras (int): number of cameras for each gt 3D pose annotation
    """
    def __init__(self,
                 feature_scale=4,
                 in_resolution=256,  # TODO does this mean 128+128?
                 out_channels=3,
                 is_deconv=True,
                 upper_bilinear=False,
                 lower_bilinear=False,
                 in_channels=3,
                 is_batchnorm=True,
                 skip_background=True,
                 num_joints=17,
                 nb_dims=3,
                 encoder_type='UNet',
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3*64,
                 latent_dropout=0.3,
                 shuffle_fg=True,
                 shuffle_3d=True,
                 from_latent_hidden_layers=False,
                 n_hidden_to3Dpose=2,
                 subbatch_size=4,
                 implicit_rotation=False,
                 output_types=('3D', 'img_crop', 'shuffled_pose', 'shuffled_appearance'),  # TODO: confirm meaning of each
                 num_cameras=4):
        super().__init__()
        assert dimension_3d % 3 == 0
        self.in_resolution = in_resolution
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.dimension_bg = dimension_bg
        self.dimension_fg = dimension_fg
        self.dimension_3d = dimension_3d
        self.shuffle_fg = shuffle_fg
        self.shuffle_3d = shuffle_3d
        self.num_encoding_layers = num_encoding_layers
        self.output_types = output_types
        self.encoder_type = encoder_type
        self.implicit_rotation = implicit_rotation
        self.num_cameras = num_cameras
        self.skip_connections = False  # TODO: add to constructor?
        self.skip_background = skip_background
        self.subbatch_size = subbatch_size
        self.latent_dropout = latent_dropout
        self.filters = [64, 128, 256, 512, 512, 512] # HACK TODO why is this a hack?
        self.filters = [int(x / self.feature_scale) for x in self.filters]
        self.bottleneck_resolution = in_resolution//(2**(num_encoding_layers-1))  # ex: 16 for a 128x128 image (res = 128+128=256)
        self.from_latent_hidden_layers = from_latent_hidden_layers
        num_output_features = self.bottleneck_resolution**2 * self.filters[num_encoding_layers-1]  # TODO size of latent space? how diff from num_output_features_3d?
        print('bottleneck_resolution', self.bottleneck_resolution,
              'num_output_features', num_output_features)

        ######################################################################################################
        #  Create and initialize encoder  ####################################################################
        self.encoder = encoder.Encoder(encoder_type=self.encoder_type,
                                       in_resolution=self.in_resolution,
                                       dimension_fg=self.dimension_fg,
                                       dimension_3d=self.dimension_3d,
                                       in_channels=self.in_channels,
                                       filters=self.filters,
                                       is_batchnorm=self.is_batchnorm,
                                       num_encoding_layers=self.num_encoding_layers,
                                       from_latent_hidden_layers=self.from_latent_hidden_layers)

        #######################################
        ############ background ###############
        if self.skip_background: # add skip connection giving bg to decoder TODO confirm correct description. why not conv1D?
            setattr(self,
                    'conv_1_stage_bg0',
                    unetConv2(self.in_channels,
                              self.filters[0],
                              self.is_batchnorm,
                              padding=1))

        ###########################################################
        ############ latent transformation and pose ############### TODO latent transformation = rotation?
        assert self.dimension_fg < self.filters[num_encoding_layers-1]
        num_output_features_3d = self.bottleneck_resolution**2 * \
                                 (self.filters[num_encoding_layers-1] - self.dimension_fg) # TODO: what is this?
        #  setattr(self, 'fc_1_stage0', Linear(num_output_features, 1024))
        # TODO creates MLP from latent space to 3D prediction? then what is self.to_pose below?
        setattr(self, 'fc_1_stage0', Linear(self.dimension_3d, 128))  # TODO never used?
        setattr(self, 'fc_2_stage0', Linear(128, num_joints * nb_dims))  # TODO never used?

        # TODO creates MLP from latent space to 3D prediction?
        self.to_pose = MLP.MLP_fromLatent(d_in=self.dimension_3d,
                                          d_hidden=2048,
                                          d_out=51,
                                          n_hidden=n_hidden_to3Dpose,
                                          dropout=0.5)
        # TODO layer from encoder to 3d latent space ?
        self.to_3d = nn.Sequential(Linear(num_output_features, self.dimension_3d),
                                   Dropout(inplace=True, p=self.latent_dropout))

        if self.implicit_rotation:
            print("WARNING: doing implicit rotation!")
            rotation_encoding_dimension = 128
            self.encode_angle = nn.Sequential(Linear(3*3, rotation_encoding_dimension//2),
                                              Dropout(inplace=True, p=self.latent_dropout),
                                              ReLU(inplace=False),
                                              Linear(rotation_encoding_dimension//2,
                                                     rotation_encoding_dimension),
                                              Dropout(inplace=True, p=self.latent_dropout),
                                              ReLU(inplace=False),
                                              Linear(rotation_encoding_dimension,
                                                     rotation_encoding_dimension))

            self.rotate_implicitely = nn.Sequential(Linear(self.dimension_3d + \
                                                           rotation_encoding_dimension,
                                                           self.dimension_3d),
                                                    Dropout(inplace=True, p=self.latent_dropout),
                                                    ReLU(inplace=False))

        if self.from_latent_hidden_layers:
            hidden_layer_dimension = 1024
            if self.dimension_fg > 0:
                self.to_fg = nn.Sequential(Linear(num_output_features, 256), # HACK pooling
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False),
                                           Linear(256, self.dimension_fg),
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False))
            self.from_latent = nn.Sequential(Linear(self.dimension_3d, hidden_layer_dimension),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False),
                                             Linear(hidden_layer_dimension, num_output_features_3d),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False))
        else:
            if self.dimension_fg > 0:
                self.to_fg = nn.Sequential(Linear(num_output_features, self.dimension_fg),
                                           Dropout(inplace=True, p=self.latent_dropout),
                                           ReLU(inplace=False))
            self.from_latent = nn.Sequential(Linear(self.dimension_3d, num_output_features_3d),
                                             Dropout(inplace=True, p=self.latent_dropout),
                                             ReLU(inplace=False))

        #  Initialize decoder  ####################################################################
        upper_conv = self.is_deconv and not upper_bilinear
        lower_conv = self.is_deconv and not lower_bilinear
        if self.skip_connections: # TODO skip connection between input and output of conv layer?
            for li in range(1, num_encoding_layers-1):
                setattr(self,
                        'upconv_' + str(li) + '_stage0',
                        unetUp(self.filters[num_encoding_layers-li],
                               self.filters[num_encoding_layers-li-1],
                               upper_conv,
                               padding=1))
        else:
            for li in range(1, num_encoding_layers-1):
                setattr(self,
                        'upconv_' + str(li) + '_stage0',
                        unetUpNoSKip(self.filters[num_encoding_layers-li],
                                     self.filters[num_encoding_layers-li-1],
                                     upper_conv,
                                     padding=1))

        if self.skip_connections or self.skip_background:
            setattr(self,
                    'upconv_' + str(num_encoding_layers-1) + '_stage0',
                    unetUp(self.filters[1], self.filters[0], lower_conv, padding=1))
        else:
            setattr(self,
                    'upconv_' + str(num_encoding_layers-1) + '_stage0',
                    unetUpNoSKip(self.filters[1], self.filters[0], lower_conv, padding=1))

        setattr(self, 'final_stage0', nn.Conv2d(self.filters[0], out_channels, 1))

        #  TODO confirm deletion of these 3 attributes:
        #  self.relu = ReLU(inplace=True)
        #  self.relu2 = ReLU(inplace=False)
        #  self.dropout = Dropout(inplace=True, p=0.3)


    def forward(self, input_dict):
        input = input_dict['img_crop']
        device = input.device
        batch_size = input.size()[0]

        ########################################################
        # Determine shuffling
        shuffled_appearance = list(range(batch_size))  # TODO: what is this?
        shuffled_pose = list(range(batch_size))        # TODO: what is this? why would we swap pose?
        num_pose_subbatches = batch_size//np.maximum(self.subbatch_size, 1)  # TODO: what is this?
        rotation_by_user = not self.training and 'external_rotation_cam' in input_dict.keys()

        if not rotation_by_user:
            if self.shuffle_fg and self.training:
                for i in range(0, num_pose_subbatches):
                    _shuffle_segment(shuffled_appearance,
                                     i*self.subbatch_size,
                                     (i+1)*self.subbatch_size,
                                     self.training,
                                     self.num_cameras)
                for i in range(0, num_pose_subbatches//2): # flip first with second subbatch
                    _flip_segment(shuffled_appearance, i*2*self.subbatch_size, self.subbatch_size)
            if self.shuffle_3d:
                for i in range(0, num_pose_subbatches):
                    _shuffle_segment(shuffled_pose,
                                     i*self.subbatch_size,
                                     (i+1)*self.subbatch_size,
                                     self.training,
                                     self.num_cameras)
        print("subbatch_size: ", self.subbatch_size)
        print("shuffled_pose: ", shuffled_pose)
        print("shuffled appear: ", shuffled_appearance)
        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size
        for i, v in enumerate(shuffled_pose):
            shuffled_pose_inv[v] = i

        shuffled_appearance = torch.LongTensor(shuffled_appearance).to(device)
        shuffled_pose = torch.LongTensor(shuffled_pose).to(device)
        shuffled_pose_inv = torch.LongTensor(shuffled_pose_inv).to(device)

        if rotation_by_user:
            if 'shuffled_appearance' in input_dict.keys():
                shuffled_appearance = input_dict['shuffled_appearance'].long()

        ###############################################
        # determine shuffled rotation
        cam_2_world = input_dict['extrinsic_rot_inv'].view((batch_size, 3, 3)).float()
        world_2_cam = input_dict['extrinsic_rot'].view((batch_size, 3, 3)).float()
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1, 3, 3).expand((batch_size, 3, 3))
            external_glob = input_dict['external_rotation_global'].view(1, 3, 3).expand((batch_size, 3, 3))
            cam2cam = torch.bmm(external_cam,
                                torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_suffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)
            cam2cam = torch.bmm(world_2_cam_suffled, cam_2_world)

        input_dict_cropped = input_dict #  fallback to using crops

        ###############################################
        # encoding stage

        has_fg = hasattr(self, "to_fg")
        # TODO: confirm correct:
        # latent_3d is the image encoded into the latent space
        # latent_fg is the encoded appearance vector. shape at inf time is [2,128] where
            # index 0 is the appearance of the person at a different time and from a different angle
            # and index 1 is the appearence vec of the primary person
        latent_3d, latent_fg = self.encoder(has_fg, input_dict_cropped, batch_size)
        print("latent_3d shape: ", latent_3d.shape)
        print("latent fg shape: ", latent_fg.shape)
        if self.skip_background:  # send bg to the decoder
            input_bg = input_dict['bg_crop'] # TODO take the rotated one/ new view
            input_bg_shuffled = torch.index_select(input_bg, dim=0, index=shuffled_pose)
            conv1_bg_shuffled = getattr(self, 'conv_1_stage_bg0')(input_bg_shuffled)

        ###############################################
        # latent rotation (to shuffled view)
        if self.implicit_rotation:
            encoded_angle = self.encode_angle(cam2cam.view(batch_size, -1))
            encoded_latent_and_angle = torch.cat([latent_3d.view(batch_size, -1),
                                                  encoded_angle], dim=1)
            latent_3d_rotated = self.rotate_implicitely(encoded_latent_and_angle)
        else:
            latent_3d_rotated = torch.bmm(latent_3d, cam2cam.transpose(1, 2))

        if 'shuffled_pose_weight' in input_dict.keys():
            w = input_dict['shuffled_pose_weight']
            # weighted average with the last one
            latent_3d_rotated = (1-w.expand_as(latent_3d))* \
                                 latent_3d + \
                                 w.expand_as(latent_3d) * \
                                 latent_3d_rotated[-1:].expand_as(latent_3d)

        if has_fg:
            latent_fg_shuffled = torch.index_select(latent_fg, dim=0, index=shuffled_appearance)
            if 'shuffled_appearance_weight' in input_dict.keys():
                w = input_dict['shuffled_appearance_weight']
                latent_fg_shuffled = (1 - w.expand_as(latent_fg)) * \
                                      latent_fg + w.expand_as(latent_fg) * \
                                      latent_fg_shuffled

        ###############################################
        # decoding
        map_from_3d = self.from_latent(latent_3d_rotated.view(batch_size, -1))  # the output from NN that takes in the latent variable
        map_width = self.bottleneck_resolution #out_enc_conv.size()[2]
        map_channels = self.filters[self.num_encoding_layers-1] #out_enc_conv.size()[1]
        if has_fg:
            latent_fg_shuffled_replicated = latent_fg_shuffled.view(batch_size,
                                                                    self.dimension_fg, 1,
                                                                    1).expand(batch_size,
                                                                              self.dimension_fg,
                                                                              map_width,
                                                                              map_width)
            latent_shuffled = torch.cat([latent_fg_shuffled_replicated,
                                         map_from_3d.view(batch_size,
                                                          map_channels-self.dimension_fg,
                                                          map_width, map_width)], dim=1)
        else:
            latent_shuffled = map_from_3d.view(batch_size, map_channels, map_width, map_width)

        if self.skip_connections:
            assert False
        else:
            out_deconv = latent_shuffled
            for li in range(1, self.num_encoding_layers-1):
                out_deconv = getattr(self, 'upconv_' + str(li) + '_stage0')(out_deconv)

            if self.skip_background:
                out_deconv = getattr(self, 'upconv_' + \
                                     str(self.num_encoding_layers-1) + \
                                     '_stage0')(conv1_bg_shuffled, out_deconv)
            else:
                out_deconv = getattr(self, 'upconv_' + \
                                     str(self.num_encoding_layers-1) + \
                                     '_stage0')(out_deconv)

        output_img_shuffled = getattr(self, 'final_stage0')(out_deconv)

        ###############################################
        # de-shuffling
        output_img = torch.index_select(output_img_shuffled, dim=0, index=shuffled_pose_inv)

        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_pose = self.to_pose.forward({'latent_3d': latent_3d})['3D']

        ###############################################
        # Select the right output
        output_dict_all = {'3D' : output_pose,
                           'img_crop' : output_img,
                           'shuffled_pose' : shuffled_pose,
                           'shuffled_appearance' : shuffled_appearance,
                           'latent_3d': latent_3d,
                           'cam2cam': cam2cam}
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict
