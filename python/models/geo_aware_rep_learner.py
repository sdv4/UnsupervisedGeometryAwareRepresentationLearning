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
from models import mlp
from models import encoder
from models import unet_decoder


def _shuffle_within_segment(to_shuffle, start, end, training, num_cameras):
    """ Shuffles elements within a portion of a given list.
        Helper method for GeoAwareRepLearner forward function used to
        shuffle within subbatches and run over the entire batch.

        TODO: make doctest allows for randomness of function
        Args:
            to_shuffle (list): the list whose elements will be shuffled within subbatches
            start (int): index of leftmost element to shuffle
            end (int): index of rightmost element to shuffle
            training (bool): True if model is being trained; False if testing
            num_cameras (int): the number of cameras used to collect images for a given time

        Example:

            batch_indicies = [0, 1, 2, 3, 4, 5, 6, 7]
            _shuffle_within_segment(batch_indicies, 0, 4, True, 4)
            = [3, 1, 0, 2, 4, 5, 6, 7]
    """
    selected = to_shuffle[start:end]
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
    to_shuffle[start:end] = selected


def _flip_segment(to_shuffle, start, width):
    """ Method to flip to subsections of a list with another at some point in the list.
        Helper method for GeoAwareRepLearner forward function.

        Args:
            to_shuffle (list):
            start (int):
            width (int):

        Example:

            batch_indicies = [0, 1, 2, 3, 4, 5, 6, 7]
            _flip_segment(batch_indicies, 0, 4)
            = [4, 5, 6, 7, 0, 1, 2, 3]
    """
    selected = to_shuffle[start:start+width]
    to_shuffle[start:start+width] = to_shuffle[start+width:start+2*width]
    to_shuffle[start+width:start+2*width] = selected


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
            num_dims (int): number of dimensions in encoding transformation
            encoder_type (string): specifies type of encoder to use - UNet or ResNet based
            num_encoding_layers (int): number of conv layers to use in encoding (also determines num in decoding)
            dimension_bg (int): num pixels on one side of square background image
            dimension_fg (int): num pixels on one side of square foreground image
            dimension_3d (int): size of 3d geometry latent space. Must be divisible by 3
            latent_dropout (float): dropout rate between latent space and PoseEstimator
            swap_appearance (bool): if True swap the appearance of the primary image with another
            shuffle_3d (bool): if True, swap the geometry of primary image with another
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
                 num_dims=3,
                 encoder_type='UNet',
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3*64,
                 latent_dropout=0.3,
                 swap_appearance=True,
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
        self.out_channels = out_channels
        self.is_deconv = is_deconv
        self.upper_bilinear = upper_bilinear
        self.lower_bilinear = lower_bilinear
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.dimension_bg = dimension_bg
        self.dimension_fg = dimension_fg
        self.dimension_3d = dimension_3d
        self.swap_appearance = swap_appearance
        self.shuffle_3d = shuffle_3d
        self.num_encoding_layers = num_encoding_layers
        self.output_types = output_types
        self.encoder_type = encoder_type
        self.implicit_rotation = implicit_rotation
        self.num_cameras = num_cameras
        self.skip_background = skip_background
        self.num_joints = num_joints
        self.num_dims = num_dims
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

        # creates MLP from latent space to 3D prediction
        self.to_pose = mlp.MLPFromLatent(d_in=self.dimension_3d,
                                          d_hidden=2048,
                                          d_out=self.num_joints*self.num_dims,
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

        #  Create and initialize decoder  ####################################################################

        self.decoder = unet_decoder.UnetDecoder(upper_bilinear=self.upper_bilinear,
                                                lower_bilinear=self.lower_bilinear,
                                                is_deconv=self.is_deconv,
                                                num_decoding_layers=self.num_encoding_layers,
                                                out_channels=self.out_channels,
                                                filters=self.filters,
                                                bottleneck_resolution=self.bottleneck_resolution,
                                                skip_background=self.skip_background,
                                                dimension_fg=self.dimension_fg)


    def forward(self, input_dict):
        input = input_dict['img_crop']
        device = input.device
        batch_size = input.size()[0]

        ########################################################
        # Determine shuffling
        shuffled_appearance = list(range(batch_size))
        shuffled_pose = list(range(batch_size))        # TODO: what is this? why would we swap pose?
        num_pose_subbatches = batch_size//np.maximum(self.subbatch_size, 1)  # TODO: confirm: the number of pairs of subbatches in the batch with the same user but different times ex. 8//4 = 2
        rotation_by_user = not self.training and 'external_rotation_cam' in input_dict.keys()

        if not rotation_by_user:
            if self.swap_appearance and self.training:
                for i in range(0, num_pose_subbatches): # shuffle each subbatch
                    _shuffle_within_segment(to_shuffle=shuffled_appearance,
                                            start=i * self.subbatch_size,
                                            end=(i+1) * self.subbatch_size,
                                            training=self.training,
                                            num_cameras=self.num_cameras)

                print("shuffled appear: ", shuffled_appearance)

                for i in range(0, num_pose_subbatches//2): # flip each subbatch with its neighbour to the right
                    _flip_segment(shuffled_appearance, start=i*2*self.subbatch_size, width=self.subbatch_size)

            print("shuffled then flipped appear: ", shuffled_appearance)

            if self.shuffle_3d:  # TODO: why shuffle latent variables? note: True in config_train_encodeDecode but False in config_train_encodeDecode_pose
                for i in range(0, num_pose_subbatches):
                    _shuffle_within_segment(to_shuffle=shuffled_pose,
                                            start=i * self.subbatch_size,
                                            end=(i+1) * self.subbatch_size,
                                            training=self.training,
                                            num_cameras=self.num_cameras)
            print("subbatch_size: ", self.subbatch_size)
            print("shuffled_pose: ", shuffled_pose)
        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size  # ex. [-1, -1, -1, -1, -1, -1, -1, -1]
        for i, v in enumerate(shuffled_pose):  # ex. if shuffled_pose = [2, 0, 1, 3, 4, 7, 6, 5] -> shuffled_pose_inv = [1, 2, 0, 3, 4, 7, 6, 5]
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
            conv1_bg_shuffled = getattr(self, 'conv_1_stage_bg0')(input_bg_shuffled)            #  TODO: why not used?

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
        map_from_3d = self.from_latent(latent_3d_rotated.view(batch_size, -1))
        output_img_shuffled = self.decoder(map_from_3d, latent_fg_shuffled, conv1_bg_shuffled, batch_size)



        # map_from_3d = self.from_latent(latent_3d_rotated.view(batch_size, -1))  # the output from NN that takes in the latent variable
        # map_width = self.bottleneck_resolution #out_enc_conv.size()[2]
        # map_channels = self.filters[self.num_encoding_layers-1] #out_enc_conv.size()[1]
        # if has_fg:
        #     latent_fg_shuffled_replicated = latent_fg_shuffled.view(batch_size,
        #                                                             self.dimension_fg, 1,
        #                                                             1).expand(batch_size,
        #                                                                       self.dimension_fg,
        #                                                                       map_width,
        #                                                                       map_width)
        #     latent_shuffled = torch.cat([latent_fg_shuffled_replicated,
        #                                  map_from_3d.view(batch_size,
        #                                                   map_channels-self.dimension_fg,
        #                                                   map_width, map_width)], dim=1)
        # else:
        #     latent_shuffled = map_from_3d.view(batch_size, map_channels, map_width, map_width)
        #
        # if self.skip_connections:
        #     assert False
        # else:
        #     out_deconv = latent_shuffled
        #     for li in range(1, self.num_encoding_layers-1):
        #         out_deconv = getattr(self, 'upconv_' + str(li) + '_stage0')(out_deconv)
        #
        #     if self.skip_background:
        #         out_deconv = getattr(self, 'upconv_' + \
        #                              str(self.num_encoding_layers-1) + \
        #                              '_stage0')(conv1_bg_shuffled, out_deconv)
        #     else:
        #         out_deconv = getattr(self, 'upconv_' + \
        #                              str(self.num_encoding_layers-1) + \
        #                              '_stage0')(out_deconv)
        #
        # output_img_shuffled = getattr(self, 'final_stage0')(out_deconv)

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
