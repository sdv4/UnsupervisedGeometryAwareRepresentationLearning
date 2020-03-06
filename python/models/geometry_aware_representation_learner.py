""" Geometry-aware Representation Learner abstract class
"""
import random
import torch
import torch.nn as nn



class GeometryAwareLearner(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def _shuffle_segment(self, list, start, end, training, num_cameras):
        """
            Helper method for EncoderDecoder forward function.
        """
        selected = list[start:end]
        if training:
            if 0 and end-start == 2: # Not enabled in ECCV submission, disbled now too HACK
                prob = np.random.random([1])
                # assuming four cameras, make it more often that one of
                # the others is taken, rather than just autoencoding
                # (no flip, which would happen 50% otherwise):
                if prob[0] > 1/num_cameras:
                    selected = selected[::-1] # reverse
                else:
                    pass # let it as it is
            else:
                random.shuffle(selected)

        else: # deterministic shuffling for testing
            selected = np.roll(selected, 1).tolist()
        list[start:end] = selected

    def _flip_segment(self, list, start, width):
        """
            Helper method for EncoderDecoder forward function.
        """
        selected = list[start:start+width]
        list[start:start+width] = list[start+width:start+2*width]
        list[start+width:start+2*width] = selected

    def forward(self, x):

        pass
