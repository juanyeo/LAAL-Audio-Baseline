import torch.nn as nn
import numpy as np
import tempfile
import random
import torch
import timm
import os
import wget

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from random import randrange
from .unireplknet_arch import UniRepLKNet, initialize_with_pretrained

import utils.logging as logging

from config.defaults import get_cfg
# from .build import MODEL_REGISTRY


logger = logging.get_logger(__name__)

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class UniRepLKNetModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, cfg):
        super(UniRepLKNetModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        label_dim = cfg.MODEL.NUM_CLASSES[0]
        verbose = True

        if verbose == True:
            logger.info('---------------UniRepLKNet Model Summary---------------')
            # logger.info('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        model = UniRepLKNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4,
                              in_chans=1, disable_iGEMM=True, num_classes=label_dim)
        
        # model = initialize_with_pretrained(model, 'unireplknet_b', False, True, False)
        self.v = model
        # self.v =  UniRepLKNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4,
        #                       in_chans=1, disable_iGEMM=True, num_classes=label_dim)
        
        # self.original_embedding_dim = 768
        # label_dim = 527
        # self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # input dim ESC50: [Batch, 1, 128, 512] EPIC: [Batch, 1, 128, 1024]
        if isinstance(x, list):
            x = x[0].transpose(2, 3)
        else:
            x = x.unsqueeze(1)
            x = x.transpose(2, 3)

        x = self.v(x)
        # x = self.mlp_head(x)
        return x