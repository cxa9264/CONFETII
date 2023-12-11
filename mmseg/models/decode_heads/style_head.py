import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torchvision.models as models
from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..losses.gan_loss import GANLoss
from ..losses.patchnce_loss import PatchNCELoss

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

@HEADS.register_module()
class StylizationHead(BaseDecodeHead):
    """Stylization Head.
    """

    def __init__(self, num_blocks=4, norm_layer=nn.InstanceNorm2d, padding_type='reflect',
                 use_dropout=False, use_bias=True, lambda_gan=1, lambda_nce=1, num_features=256, share_cut=False, **kwargs):

        super(StylizationHead, self).__init__(**kwargs)
        
        self.num_blocks = num_blocks
        self.norm_layer = norm_layer
        self.padding_type = padding_type
        self.use_dropout = use_dropout
        self.use_bias = use_bias
        self.lambda_gan = lambda_gan
        self.lambda_nce = lambda_nce
        self.num_features = num_features
        self.share_cut = share_cut

        del self.conv_seg

        decoder = []
        in_c = self.in_channels[-1] if not self.share_cut else int(np.sum(self.in_channels))
        decoder += [
            nn.Conv2d(in_c, self.channels, kernel_size=1, bias=self.use_bias),
            self.norm_layer(self.channels),
            nn.ReLU(True),
        ]
        for i in range(self.num_blocks):
            decoder.append(ResnetBlock(self.channels, self.padding_type, 
                                            self.norm_layer, self.use_dropout, 
                                            self.use_bias))
        

        decoder += [
            Upsample(self.channels),
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=3, padding=1, bias=self.use_bias),
            self.norm_layer(self.channels // 2),
            nn.ReLU(True),

            Upsample(self.channels // 2),
            nn.Conv2d(self.channels // 2, self.channels // 4, kernel_size=3, padding=1, bias=self.use_bias),
            self.norm_layer(self.channels // 4),
            nn.ReLU(True),
        ]

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.channels // 4, 3, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.decoder = nn.Sequential(*decoder)

        self.discriminator = NLayerDiscriminator(3, ndf=64, n_layers=3, norm_layer=self.norm_layer)

        self.gan_loss = GANLoss(gan_mode='lsgan')
        self.patchnce_loss = PatchNCELoss()

    
    def compute_D_loss(self):

        # Fake
        set_requires_grad(self.discriminator, True)
        fake = self.s2t_img.detach()
        pred_fake = self.discriminator(fake)
        loss_D_fake = self.gan_loss(pred_fake, False).mean()
        # Real
        pred_real = self.discriminator(self.real_target)
        loss_D_real = self.gan_loss(pred_real, True).mean()

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss = dict()
        loss['gan_D_fake'] = loss_D_fake
        loss['gan_D_real'] = loss_D_real
        loss['gan_D_loss'] = loss_D # * self.lambda_gan
        return loss

    def compute_G_loss(self):

        loss = dict()

        set_requires_grad(self.discriminator, False)
        loss['gan_G_loss'] = self.gan_loss(
            self.discriminator(self.s2t_img), True).mean() * self.lambda_gan

        return loss

    def compute_NCE_loss(self, inputs, projections, proto=None):
        loss = dict()

        features = self._transform_inputs(inputs)

        for i, (src_features, tgt_features) in enumerate(zip(
                                                    self.features, features)):
            
            src_feat = F.normalize(projections[i](src_features), dim=1, p=2)
            tgt_feat = F.normalize(projections[i](tgt_features), dim=1, p=2)

            b, c, h, w = src_feat.shape
            src_feat = src_feat.permute(0, 2, 3, 1).reshape(b*h*w, c)
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).reshape(b*h*w, c)

            idx = torch.randperm(src_feat.shape[0])[:self.num_features]
            feat_k = src_feat[idx]
            feat_q = tgt_feat[idx]
            loss[f'patchnce_{i}_loss'] = self.patchnce_loss(feat_q, feat_k) * self.lambda_nce / len(features)

        return loss


    def forward(self, inputs, save_feature=False):
        """Forward function."""
        features = self._transform_inputs(inputs)

        if self.share_cut:
            inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners) for x in features
                ]
            inputs = torch.cat(inputs, dim=1)
        else:
            inputs = features[-1]

        # generate fake image in target domain
        x = self.decoder(inputs)

        # save for loss
        if save_feature:
            self.features = features
            self.s2t_img = x

        return x

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        return dict()
        
    def loss(self, inputs, seg_label):
        raise NotImplementedError


