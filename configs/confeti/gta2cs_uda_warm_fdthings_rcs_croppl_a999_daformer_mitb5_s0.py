# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
    # '../_base_/schedules/poly10.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
contrastive_head = dict(
    type='ContrastiveHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    num_classes=19,
    proto_dims=256,
    channels=256,
    tau=.1,
    num_samples=1,
    reg_weight=.5,
    reg_norm=55.94434060416236,
    loss_weight=.3,
    input_transform='multiple_select')
cam_head = dict(
    type='CAMHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    num_classes=19,
    channels=256,
    loss_weight=1,
    input_transform='multiple_select')
model = dict(
    auxiliary_head=[dict(
        type='StylizationHead',
        in_channels=[3, 128, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=19,
        input_transform='multiple_select',
        lambda_gan=1,
        lambda_nce=1,
    ), contrastive_head, cam_head])

uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    debug_img_interval=100,
    in_index=[0, 1, 2, 3],
    in_channels=[64, 128, 320, 512],
    channels=256,
    style_index=[0],
    contrast_index=[0, 1, 2, 3],
    use_fp16=True,
    enable_cut=True)

data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            decoder=dict(lr_mult=0.0),
            discriminator=dict(lr_mult=0.0),
            cut=dict(lr_mult=0.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

# Meta Information for Result Analysis
name = 'gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
