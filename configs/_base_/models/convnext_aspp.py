checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='ASPPHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        input_transform='resize_concat'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))