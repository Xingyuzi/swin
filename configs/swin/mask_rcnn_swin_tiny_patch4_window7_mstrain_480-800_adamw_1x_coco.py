_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    # '../_base_/datasets/coco_instance.py',
    '../_base_/datasets/wider_face.py', # TODO
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        # depths=[2, 2], # 16.XYZ TODO
        # depths=[2, 2, 4, 2], # 16.XYZ TODO
        num_heads=[3, 6, 12, 24],
        # num_heads=[3, 6], # 17.XYZ TODO
        window_size=7,
        # window_size=10, # 18.XYZ TODO
        ape=False,
        drop_path_rate=0.1,
        patch_norm=True,
        # out_indices=(0, 1), # 20.XYZ TODO
        # use_checkpoint=False 
        use_checkpoint=True # TODO
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))
    # neck=dict(in_channels=[96, 192],
    #     num_outs=3,
    # )) # 19.XYZ TODO

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', flip_ratio=0.5), # 以0.5的概率翻转图像
#     dict(type='AutoAugment', # 随机选择一种增强方法
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                                  (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                                  (736, 1333), (768, 1333), (800, 1333)],# TODO
#                       multiscale_mode='value',
#                       keep_ratio=True)
#                 # dict(type='Resize',
#                 #       img_scale=[(608, 1333), (800, 1333), (1024,1333)],# TODO
#                 #       multiscale_mode='value',
#                 #       keep_ratio=True) # XYZ TODO
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                 # dict(type='Resize',
#                 #       img_scale=[(600, 1333), (800, 1333), (1000, 1333)],
#                 #       multiscale_mode='value',
#                 #       keep_ratio=True), # XYZ TODO
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(384, 600),
#                     #   crop_size=(300,300), # XYZ TODO
#                       allow_negative_crop=True
#                     #   allow_negative_crop=False # XYZ TODO
#                       ),
#                 #  dict(type='RandomCrop',
#                 #       crop_type='relative_range',
#                 #       crop_size=(384, 600),
#                 #     #   crop_size=(0.2, 0.2), # XYZ TODO
#                 #       allow_negative_crop=True
#                 #     #   allow_negative_crop=False # XYZ TODO
#                 #       ),
#                  dict(type='Resize',
#                       img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                  (576, 1333), (608, 1333), (640, 1333),
#                                  (672, 1333), (704, 1333), (736, 1333),
#                                  (768, 1333), (800, 1333)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#                 # XYZ TODO
#                 #  dict(type='Resize',
#                 #       img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                 #                  (576, 1333), (608, 1333), (640, 1333),
#                 #                  (672, 1333), (704, 1333), (736, 1333),
#                 #                  (768, 1333), (800, 1333), (1024, 1333)],
#                 #       multiscale_mode='value',
#                 #       override=True,
#                 #       keep_ratio=True)
#              ]
#          ]),
#     dict(type='Normalize', **img_norm_cfg), # 图像减去均值等其操作
#     dict(type='Pad', size_divisor=32), # Pad an image to ensure each edge to be multiple to some number
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]

# 4.wider_face增强策略
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False), # 原来是resize到300
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32), # 15.Pad an image to ensure each edge to be multiple to some number
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
# 7.XYZ TODO
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[8, 11])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# 6.XYZ TODO
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    # grad_clip=None,
    # 5.XYZ TODO
    _delete_=True, # 为了clip额外添加的 XYZ TODO
    grad_clip=dict(max_norm=35, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
