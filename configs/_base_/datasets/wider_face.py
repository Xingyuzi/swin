# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = '/home/techart/xyz/swin/swin_master/data/WIDERFace/'
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
    # dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False), # XYZ TODO 2022.12.12
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug', # 可以进行多尺度测试
#         # img_scale=[(300, 300), (1024, 1024)], # 13.XYZ TODO
#         img_scale=(1024, 1024),
#         # img_scale=(1333, 1333), # TODO
#         # img_scale=(1024, 1024), # 13.XYZ TODO
#         # img_scale=(2400, 2400), # XYZ TODO
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=False),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# 14.XYZ TODO
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug', # 可以进行多尺度测试
        # img_scale=[(300, 300), (1024, 1024)], # 13.XYZ TODO
        # img_scale=(1024, 1024),
        # img_scale=(1333, 1333), # TODO
        img_scale=(1024, 1024), # 13.XYZ TODO
        # img_scale=(2400, 2400), # XYZ TODO
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomFlip', flip_ratio=0.5), # 14.XYZ TODO
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # samples_per_gpu=1, # 8.XYZ TODO
    # workers_per_gpu=1, # 8.XYZ TODO
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type, # 使用的数据集，例：type='WIDERFaceDataset'
            ann_file=data_root + 'train2.txt', # 训练数据的信息，例：widerface中的图片名字
            img_prefix=data_root + 'WIDER_train/', # 训练数据的位置
            min_size=17,
            pipeline=train_pipeline)), # 处理流程
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val2.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val2.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=test_pipeline))
