# model settings
model = dict(
    type='MaskRCNN',
    pretrained=None,
    # pretrained='/home/techart/xyz/swin/swin_master/pretrain/swin_tiny_patch4_window7_224.pth', # XYZ TODO
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7, # XYZ TODO ori=7
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768], # 输入的各个stage的通道数
        out_channels=256, # 输出的特征层的通道数
        num_outs=5), # 输出的特征层数量
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[8], # 只有一种scale吗，很奇怪
            # ratios=[0.5, 1.0, 2.0], # 三种长宽比
            scales=[4, 8], # 只有一种scale吗，很奇怪 # 9.XYZ TODO
            ratios=[0.7, 1.0, 1.5], # 三种长宽比 # 9.XYZ TODO
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0)), # 10.XYZ TODO
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            # num_classes=80,
            num_classes=1, # TODO
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            loss_bbox=dict(type='L1Loss', loss_weight=2.0)), # 11.XYZ TODO
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=None # TODO
        # dict(
            # type='FCNMaskHead',
            # num_convs=4,
            # in_channels=256,
            # conv_out_channels=256,
            # # num_classes=80,
            # num_classes=1, # TODO
            # loss_mask=dict(
            #     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
                ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # pos_iou_thr=0.7, # 将最大IoU大于等于pos_iou_thr的bbox标记为正样本
                # neg_iou_thr=0.3, # 将最大IoU在[0，neg_iou_thr）之间的bbox标记为负样本
                # min_pos_iou=0.3, # 正样本阈值下限，仅当开启匹配低质量时该阈值才发挥作用
                # 1.XYZ TODO
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.,
                match_low_quality=True, # 尽量要使所有gt都有一个anchor，不能有的gt没匹配上anchor
                ignore_iof_thr=-1),
            # 上一步可以区分正负和忽略样本，但是依然存在大量的正负样本不平衡问题， 
            # 解决办法可以通过正负样本采样或者loss上面一定程度解决
            sampler=dict(
                type='RandomSampler',
                num=256, # 取256个ROI送去训练
                pos_fraction=0.5, # 正样本占50%
                neg_pos_ub=-1,
                add_gt_as_proposals=False), # 如果是True的话，当正样本不够的时候就用gt_box当做proposal
            # XYZ TODO
            # sampler=dict( # 12.XYZ TODO
            #     type='OHEMSampler', # 12.XYZ TODO
            #     num=512,
            #     pos_fraction=0.25,
            #     neg_pos_ub=-1,
            #     add_gt_as_proposals=True),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
            

        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            # sampler=dict( # 12.XYZ TODO
            #     type='OHEMSampler', # 12.XYZ TODO
            #     num=512,
            #     pos_fraction=0.2,
            #     neg_pos_ub=-1,
            #     add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            # # 2.XYZ TODO
            # nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            # 3.XYZ TODO
            # score_thr=0.02,
            # nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100,
            mask_thr_binary=0.5)))
