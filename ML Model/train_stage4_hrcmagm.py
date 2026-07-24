import os
import sys

# Add baseline repo to sys.path first so custom modules are registered
REPO_DIR = os.path.abspath(os.path.join("baseline_qfdet_repo", "mmdet-rgbtdroneperson-main"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Ensure torch DLL path is registered on Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'C:\env\Lib\site-packages\torch\lib')

import time
import mmcv
import torch
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

import mmdet.datasets.vtuav
import mmdet.models.detectors.qfdet

DATA_ROOT = os.path.abspath(os.path.join("VTUAV_subset", "VTUAV_subset"))
PRETRAINED_CHECKPOINT = os.path.abspath(os.path.join("checkpoints", "qfdet_vtuav.pth"))
WORK_DIR = os.path.abspath(os.path.join("work_dirs", "qfdet_hrcmagm_stage4"))

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("\n" + "="*70)
    print("STARTING STAGE 4 FINE-TUNING: HIGH-RESOLUTION SMALL-OBJECT CMAGM (HR-CMAGM)")
    print(f"Work Directory: {WORK_DIR}")
    print("="*70)

    train_ann_file = os.path.join(DATA_ROOT, "annotations", "train.json")
    val_ann_file = os.path.join(DATA_ROOT, "annotations", "val.json")
    img_prefix = DATA_ROOT + "/"
    spectral_pair = ("VTUAV_co/train/images", "VTUAV_ir/train/images")
    val_spectral_pair = ("VTUAV_co/val/images", "VTUAV_ir/val/images")

    config = dict(
        gpu_ids=[0],
        model=dict(
            type='QFDet',
            pretrained=PRETRAINED_CHECKPOINT,
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch'),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_output',
                num_outs=5),
            bbox_head=dict(
                type='ATSSQHead',
                num_classes=3,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                centerness=1,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
            bbox_prehead=dict(
                type='QFDetPreHead',
                num_classes=3,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                centerness=1,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
                loss_quality=dict(type='MSELoss', loss_weight=0.5)),
            base_fusion='cat',
            quality_attention=True,
            poolupsample=1,
            reweight=True,
            train_cfg=dict(
                assigner=dict(type='ATSSAssigner', topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        ),
        data=dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(
                type='VTUAVdet',
                ann_file=train_ann_file,
                img_prefix=img_prefix,
                pipeline=[
                    dict(type='LoadImagePairFromFile', spectrals=spectral_pair),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='MultiNormalize',
                        mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
                        std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
                ]
            ),
            val=dict(
                type='VTUAVdet',
                ann_file=val_ann_file,
                img_prefix=img_prefix,
                pipeline=[
                    dict(type='LoadImagePairFromFile', spectrals=val_spectral_pair),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(640, 512),
                        flip=False,
                        transforms=[
                            dict(type='Resize', keep_ratio=True),
                            dict(type='RandomFlip'),
                            dict(
                                type='MultiNormalize',
                                mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
                                std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]),
                                to_rgb=True),
                            dict(type='Pad', size_divisor=32),
                            dict(type='DefaultFormatBundle'),
                            dict(type='Collect', keys=['img']),
                        ])
                ]
            ),
            test=dict(
                type='VTUAVdet',
                ann_file=val_ann_file,
                img_prefix=img_prefix,
                pipeline=[
                    dict(type='LoadImagePairFromFile', spectrals=val_spectral_pair),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(640, 512),
                        flip=False,
                        transforms=[
                            dict(type='Resize', keep_ratio=True),
                            dict(type='RandomFlip'),
                            dict(
                                type='MultiNormalize',
                                mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
                                std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]),
                                to_rgb=True),
                            dict(type='Pad', size_divisor=32),
                            dict(type='DefaultFormatBundle'),
                            dict(type='Collect', keys=['img']),
                        ])
                ]
            )
        ),
        optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
        optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)),
        lr_config=dict(
            policy='step',
            warmup='linear',
            warmup_iters=200,
            warmup_ratio=0.001,
            step=[2]),
        runner=dict(type='EpochBasedRunner', max_epochs=3),
        checkpoint_config=dict(interval=1, by_epoch=True),
        log_config=dict(
            interval=10,
            hooks=[dict(type='TextLoggerHook')]),
        log_level='INFO',
        device='cpu',
        resume_from=None,
        load_from=None,
        workflow=[('train', 1)],
        evaluation=dict(interval=1, metric='bbox'),
        work_dir=WORK_DIR,
        seed=42
    )

    cfg = mmcv.Config(config)
    
    # Build dataset & detector
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    # Start training
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict()
    )

if __name__ == '__main__':
    main()
