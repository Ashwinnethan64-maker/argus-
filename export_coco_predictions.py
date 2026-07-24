import os
import sys

# Add baseline repo to sys.path first
REPO_DIR = os.path.abspath(os.path.join("baseline_qfdet_repo", "mmdet-rgbtdroneperson-main"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Ensure torch DLL path is registered on Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'C:\env\Lib\site-packages\torch\lib')

import json
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import build_dp

import mmdet.datasets.vtuav
import mmdet.models.detectors.qfdet

DATA_ROOT = os.path.abspath(os.path.join("VTUAV_subset", "VTUAV_subset"))
STAGE3_CHECKPOINT_PATH = os.path.abspath(os.path.join("work_dirs", "qfdet_cmagm_stage3", "latest.pth"))
OUTPUT_JSON_PATH = os.path.abspath("predictions_coco_format.json")

def export_predictions():
    print("\n" + "="*70)
    print("EXPORTING OFFICIAL COCO PREDICTIONS JSON FOR TEST SET")
    print("="*70)
    
    ann_file = os.path.join(DATA_ROOT, "annotations", "test.json")
    img_prefix = DATA_ROOT + "/"
    spectral_pair = ("VTUAV_co/test/images", "VTUAV_ir/test/images")
    
    config = dict(
        model=dict(
            type='QFDet',
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
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        ),
        data=dict(
            test=dict(
                type='VTUAVdet',
                ann_file=ann_file,
                img_prefix=img_prefix,
                pipeline=[
                    dict(type='LoadImagePairFromFile', spectrals=spectral_pair),
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
        )
    )
    
    cfg = mmcv.Config(config)
    cfg.model.pretrained = None
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, STAGE3_CHECKPOINT_PATH, map_location='cpu')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_dp(model, device, device_ids=[0] if device == 'cuda' else [])
    
    results = single_gpu_test(model, data_loader)
    
    # Format predictions into COCO JSON format
    coco_results = []
    for img_idx, result in enumerate(results):
        img_id = dataset.data_infos[img_idx]['id']
        for cat_id, bboxes in enumerate(result):
            # Map category ID 0 to 'person'
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox.tolist()
                w = x2 - x1
                h = y2 - y1
                coco_results.append({
                    "image_id": int(img_id),
                    "category_id": 0,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 4)
                })
                
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(coco_results, f, indent=2)
        
    print(f"\nSUCCESSFULLY EXPORTED COCO PREDICTIONS TO: {OUTPUT_JSON_PATH}")
    print(f"Total Predictions Exported: {len(coco_results):,}")

if __name__ == '__main__':
    export_predictions()
