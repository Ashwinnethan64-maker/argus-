import os
import sys

# Add baseline repo to sys.path first so custom modules are registered
REPO_DIR = os.path.abspath(os.path.join("baseline_qfdet_repo", "mmdet-rgbtdroneperson-main"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Ensure torch DLL path is registered on Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'C:\env\Lib\site-packages\torch\lib')

import json
import cv2
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector
from mmdet.models import build_detector

import mmdet.datasets.vtuav
import mmdet.models.detectors.qfdet

DATA_ROOT = os.path.abspath(os.path.join("VTUAV_subset", "VTUAV_subset"))
BASELINE_CKPT = os.path.abspath(os.path.join("checkpoints", "qfdet_vtuav.pth"))
CMAGM_CKPT = os.path.abspath(os.path.join("work_dirs", "qfdet_cmagm_stage3", "latest.pth"))
OUTPUT_DIR = os.path.abspath("visual_comparison_results")

def build_qfdet_config(spectral_pair):
    ann_file = os.path.join(DATA_ROOT, "annotations", "test.json")
    img_prefix = DATA_ROOT + "/"
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
                score_thr=0.25,
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
    return mmcv.Config(config)

def draw_bboxes(img, bboxes, color=(0, 255, 0), label_prefix=""):
    img_draw = img.copy()
    for bbox in bboxes:
        x1, y1, x2, y2, score = bbox
        if score < 0.30:
            continue
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label_prefix}{score:.2f}"
        cv2.putText(img_draw, text, (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img_draw

def run_visual_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "="*70)
    print("GENERATING SIDE-BY-SIDE INFERENCE VISUAL COMPARISONS")
    print("="*70)
    
    cfg_base = build_qfdet_config(("VTUAV_co/test/images", "VTUAV_ir/test/images"))
    
    # Build dataset & loader
    from mmdet.datasets import build_dataset, build_dataloader
    dataset = build_dataset(cfg_base.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    from mmdet.utils import build_dp
    
    # Load Baseline detector
    base_model = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg'))
    load_checkpoint(base_model, BASELINE_CKPT, map_location='cpu')
    base_model.eval()
    base_model = build_dp(base_model, 'cpu', device_ids=[])
    
    # Load CMAGM detector
    cmagm_model = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg'))
    load_checkpoint(cmagm_model, CMAGM_CKPT, map_location='cpu')
    cmagm_model.eval()
    cmagm_model = build_dp(cmagm_model, 'cpu', device_ids=[])
    
    from mmdet.apis import single_gpu_test
    
    print("Running Baseline QFDet Inference on Test Set...")
    res_base_all = single_gpu_test(base_model, data_loader)
    
    print("Running Stage 3 CMAGM QFDet Inference on Test Set...")
    res_cmagm_all = single_gpu_test(cmagm_model, data_loader)
    
    for idx in range(min(10, len(dataset))):
        img_info = dataset.data_infos[idx]
        file_name = img_info['filename']
        
        rgb_path = os.path.join(DATA_ROOT, "VTUAV_co", "test", "images", file_name)
        ir_path = os.path.join(DATA_ROOT, "VTUAV_ir", "test", "images", file_name)
        
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            continue
            
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        res_base = res_base_all[idx]
        res_cmagm = res_cmagm_all[idx]
        
        base_bboxes = res_base[0] if len(res_base) > 0 else np.empty((0, 5))
        cmagm_bboxes = res_cmagm[0] if len(res_cmagm) > 0 else np.empty((0, 5))
        
        rgb_base = draw_bboxes(rgb_img, base_bboxes, color=(0, 0, 255), label_prefix="Base: ")
        ir_base = draw_bboxes(ir_img, base_bboxes, color=(0, 0, 255), label_prefix="Base: ")
        
        rgb_cmagm = draw_bboxes(rgb_img, cmagm_bboxes, color=(0, 255, 0), label_prefix="CMAGM: ")
        ir_cmagm = draw_bboxes(ir_img, cmagm_bboxes, color=(0, 255, 0), label_prefix="CMAGM: ")
        
        top_row = np.hstack([rgb_base, ir_base])
        bot_row = np.hstack([rgb_cmagm, ir_cmagm])
        
        banner_top = np.zeros((40, top_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner_top, f"BASELINE QFDET (Naive Fusion) | Test Pair #{idx+1}: {file_name}", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        banner_bot = np.zeros((40, bot_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner_bot, f"STAGE 3 CMAGM QFDET (Attention Gated) | Test Pair #{idx+1}: {file_name}", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        grid = np.vstack([banner_top, top_row, banner_bot, bot_row])
        
        h, w = grid.shape[:2]
        grid_resized = cv2.resize(grid, (1280, int(h * (1280 / w))))
        
        out_name = f"comparison_pair_{idx+1:02d}_{os.path.splitext(file_name)[0]}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, grid_resized)
        print(f"Saved visual comparison panel to: {out_path}")
        
    print(f"\nCOMPLETED GENERATING VISUAL COMPARISON OVERLAYS IN: {OUTPUT_DIR}")

if __name__ == '__main__':
    run_visual_comparison()
