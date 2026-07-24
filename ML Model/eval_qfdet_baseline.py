import os
import sys

# Add baseline repo to sys.path first so custom modules (vtuav, qfdet) are loaded
REPO_DIR = os.path.abspath(os.path.join("baseline_qfdet_repo", "mmdet-rgbtdroneperson-main"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Ensure torch DLL path is registered on Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'C:\env\Lib\site-packages\torch\lib')

import time
import json
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector

import mmdet.datasets.vtuav
import mmdet.models.detectors.qfdet

DATA_ROOT = os.path.abspath(os.path.join("VTUAV_subset", "VTUAV_subset"))
CHECKPOINT_PATH = os.path.abspath(os.path.join("checkpoints", "qfdet_vtuav.pth"))

from mmdet.apis import single_gpu_test
from mmdet.utils import build_dp

def evaluate_model(config_dict, checkpoint_path, split_name, mode_name):
    print(f"\n" + "="*70)
    print(f"EVALUATING MODE: {mode_name} | SPLIT: {split_name}")
    print("="*70)
    
    cfg = mmcv.Config(config_dict)
    cfg.model.pretrained = None
    
    # Build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    # Build detector model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load weights
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Calculate params & model size
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    print(f"Model File Size : {model_size_mb:.2f} MB")
    print(f"Total Parameters: {num_params:,} ({num_params/1e6:.2f} M)")
    print(f"Trainable Params: {num_trainable:,}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_dp(model, device, device_ids=[0] if device == 'cuda' else [])
    
    start_time = time.time()
    results = single_gpu_test(model, data_loader)
    total_time = time.time() - start_time
    
    avg_inference_time_ms = (total_time / len(dataset)) * 1000
    fps = len(dataset) / total_time
    
    print(f"Inference Time  : {avg_inference_time_ms:.2f} ms / image")
    print(f"FPS             : {fps:.2f} FPS")
    
    # Evaluate using COCO metric
    print("\n--- Running Official COCO Evaluation ---")
    eval_results = dataset.evaluate(results, metric='bbox')
    
    metrics_summary = {
        "mode": mode_name,
        "split": split_name,
        "model_size_mb": model_size_mb,
        "total_params": num_params,
        "inference_time_ms": avg_inference_time_ms,
        "fps": fps,
        "metrics": {k: float(v) for k, v in eval_results.items() if isinstance(v, (int, float, np.number))}
    }
    
    return metrics_summary

def get_base_config(split='val', modality='both'):
    img_prefix = DATA_ROOT + "/"
    ann_file = os.path.join(DATA_ROOT, "annotations", f"{split}.json")
    
    if modality == 'both':
        spectral_pair = (f"VTUAV_co/{split}/images", f"VTUAV_ir/{split}/images")
    elif modality == 'rgb':
        spectral_pair = (f"VTUAV_co/{split}/images", f"VTUAV_co/{split}/images")
    elif modality == 'thermal':
        spectral_pair = (f"VTUAV_ir/{split}/images", f"VTUAV_ir/{split}/images")

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
                num_classes=3, # Match checkpoint 3 classes
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
                num_classes=3, # Match checkpoint 3 classes
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
    return config

if __name__ == '__main__':
    results_file = "stage2_benchmark_results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_benchmark_results = json.load(f)
        except Exception:
            all_benchmark_results = {}
    else:
        all_benchmark_results = {}
    
    modes = ['both', 'rgb', 'thermal']
    splits = ['val', 'test']
    
    for split in splits:
        for mode in modes:
            key = f"{mode}_{split}"
            if key in all_benchmark_results:
                print(f"\nSkipping {key} - already evaluated.")
                continue
                
            cfg_dict = get_base_config(split=split, modality=mode)
            res = evaluate_model(cfg_dict, CHECKPOINT_PATH, split_name=split, mode_name=mode)
            all_benchmark_results[key] = res
            
            with open(results_file, "w") as f:
                json.dump(all_benchmark_results, f, indent=2)
            print(f"Saved progress to {results_file} after evaluating {key}")

    print("\nSaved full Stage 2 benchmark results to stage2_benchmark_results.json")
