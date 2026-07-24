import os
import sys
import json
import numpy as np

DATA_ROOT = os.path.abspath(os.path.join("VTUAV_subset", "VTUAV_subset"))

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes in [x1, y1, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)
    
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def get_scale_category(area):
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"

def analyze_split_gt(split='test'):
    ann_file = os.path.join(DATA_ROOT, "annotations", f"{split}.json")
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return None
        
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
        
    gt_by_scale = {"small": 0, "medium": 0, "large": 0}
    gt_boxes_per_image = {}
    
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        area = ann.get('area', ann['bbox'][2] * ann['bbox'][3])
        scale = get_scale_category(area)
        gt_by_scale[scale] += 1
        
        if img_id not in gt_boxes_per_image:
            gt_boxes_per_image[img_id] = []
        gt_boxes_per_image[img_id].append({
            'id': ann['id'],
            'bbox': ann['bbox'],
            'scale': scale
        })
        
    return {
        "num_images": len(coco_data.get('images', [])),
        "num_annotations": len(coco_data.get('annotations', [])),
        "scale_distribution": gt_by_scale,
        "gt_boxes_per_image": gt_boxes_per_image
    }

def analyze_predictions(pred_json_path, gt_info, score_thr=0.30, iou_thr=0.50):
    if not os.path.exists(pred_json_path):
        print(f"Predictions JSON not found: {pred_json_path}")
        return None
        
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
        
    gt_boxes_per_image = gt_info['gt_boxes_per_image']
    
    # Filter predictions above threshold
    filtered_preds = [p for p in preds if p.get('score', 0) >= score_thr]
    
    total_preds = len(filtered_preds)
    tp_count = 0
    fp_count = 0
    
    scale_tp = {"small": 0, "medium": 0, "large": 0}
    scale_fp = {"small": 0, "medium": 0, "large": 0}
    scale_gt = dict(gt_info['scale_distribution'])
    
    # Per-image matching
    preds_by_img = {}
    for p in filtered_preds:
        img_id = p['image_id']
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(p)
        
    for img_id, gt_list in gt_boxes_per_image.items():
        img_preds = preds_by_img.get(img_id, [])
        img_preds = sorted(img_preds, key=lambda x: x['score'], reverse=True)
        
        gt_matched = set()
        for pred in img_preds:
            p_bbox = pred['bbox']
            p_area = p_bbox[2] * p_bbox[3]
            p_scale = get_scale_category(p_area)
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_list):
                if idx in gt_matched:
                    continue
                iou = compute_iou(p_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            if best_iou >= iou_thr and best_gt_idx != -1:
                tp_count += 1
                gt_matched.add(best_gt_idx)
                matched_scale = gt_list[best_gt_idx]['scale']
                scale_tp[matched_scale] += 1
            else:
                fp_count += 1
                scale_fp[p_scale] += 1
                
    total_gt = gt_info['num_annotations']
    fn_count = total_gt - tp_count
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    scale_metrics = {}
    for scale in ["small", "medium", "large"]:
        s_gt = scale_gt[scale]
        s_tp = scale_tp[scale]
        s_fp = scale_fp[scale]
        s_rec = s_tp / s_gt if s_gt > 0 else 0.0
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        scale_metrics[scale] = {
            "gt_count": s_gt,
            "tp_count": s_tp,
            "fp_count": s_fp,
            "missed_count": s_gt - s_tp,
            "recall": round(s_rec, 4),
            "precision": round(s_prec, 4)
        }
        
    return {
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "total_gt": total_gt,
        "total_predictions": total_preds,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "scale_breakdown": scale_metrics
    }

def run_error_analysis():
    print("="*70)
    print("ARGUS RGBT PERSON DETECTION - ERROR & SCALE ANALYSIS")
    print("="*70)
    
    # 1. GT Dataset breakdown
    test_gt_info = analyze_split_gt(split='test')
    val_gt_info = analyze_split_gt(split='val')
    
    print("\n--- GROUND TRUTH OBJECT SCALE DISTRIBUTION ---")
    if test_gt_info:
        print(f"Test Split GT Objects: {test_gt_info['num_annotations']} images across {test_gt_info['num_images']} images")
        for scale, cnt in test_gt_info['scale_distribution'].items():
            pct = (cnt / test_gt_info['num_annotations']) * 100
            print(f"  - {scale.capitalize():<8}: {cnt:>5} ({pct:.1f}%)")

    # 2. Stage 2 vs Stage 3 Benchmark Metric Breakdown
    stage2_file = "stage2_benchmark_results.json"
    stage3_file = "stage3_cmagm_results.json"
    
    b2_data = {}
    b3_data = {}
    
    if os.path.exists(stage2_file):
        with open(stage2_file) as f:
            b2_data = json.load(f)
    if os.path.exists(stage3_file):
        with open(stage3_file) as f:
            b3_data = json.load(f)

    print("\n" + "="*70)
    print("MODALITY & SCALE AP COMPARISON (STAGE 2 BASELINE vs STAGE 3 CMAGM)")
    print("="*70)
    print(f"{'Experiment':<25} | {'mAP':<6} | {'mAP50':<6} | {'mAP_s':<6} | {'mAP_m':<6} | {'mAP_l':<6}")
    print("-" * 70)
    
    all_experiments = [
        ("Baseline (Both Val)", b2_data.get("both_val", {})),
        ("Baseline (RGB Val)", b2_data.get("rgb_val", {})),
        ("Baseline (Thermal Val)", b2_data.get("thermal_val", {})),
        ("Baseline (Both Test)", b2_data.get("both_test", {})),
        ("Baseline (RGB Test)", b2_data.get("rgb_test", {})),
        ("Baseline (Thermal Test)", b2_data.get("thermal_test", {})),
        ("CMAGM (Both Val)", b3_data.get("both_val", {})),
        ("CMAGM (RGB Val)", b3_data.get("rgb_val", {})),
        ("CMAGM (Thermal Val)", b3_data.get("thermal_val", {})),
        ("CMAGM (Both Test)", b3_data.get("both_test", {})),
        ("CMAGM (RGB Test)", b3_data.get("rgb_test", {})),
        ("CMAGM (Thermal Test)", b3_data.get("thermal_test", {})),
    ]
    
    for label, exp in all_experiments:
        m = exp.get("metrics", {})
        map_val = m.get("bbox_mAP", 0.0)
        map50 = m.get("bbox_mAP_50", 0.0)
        map_s = m.get("bbox_mAP_s", 0.0)
        map_m = m.get("bbox_mAP_m", 0.0)
        map_l = m.get("bbox_mAP_l", 0.0)
        print(f"{label:<25} | {map_val:<6.3f} | {map50:<6.3f} | {map_s:<6.3f} | {map_m:<6.3f} | {map_l:<6.3f}")

    # 3. Detection prediction error metrics
    pred_path = "predictions_coco_format.json"
    pred_analysis = None
    if test_gt_info and os.path.exists(pred_path):
        print("\n" + "="*70)
        print("PREDICTION DISSECTION & SCALE RECALL ANALYSIS (Score Thr >= 0.30)")
        print("="*70)
        pred_analysis = analyze_predictions(pred_path, test_gt_info, score_thr=0.30, iou_thr=0.50)
        
        print(f"Total Detections   : {pred_analysis['total_predictions']}")
        print(f"True Positives (TP): {pred_analysis['tp']}")
        print(f"False Positives(FP): {pred_analysis['fp']}")
        print(f"False Negatives(FN): {pred_analysis['fn']}")
        print(f"Precision          : {pred_analysis['precision']:.4f}")
        print(f"Recall             : {pred_analysis['recall']:.4f}")
        print(f"F1 Score           : {pred_analysis['f1_score']:.4f}")
        
        print("\n--- Breakdown by Object Scale ---")
        for scale, s_data in pred_analysis['scale_breakdown'].items():
            print(f"Scale: {scale.upper():<6} | GT: {s_data['gt_count']:<4} | TP: {s_data['tp_count']:<4} | FP: {s_data['fp_count']:<4} | Missed: {s_data['missed_count']:<4} | Recall: {s_data['recall']:.2%} | Prec: {s_data['precision']:.2%}")

    # Save to report
    report_output = {
        "gt_summary": {
            "test": test_gt_info['scale_distribution'] if test_gt_info else {},
            "val": val_gt_info['scale_distribution'] if val_gt_info else {}
        },
        "modality_scale_comparison": {label: exp.get("metrics", {}) for label, exp in all_experiments},
        "prediction_error_analysis": pred_analysis
    }
    
    report_file = "error_analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report_output, f, indent=2)
    print(f"\nFull Error & Scale Breakdown saved to: {report_file}")

if __name__ == '__main__':
    run_error_analysis()
