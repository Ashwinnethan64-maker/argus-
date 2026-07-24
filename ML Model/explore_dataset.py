import os
import json
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = os.path.join("VTUAV_subset", "VTUAV_subset")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "annotations")
RGB_DIR = os.path.join(DATASET_ROOT, "VTUAV_co")
IR_DIR = os.path.join(DATASET_ROOT, "VTUAV_ir")
OUTPUT_VIS_DIR = os.path.join("stage1_visualizations")

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

splits = ["train", "val", "test"]

print("="*60)
print("STAGE 1: DATASET EXPLORATION & STATISTICAL ANALYSIS")
print("="*60)

dataset_stats = {}

for split in splits:
    json_path = os.path.join(ANNOTATIONS_DIR, f"{split}.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        continue
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    
    num_images = len(images)
    num_annotations = len(annotations)
    avg_instances = num_annotations / num_images if num_images > 0 else 0
    
    # Scale counts
    small_count = 0  # area < 32^2 = 1024
    medium_count = 0 # 1024 <= area < 9216
    large_count = 0  # area >= 96^2 = 9216
    
    # Track resolutions & channels
    resolutions = set()
    
    for ann in annotations:
        area = ann.get("area", 0)
        # If area is not precalculated or 0, compute from bbox [x, y, w, h]
        if area <= 0 and "bbox" in ann:
            w, h = ann["bbox"][2], ann["bbox"][3]
            area = w * h
            
        if area < 32 * 32:
            small_count += 1
        elif area < 96 * 96:
            medium_count += 1
        else:
            large_count += 1

    for img in images:
        resolutions.add((img.get("width"), img.get("height")))

    dataset_stats[split] = {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "avg_instances_per_image": avg_instances,
        "small": small_count,
        "medium": medium_count,
        "large": large_count,
        "resolutions": list(resolutions),
        "categories": categories
    }
    
    print(f"\n--- Split: {split.upper()} ---")
    print(f"Total Images: {num_images}")
    print(f"Total Pedestrian Instances: {num_annotations}")
    print(f"Average Instances per Image: {avg_instances:.2f}")
    print(f"Scale Distribution:")
    print(f"  Small  (area < 32^2 = 1024 px^2) : {small_count:6d} ({small_count/num_annotations*100:.2f}%)")
    print(f"  Medium (32^2 <= area < 96^2)    : {medium_count:6d} ({medium_count/num_annotations*100:.2f}%)")
    print(f"  Large  (area >= 96^2 = 9216 px^2): {large_count:6d} ({large_count/num_annotations*100:.2f}%)")
    print(f"Resolutions: {list(resolutions)}")

# Examine sample images for channel / format details
sample_val_json = os.path.join(ANNOTATIONS_DIR, "val.json")
with open(sample_val_json, "r") as f:
    val_data = json.load(f)

val_images = val_data["images"]
val_anns_by_img = defaultdict(list)
for ann in val_data["annotations"]:
    val_anns_by_img[ann["image_id"]].append(ann)

# Inspect first image properties
first_img_info = val_images[0]
sample_rgb_path = os.path.join(RGB_DIR, "val", "images", first_img_info["file_name"])
sample_ir_path = os.path.join(IR_DIR, "val", "images", first_img_info["file_name"])

print("\n--- Image Channel & Format Analysis ---")
if os.path.exists(sample_rgb_path):
    with Image.open(sample_rgb_path) as im_rgb:
        print(f"RGB Image: mode={im_rgb.mode}, size={im_rgb.size}, format={im_rgb.format}")
if os.path.exists(sample_ir_path):
    with Image.open(sample_ir_path) as im_ir:
        print(f"IR Image : mode={im_ir.mode}, size={im_ir.size}, format={im_ir.format}")

# Visualization of 20 paired RGB-thermal images with bounding boxes
print("\n--- Generating 20+ Paired RGB-Thermal Visualizations ---")
num_to_vis = 25
vis_count = 0

for img_info in val_images:
    if vis_count >= num_to_vis:
        break
    
    file_name = img_info["file_name"]
    img_id = img_info["id"]
    anns = val_anns_by_img.get(img_id, [])
    
    rgb_file = os.path.join(RGB_DIR, "val", "images", file_name)
    ir_file = os.path.join(IR_DIR, "val", "images", file_name)
    
    if not os.path.exists(rgb_file) or not os.path.exists(ir_file):
        continue
    
    img_rgb = Image.open(rgb_file).convert("RGB")
    img_ir = Image.open(ir_file).convert("RGB")
    
    draw_rgb = ImageDraw.Draw(img_rgb)
    draw_ir = ImageDraw.Draw(img_ir)
    
    for ann in anns:
        bbox = ann["bbox"] # [x, y, w, h]
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        draw_rgb.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw_ir.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
    
    # Side-by-side canvas
    canvas_w = img_rgb.width + img_ir.width
    canvas_h = img_rgb.height
    canvas = Image.new("RGB", (canvas_w, canvas_h))
    canvas.paste(img_rgb, (0, 0))
    canvas.paste(img_ir, (img_rgb.width, 0))
    
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((20, 20), f"RGB (GT Boxes in Red) - {file_name}", fill="yellow")
    draw_canvas.text((img_rgb.width + 20, 20), f"Thermal IR (GT Boxes in Cyan) - {file_name}", fill="yellow")
    
    out_path = os.path.join(OUTPUT_VIS_DIR, f"paired_vis_{vis_count+1:02d}_{file_name}")
    canvas.save(out_path)
    vis_count += 1

print(f"Successfully saved {vis_count} paired visualizations to: {OUTPUT_VIS_DIR}")

# Save json summary of dataset stats
with open("stage1_dataset_summary.json", "w") as f:
    json.dump(dataset_stats, f, indent=2)
print("Saved summary stats to stage1_dataset_summary.json")
