import os
import csv
import shutil
import json

base_dir = r"c:\Users\13412\Desktop\2026023169-作品主文件夹"
csv_path = os.path.join(base_dir, r"2026023169-02素材与源码\data\data\merged_data.csv")
no_dir = os.path.join(base_dir, "no")
yes_dir = os.path.join(base_dir, "yes")
demo_dir = os.path.join(base_dir, r"2026023169-02素材与源码\demo")
output_json = os.path.join(demo_dir, "data.json")

# Create assets directories in demo
demo_no_dir = os.path.join(demo_dir, "assets", "no")
demo_yes_dir = os.path.join(demo_dir, "assets", "yes")
os.makedirs(demo_no_dir, exist_ok=True)
os.makedirs(demo_yes_dir, exist_ok=True)

# 1. Read and copy images
images = []
if os.path.exists(no_dir):
    for f in os.listdir(no_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(no_dir, f)
            dst = os.path.join(demo_no_dir, f)
            shutil.copy2(src, dst)
            images.append({"path": f"./assets/no/{f}", "isIce": False})

if os.path.exists(yes_dir):
    for f in os.listdir(yes_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(yes_dir, f)
            dst = os.path.join(demo_yes_dir, f)
            shutil.copy2(src, dst)
            images.append({"path": f"./assets/yes/{f}", "isIce": True})

# 2. Read prediction data from CSV
# We will just take the last 48 records of terminal CC0474 as "future prediction" 
# or just take an interesting slice. 
# Let's read the file and get a slice of data where thickness changes.
predictions = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    records = []
    for row in reader:
        if row['终端编号'] == 'CC0474':
            records.append({
                'time': row['时间'],
                'thickness': float(row['覆冰厚度']),
                'ratio': float(row['覆冰比值'])
            })
    
    # We want a slice that shows a curve. Let's take index 300 to 348 (48 points)
    if len(records) > 350:
        slice_records = records[300:348]
    else:
        slice_records = records[-48:] if len(records) >= 48 else records
        
    for idx, r in enumerate(slice_records):
        # We can pretend the times are "future" +idx hours from now
        predictions.append({
            "hour": f"+{idx}h",
            "thickness": r['thickness']
        })

# Output JSON
output_data = {
    "images": images,
    "predictions": predictions
}

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("Data preparation complete. Output saved to:", output_json)
