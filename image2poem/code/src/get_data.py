from datasets import load_dataset
import os
import csv
from PIL import Image
import io

# 1. download dataset
print("Downloading and loading dataset...")
ds = load_dataset("SHENJJ1017/Image-Text")

# 2. specify output directory
OUTPUT_DIR = "hf_exported_data"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

print(f"Exporting dataset to images and CSV in: {OUTPUT_DIR}")

# 3. iterate over all dataset splits (train, test, validation, etc.) and export
for split_name, dataset in ds.items():
    csv_file = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    split_images_dir = os.path.join(IMAGES_DIR, split_name)
    os.makedirs(split_images_dir, exist_ok=True)
    
    print(f"\nProcessing split '{split_name}'...")
    print(f"  Images will be saved to: {split_images_dir}")
    print(f"  CSV will be saved to: {csv_file}")
    
    # Open CSV file for writing
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename', 'text'])  # Header
        
        # Process each example in the dataset
        for idx, example in enumerate(dataset):
            try:
                # Get image and text
                image_data = example.get('image')
                text = example.get('text', '')
                file_name = example.get('file_name', f'image_{idx:06d}.jpg')
                
                # Ensure filename has proper extension
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_name = f"{file_name}.jpg"
                
                # Save image
                image_path = os.path.join(split_images_dir, file_name)
                
                if image_data is not None:
                    # Handle different image formats
                    if isinstance(image_data, Image.Image):
                        # Already a PIL Image
                        image_data.save(image_path)
                    elif isinstance(image_data, dict) and 'bytes' in image_data:
                        # Image bytes from HF dataset
                        img_bytes = image_data['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                        img.save(image_path)
                    else:
                        print(f"  Warning: Skipping example {idx} - unsupported image format")
                        continue
                    
                    # Write to CSV
                    writer.writerow([file_name, text])
                    
                    if (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{len(dataset)} examples...")
                else:
                    print(f"  Warning: Skipping example {idx} - no image data")
            
            except Exception as e:
                print(f"  Error processing example {idx}: {e}")
    
    print(f"  Completed split '{split_name}': {len(dataset)} examples processed")

print("\nExport complete!")
print(f"\nOutput structure:")
print(f"  {OUTPUT_DIR}/")
print(f"    ├── images/")
print(f"    │   ├── train/  (image files)")
print(f"    │   └── test/   (image files, if exists)")
print(f"    ├── train.csv")
print(f"    └── test.csv (if exists)")