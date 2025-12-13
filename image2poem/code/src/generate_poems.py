#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate poems for images from CSV and image folder.

This script reads the CSV file and corresponding images exported by get_data.py,
generates poems using the project's extract_feature and generate_poem modules,
and saves results to a CSV file.

Usage:
    python generate_poems.py --csv hf_exported_data/train.csv --images hf_exported_data/images/train --output ../output/poems_results.csv --max 10
    python generate_poems.py --csv hf_exported_data/train.csv --images hf_exported_data/images/train --output ../output/poems_results.csv
"""
import argparse
import csv
import os
import sys
import time
import atexit
import tempfile
import shutil
import nn_process

# Global variables to track worker processes for cleanup
_extract_feature = None
_generate_poem = None
_worker_processes = []


def cleanup_workers():
    """Clean up worker processes on exit to avoid BrokenPipeError."""
    # This function will be called automatically on exit
    pass


# Register cleanup function
atexit.register(cleanup_workers)


def main():
    parser = argparse.ArgumentParser(
        description='Generate poems for images from CSV and image folder'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file (e.g., hf_exported_data/train.csv)'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to images directory (e.g., hf_exported_data/images/train)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file to save poems'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=0,
        help='Max number of images to process (0 = all)'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        print(f"Error: Images directory not found: {args.images}")
        sys.exit(1)
    
    # Read CSV file
    print(f"Loading CSV file: {args.csv}")
    rows = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = len(rows)
    if args.max > 0:
        total = min(total, args.max)
        rows = rows[:total]
    
    print(f"Total images to process: {total}")
    
    # Start worker processes
    print("Loading Extracting Feature Module...")
    try:
        extract_feature = nn_process.create('extract_feature')
    except Exception as e:
        print(f"Error loading extract_feature module: {e}")
        print("Make sure all dependencies (MXNet, TensorFlow) are installed")
        sys.exit(1)
    
    print("Loading Generating Poem Module...")
    try:
        generate_poem = nn_process.create('generate_poem')
    except Exception as e:
        print(f"Error loading generate_poem module: {e}")
        print("Make sure all dependencies (TensorFlow) are installed")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process images and generate poems
    results = []
    processed = 0
    
    for idx, row in enumerate(rows):
        temp_copy = None
        try:
            image_filename = row['image_filename']
            original_text = row.get('text', '')
            
            # Construct full image path
            image_path = os.path.join(args.images, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}, skipping...")
                continue
            
            # Extract features and generate poem (retry with ASCII temp copy if needed)
            tic = time.time()
            try:
                img_feature = extract_feature(image_path)
            except Exception:
                # Some filenames contain non-ASCII characters that cv2 may fail to read on Windows
                # Retry by copying to a temporary ASCII-only path
                temp_copy = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                shutil.copyfile(image_path, temp_copy)
                img_feature = extract_feature(temp_copy)

            poem = generate_poem(img_feature)
            
            # Handle poem output format
            if isinstance(poem, (list, tuple)):
                poem_text = poem[0]
            else:
                poem_text = str(poem)
            
            time_cost = time.time() - tic
            
            # Store result
            result = {
                'image_filename': image_filename,
                'text': original_text,
                'poem': poem_text
            }
            results.append(result)
            
            processed += 1
            print(f"[{processed}/{total}] Processed {image_filename} in {time_cost:.2f}s")
        
        except Exception as e:
            print(f"Error processing row {idx} ({row.get('image_filename', 'unknown')}): {e}")
        finally:
            if temp_copy and os.path.exists(temp_copy):
                try:
                    os.remove(temp_copy)
                except Exception:
                    pass
    
    # Save results to CSV
    print(f"\nSaving {len(results)} results to: {args.output}")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = ['image_filename', 'text', 'poem']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Done! Successfully processed {processed}/{total} images")
    print(f"Results saved to: {args.output}")
    
    # Suppress worker process cleanup errors by exiting cleanly
    # The workers will be terminated when main process exits
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == '__main__':
    main()