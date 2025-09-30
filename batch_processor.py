#!/usr/bin/env python3
"""
Batch processor for multiple images
Processes all images in a directory and saves results
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import time
from professional_analyzer import ProfessionalImageAnalyzer
from vision_api_manager import VisionAPIManager

class BatchProcessor:
    """Process multiple images in parallel"""

    def __init__(self, max_workers: int = 4, use_vision: bool = False):
        self.max_workers = max_workers
        self.use_vision = use_vision
        self.analyzer = ProfessionalImageAnalyzer()
        self.vision_manager = VisionAPIManager() if use_vision else None

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image"""
        try:
            print(f"📸 Processing: {image_path}")

            # Basic analysis
            result = self.analyzer.analyze_image(image_path)

            # Add vision analysis if enabled
            if self.use_vision and self.vision_manager:
                vision_result = self.vision_manager.analyze_image(image_path)

                if vision_result.success:
                    result['services']['vision-enhanced'] = {
                        'status': 'success',
                        'result': {
                            'provider': vision_result.provider,
                            'scene_type': vision_result.scene_type,
                            'people_count': vision_result.people_count,
                            'face_count': vision_result.face_count,
                            'objects': vision_result.objects,
                            'text_detected': vision_result.text_detected,
                            'description': vision_result.description
                        },
                        'metrics': {
                            'processing_time_ms': 100,
                            'service_version': '1.0.0'
                        },
                        'error': None
                    }

            result['source_file'] = os.path.basename(image_path)
            return result

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return {
                'source_file': os.path.basename(image_path),
                'error': str(e),
                'status': 'failed'
            }

    def process_directory(self, directory: str, pattern: str = "*") -> Dict[str, Any]:
        """Process all images in directory"""

        # Find all image files
        extensions = ['jpg', 'jpeg', 'png', 'heic', 'heif', 'gif']
        image_files = []

        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(directory, f"{pattern}.{ext}"), recursive=False))
            image_files.extend(glob.glob(os.path.join(directory, f"{pattern}.{ext.upper()}"), recursive=False))

        if not image_files:
            print(f"⚠️  No images found in {directory} matching pattern '{pattern}'")
            return {'error': 'No images found', 'directory': directory, 'pattern': pattern}

        print(f"\n🔍 Found {len(image_files)} images to process")
        print(f"⚙️  Using {self.max_workers} workers")

        start_time = time.time()
        results = []

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_image, img): img
                            for img in image_files}

            # Collect results as they complete
            for future in as_completed(future_to_file):
                image_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Completed: {os.path.basename(image_file)}")
                except Exception as e:
                    print(f"❌ Failed: {os.path.basename(image_file)} - {e}")
                    results.append({
                        'source_file': os.path.basename(image_file),
                        'error': str(e),
                        'status': 'failed'
                    })

        processing_time = time.time() - start_time

        # Create summary
        successful = sum(1 for r in results if r.get('status') != 'failed')
        failed = len(results) - successful

        summary = {
            'batch_processing': {
                'total_files': len(image_files),
                'successful': successful,
                'failed': failed,
                'processing_time_seconds': round(processing_time, 2),
                'average_time_per_image': round(processing_time / len(image_files), 2),
                'workers_used': self.max_workers
            },
            'results': results
        }

        print(f"\n📊 Batch Processing Complete!")
        print(f"   Total: {len(image_files)} images")
        print(f"   Success: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Time: {processing_time:.2f}s")
        print(f"   Average: {processing_time/len(image_files):.2f}s per image")

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Batch process images for analysis')
    parser.add_argument('directory', help='Directory containing images')
    parser.add_argument('-o', '--output', help='Output JSON file', default='batch_results.json')
    parser.add_argument('-p', '--pattern', help='File pattern (e.g., IMG_*)', default='*')
    parser.add_argument('-w', '--workers', type=int, help='Number of parallel workers', default=4)
    parser.add_argument('-v', '--vision', action='store_true', help='Enable vision API analysis')

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"❌ Error: Directory '{args.directory}' does not exist")
        sys.exit(1)

    # Create processor
    processor = BatchProcessor(max_workers=args.workers, use_vision=args.vision)

    # Process directory
    print(f"\n🚀 Starting batch processing...")
    print(f"📁 Directory: {args.directory}")
    print(f"🔍 Pattern: {args.pattern}")

    results = processor.process_directory(args.directory, args.pattern)

    # Save results
    processor.save_results(results, args.output)

    print("\n✨ Done!")

if __name__ == "__main__":
    main()