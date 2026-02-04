"""
Microplastic Detection System
Uses Roboflow's pre-trained model to detect microplastics in images
"""

import os
import json
import base64
import requests
from pathlib import Path


class MicroplasticDetector:
    """Detector for microplastics using Roboflow API"""
    
    def __init__(self, api_key=None):
        """
        Initialize the microplastic detector
        
        Args:
            api_key (str): Roboflow API key. If not provided, will look for ROBOFLOW_API_KEY env var
        """
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY', 'ST2dC0JPcQQ2wjaZm8Cm')
        self.model_id = "microplastic_100/4"
        self.api_url = "https://detect.roboflow.com"
    
    def detect(self, image_path):
        """
        Detect microplastics in an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Detection results containing predictions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Analyzing image: {image_path}")
        result = self.client.infer(image_path, model_id=self.model_id)
        
        return result
    
    def 
        # Initialize model on first use
        if self.model is None:
            self.model = get_model(model_id=self.model_id, api_key=self.api_key)
        
        result = self.model.infer(image_path)[0]
        
        # Convert to dict format
        result_dict = result.dict() if hasattr(result, 'dict') else result
        
        return result_dicplastics and save results to JSON
        
        Args:
            image_path (str): Path to the image file
            output_path (str): Path to save results. If None, uses default naming
            
        Returns:
            dict: Detection results
        """
        result = self.detect(image_path)
        
        if output_path is None:
            output_path = Path(image_path).stem + '_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return result
    
    def print_summary(self, result):
        """
        Print a summary of detection results
        
        Args:
            result (dict): Detection results from API
        """
        predictions = result.get('predictions', [])
        print(f"\n{'='*50}")
        print(f"Detection Summary")
        print(f"{'='*50}")
        print(f"Total microplastics detected: {len(predictions)}")
        
        if predictions:
            print(f"\nDetailed Results:")
            for i, pred in enumerate(predictions, 1):
                confidence = pred.get('confidence', 0) * 100
                class_name = pred.get('class', 'unknown')
                print(f"  {i}. Class: {class_name} | Confidence: {confidence:.1f}%")
        else:
            print("No microplastics detected in the image.")
        print(f"{'='*50}\n")


def main():
    """Main function to run detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect microplastics in images')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--output', '-o', help='Output path for results JSON')
    parser.add_argument('--api-key', help='Roboflow API key (optional)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MicroplasticDetector(api_key=args.api_key)
    
    # Run detection
    result = detector.detect_and_save(args.image, args.output)
    
    # Print summary
    detector.print_summary(result)


if __name__ == "__main__":
    main()
