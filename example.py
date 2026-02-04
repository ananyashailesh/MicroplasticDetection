"""
Example usage of the microplastic detection system
"""

from detect_microplastics import MicroplasticDetector

def example_basic_detection():
    """Example: Basic detection"""
    detector = MicroplasticDetector()
    
    # Replace with your actual image path
    image_path = "your_image.jpg"
    
    try:
        result = detector.detect(image_path)
        detector.print_summary(result)
    except FileNotFoundError:
        print(f"Please provide a valid image path")

def example_batch_processing():
    """Example: Process multiple images"""
    import glob
    
    detector = MicroplasticDetector()
    
    # Process all JPG files in a directory
    for image_path in glob.glob("images/*.jpg"):
        print(f"\nProcessing: {image_path}")
        result = detector.detect_and_save(image_path)
        detector.print_summary(result)

def example_custom_api_key():
    """Example: Use custom API key"""
    detector = MicroplasticDetector(api_key="YOUR_CUSTOM_API_KEY")
    result = detector.detect("your_image.jpg")
    print(result)

if __name__ == "__main__":
    # Run the basic example
    example_basic_detection()
