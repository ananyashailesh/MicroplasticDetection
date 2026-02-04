# Microplastic Detection System

A Python application for detecting microplastics in images using Roboflow's pre-trained computer vision model.

## Features

- 🔍 Detect microplastics in images using state-of-the-art AI
- 📊 Get detailed confidence scores for each detection
- 💾 Save results to JSON format
- 🎯 Simple command-line interface

## Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up your API key as an environment variable:
```bash
cp .env.example .env
# Edit .env and add your Roboflow API key
```

## Usage

### Basic Usage

```bash
python detect_microplastics.py your_image.jpg
```

### Save Results to Custom Location

```bash
python detect_microplastics.py your_image.jpg --output results.json
```

### Use Custom API Key

```bash
python detect_microplastics.py your_image.jpg --api-key YOUR_API_KEY
```

### Python API

You can also use the detector in your own Python code:

```python
from detect_microplastics import MicroplasticDetector

# Initialize detector
detector = MicroplasticDetector()

# Detect microplastics
result = detector.detect('your_image.jpg')

# Print summary
detector.print_summary(result)

# Save results
detector.detect_and_save('your_image.jpg', 'output.json')
```

## Model Information

- **Model**: microplastic_100/4
- **Type**: Roboflow 2.0 Object Detection (Fast)
- **Checkpoint**: COCOv8n
- **Performance**: 
  - mAP@50: 95.8%
  - Precision: 97.8%
  - Recall: 96.5%

## Output Format

The detector returns results in JSON format:

```json
{
  "predictions": [
    {
      "x": 1409.5,
      "y": 2246,
      "width": 163,
      "height": 478,
      "confidence": 0.832,
      "class": "film",
      "class_id": 0,
      "detection_id": "G8bH..."
    }
  ]
}
```

## License

This project uses the Roboflow inference SDK. Please refer to Roboflow's terms of service for API usage.

## Contributing

Feel free to submit issues or pull requests for improvements!
