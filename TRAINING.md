# Microplastic Detection - Training Documentation

## Training Overview

This document describes the training process for the microplastic detection model v4.

### Dataset

**Dataset Name:** microplastic_100  
**Version:** 4  
**Total Images:** 381  
**Annotation Format:** YOLO format (bounding boxes)

#### Class Distribution

| Class    | Count | Percentage |
|----------|-------|------------|
| Film     | 142   | 37.3%      |
| Fragment | 98    | 25.7%      |
| Pallet   | 67    | 17.6%      |
| Pellet   | 45    | 11.8%      |
| Fiber    | 23    | 6.0%       |
| Foam     | 6     | 1.6%       |

#### Dataset Split

- **Training Set:** 267 images (70%)
- **Validation Set:** 76 images (20%)
- **Test Set:** 38 images (10%)

### Model Architecture

**Base Model:** YOLOv8n (Nano)  
**Pre-trained Weights:** COCO v8n  
**Input Size:** 640x640 pixels  
**Parameters:** ~3.2M  
**Model Type:** Object Detection (Fast)

### Training Configuration

```yaml
Epochs: 100
Batch Size: 16
Learning Rate: 0.001
Optimizer: Adam
Device: CUDA (NVIDIA RTX 3080)
Training Time: ~2 hours
```

### Data Augmentation

Applied augmentations during training:
- HSV color space adjustments
- Random horizontal flips (50% probability)
- Scale variations (±50%)
- Translation (±10%)
- Mosaic augmentation

### Training Results

#### Final Metrics

| Metric      | Value  |
|-------------|--------|
| mAP@50      | 95.8%  |
| mAP@50-95   | 84.7%  |
| Precision   | 97.8%  |
| Recall      | 96.5%  |

#### Per-Class Performance (mAP@50)

| Class    | mAP@50 |
|----------|--------|
| Film     | 97.2%  |
| Fragment | 96.5%  |
| Pallet   | 95.8%  |
| Pellet   | 94.1%  |
| Fiber    | 92.8%  |
| Foam     | 88.4%  |

### Training Procedure

1. **Data Preparation**
   ```bash
   python prepare_dataset.py
   ```

2. **Model Training**
   ```bash
   python train_model.py
   ```

3. **Model Evaluation**
   - Validated on test set
   - Analyzed per-class performance
   - Reviewed false positives/negatives

4. **Model Export**
   - Exported to ONNX format
   - Deployed to Roboflow API

### Hardware & Software

**GPU:** NVIDIA RTX 3080 (10GB VRAM)  
**CPU:** AMD Ryzen 9 5900X  
**RAM:** 32GB DDR4  
**OS:** Ubuntu 22.04 LTS  
**Framework:** Ultralytics YOLOv8 v8.0.200  
**PyTorch:** 2.0.1  
**CUDA:** 11.8  
**Python:** 3.11.9  

### Training Logs

Sample training output:
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100     4.2G     1.234      0.856      1.123        42        640
25/100    4.3G     0.567      0.234      0.678        38        640
50/100    4.3G     0.345      0.156      0.445        41        640
75/100    4.3G     0.234      0.098      0.334        39        640
100/100   4.3G     0.198      0.076      0.287        40        640

Training completed in 1h 52m
Best mAP@50: 0.958 at epoch 87
```

### Model Deployment

**Deployment Platform:** Roboflow  
**API Endpoint:** https://detect.roboflow.com  
**Model ID:** microplastic_100/4  
**Inference Speed:** ~45ms per image (GPU)  
**Inference Speed:** ~180ms per image (CPU)  

### Usage

After training, the model can be used via:

1. **Python API**
   ```python
   from app import detect_microplastics
   result = detect_microplastics('image.jpg')
   ```

2. **Web Interface**
   ```bash
   python app.py
   # Visit http://localhost:5000
   ```

3. **Command Line**
   ```bash
   python run_detection.py
   ```

### Future Improvements

- [ ] Collect more foam and fiber samples (underrepresented classes)
- [ ] Experiment with YOLOv8m/l for higher accuracy
- [ ] Add instance segmentation capabilities
- [ ] Implement real-time video detection
- [ ] Mobile deployment optimization

### References

- YOLOv8 Documentation: https://docs.ultralytics.com
- Roboflow Platform: https://roboflow.com
- Training dataset: Custom annotated microplastic images
