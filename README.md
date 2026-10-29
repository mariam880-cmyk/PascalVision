# PascalVision
Object detection using YOLO and the Pascal VOC dataset
# PascalVision — Object Detection with YOLO

## Project Overview

PascalVision is a computer vision project that detects and localizes objects in images using a pretrained YOLO model fine-tuned on the Pascal VOC dataset.

The system predicts object classes and draws bounding boxes around detected objects.

## Dataset

The project uses the Pascal VOC dataset with 20 object classes, including people, cars, animals, chairs, and buses.

The dataset is not included in this repository. Configure the dataset paths in the notebook before running training or evaluation.

## Model and Experiments

Two training runs were compared:

- Baseline: 5 epochs
- Experiment: 10 epochs

The models were evaluated using precision, recall, mAP@0.5, and mAP@0.5:0.95. See the notebook for the evaluation code and results.

## Repository Contents

- `PascalVision.ipynb` — data configuration, training, evaluation, and prediction demo
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

## Setup

1. Open the notebook in Kaggle or another compatible Python environment.
2. Enable a GPU if available.
3. Install the dependencies from `requirements.txt`.
4. Download and prepare the Pascal VOC dataset.
5. Update the dataset configuration paths in the notebook.
6. Run the notebook cells in order.

The notebook currently uses Kaggle-specific paths, so those paths must be adapted if running elsewhere.

## Running Predictions

Load a trained model using Ultralytics YOLO, then call `model.predict()` on an image. The notebook contains a working prediction example that displays detected objects with bounding boxes.

## Limitations

Detection quality varies by object class. Some classes, including potted plants and chairs, were more challenging in the evaluation. Results may also vary with image size, confidence threshold, and training settings.

## Future Work

- Train for more epochs and tune hyperparameters
- Analyze false positives and missed detections
- Test on images outside the training dataset
- Compare inference speed and model size

