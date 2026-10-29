# PascalVision: Object Detection with YOLO on Pascal VOC

## 1. Executive summary

PascalVision is a computer-vision project for object detection and
localization. It uses Ultralytics YOLO initialized from pretrained
weights and fine-tuned on Pascal VOC. For each image, the model predicts
object classes and bounding boxes, and the notebook visualizes the
detections.

The workflow includes dataset configuration and validation, transfer
learning, evaluation, comparison of two training runs, qualitative error
inspection, and inference on sample images.

**Environment recorded:** Kaggle Notebook, Tesla T4 GPU, Python 3.13.15,
PyTorch 2.11.0+cu128, Ultralytics 8.4.192.\
**Model initialization:** `yolo26n.pt`.\
**Task:** Object detection; segmentation is not part of this
implementation.

## 2. Problem statement

Image classification assigns labels to a whole image. Object detection
instead identifies individual objects and estimates their locations
using bounding boxes. The objective of PascalVision is to build a
working detector, evaluate its performance with standard metrics, and
demonstrate its predictions on images.

## 3. Dataset

The project uses Pascal VOC with 20 classes: aeroplane, bicycle, bird,
boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse,
motorbike, person, pottedplant, sheep, sofa, train, and tvmonitor.

The notebook uses images and labels from `train2007` and `train2012` for
training and images from `test2007` for evaluation. In the current
Kaggle environment, the dataset root is `/kaggle/working/datasets/VOC`
and the YAML configuration is `/kaggle/working/VOC.yaml`. The YAML maps
class IDs 0--19 to the class names and identifies the training and
evaluation directories. The configuration was checked with Ultralytics'
dataset validation utility.

The dataset is not included in the GitHub repository. Anyone reproducing
the project must obtain the dataset from an authorized source, prepare
the expected YOLO-format folders and labels, and update paths for their
environment.

**Evaluation caveat:** `test2007` is used as the evaluation/validation
split in this notebook. Because it was used to compare runs, it should
not be described as a completely untouched final test set.

## 4. Data preparation

The model expects YOLO-format object-detection labels: class IDs and
normalized bounding-box coordinates associated with each image. The
notebook checks the expected folders and sample images, creates or
verifies `VOC.yaml`, validates the dataset configuration, and then uses
the same configuration for both runs. The project relies on existing
Pascal VOC annotations; no new annotation campaign or segmentation masks
were created.

## 5. Model and training

The notebook initializes Ultralytics YOLO from pretrained `yolo26n.pt`
weights and fine-tunes it on Pascal VOC. This is transfer learning:
pretrained visual features are adapted to the target dataset.

  --------------------------------------------------------------------------------
  Run          Initial               Epochs    Image size    Batch size Hardware
               weights                                                  
  ------------ -------------- ------------- ------------- ------------- ----------
  Baseline     `yolo26n.pt`               5           640            16 Kaggle
                                                                        Tesla T4

  Longer       `yolo26n.pt`              10           640            16 Kaggle
  training                                                              Tesla T4
  experiment                                                            
  --------------------------------------------------------------------------------

Saved checkpoints in the current Kaggle working directory: -
`pascalvision/runs/baseline_5_epochs_kaggle/weights/best.pt` -
`pascalvision/runs/experiment_10_epochs_kaggle/weights/best.pt`

These are Kaggle-local paths, not automatically accessible to someone
cloning the repository. To make the project reproducible, upload
suitable checkpoint files or provide clear, working download
instructions; alternatively, users can rerun training.

## 6. Evaluation metrics

-   **Precision:** the fraction of predicted detections that are correct
    under the evaluator's matching criteria; higher precision generally
    means fewer false positives.
-   **Recall:** the fraction of ground-truth objects detected; higher
    recall generally means fewer missed objects.
-   **mAP50:** mean average precision at intersection-over-union (IoU)
    threshold 0.50.
-   **mAP50-95:** mean average precision averaged over IoU thresholds
    0.50 to 0.95 in 0.05 increments. It is stricter about localization
    quality than mAP50.

The 10-epoch evaluation output reported 4,952 images and 12,032 labeled
instances.

## 7. Results and comparison

These values were recorded from the executed notebook:

  Metric        5 epochs   10 epochs   Change (10 − 5, percentage points)
  ----------- ---------- ----------- ------------------------------------
  Precision       0.8306      0.8251                                -0.55
  Recall          0.7784      0.7898                                +1.14
  mAP50           0.8620      0.8605                                -0.15
  mAP50-95        0.6649      0.6672                                +0.23

The 10-epoch run increased recall by 1.14 percentage points and mAP50-95
by 0.23 points, while precision fell by 0.55 points and mAP50 fell by
0.15 points. Overall, the two runs are very similar. The longer run
shows a small trade-off rather than a clear improvement across all
metrics. These results do not prove that simply adding epochs is an
effective optimization.

## 8. Per-class evaluation and error analysis

Recorded per-class mAP50 values for the 10-epoch model:

  Class           mAP50
  ------------- -------
  aeroplane       0.721
  bicycle         0.716
  bird            0.619
  boat            0.524
  bottle          0.561
  bus             0.820
  car             0.760
  cat             0.780
  chair           0.499
  cow             0.691
  diningtable     0.643
  dog             0.723
  horse           0.758
  motorbike       0.679
  person          0.671
  pottedplant     0.376
  sheep           0.681
  sofa            0.711
  train           0.731
  tvmonitor       0.680

Bus had the highest listed class score (0.820). Pottedplant was the
lowest (0.376), followed by chair (0.499), boat (0.524), and bottle
(0.561). These classes are promising targets for further error analysis.
Possible factors include object size, occlusion, visual ambiguity, and
variation in examples, but these causes have not been proven by the
metrics alone.

The notebook also visualizes predictions on sample images and compares
outputs. Some objects were missed in inspected examples, illustrating
that good aggregate metrics do not guarantee every object is detected. A
confidence-threshold demonstration tested 0.15, 0.25, and 0.50 on one
image; the same two detections remained because both scores exceeded
0.50. This is a single-image demonstration, not a dataset-wide threshold
evaluation.

## 9. Practical inference

The notebook loads a trained checkpoint and predicts boxes and class
labels on sample images. The basic Ultralytics inference pattern is:

``` python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=False)
for result in results:
    plotted_image = result.plot()
```

Replace the checkpoint and image paths with real paths in the
environment being used. The notebook's `/kaggle/working/` paths will not
work on another computer without modification. A confidence threshold
controls which detections are returned/displayed: a higher threshold
generally removes more low-confidence detections, while a lower
threshold can show more detections and may increase false positives.

## 10. Optimization and before/after evidence

The completed comparison is a training-duration experiment (5 vs 10
epochs), not a complete deployment optimization study. It provides
before/after metrics but no measured inference latency, model-size
reduction, or memory reduction. If the assignment specifically requires
optimizing speed, size, or resource use, that part still needs a
measured experiment.

Recommended follow-up: measure baseline inference time and checkpoint
size; apply one change (for example ONNX export, FP16 inference on
supported hardware, or a different image size); then measure again using
the same images, hardware, warm-up procedure, and settings. Record both
quality metrics and latency/size. Do not claim an optimization succeeded
until the measurements exist. Confidence thresholds should be selected
using a validation set, not one image.

## 11. Strengths

-   Uses pretrained weights and transfer learning.
-   Implements training, evaluation, and inference in a notebook.
-   Uses standard object-detection metrics.
-   Compares two training durations and reports both gains and
    regressions.
-   Includes visual prediction examples and identifies difficult
    classes.

## 12. Limitations

-   Only two short training runs were compared.
-   Metric differences are small and not consistently positive.
-   `test2007` was used for evaluation and model comparison, so it is
    not an untouched final test set.
-   Qualitative error analysis covers only a limited number of images.
-   Threshold testing was performed on one image, not across the
    dataset.
-   Speed/size optimization has not yet been measured.
-   Dataset and checkpoint paths are specific to the Kaggle environment.
-   The dataset and trained weights are not automatically included in
    GitHub.

## 13. Future work

1.  Use separate training, validation, and untouched test splits.
2.  Run controlled experiments that change one hyperparameter at a time.
3.  Inspect false positives, missed detections, confusion patterns, and
    performance by object size.
4.  Evaluate confidence thresholds across the full validation set.
5.  Measure latency and model size, then test ONNX or FP16 where
    supported.
6.  Test on external images outside Pascal VOC.
7.  Provide a stable checkpoint download or a clear retraining
    procedure.
8.  Add a small app/API if required for a more accessible demonstration.

## 14. Reproducibility and repository contents

Recommended repository structure:

``` text
PascalVision/
├── PascalVision.ipynb
├── README.md
├── requirements.txt
├── model_comparison.csv
└── reports/
    └── technical_report.md
```

To reproduce: open the notebook in Kaggle or a compatible Python
environment; install the listed dependencies; obtain and prepare Pascal
VOC; update the YAML paths and confirm the class mapping; run cells in
order; then train/evaluate or load a saved checkpoint. Record software
versions, hardware, and changed settings. Results can vary with
versions, random seeds, data preparation, hardware, and training
settings.

## 15. AI-use disclosure

AI tools were used to assist with coding guidance, troubleshooting,
notebook organization, and drafting this report. The project author is
responsible for checking the implementation, verifying reported metrics
against the executed notebook, understanding the code, and following
course rules for AI disclosure.

## 16. Conclusion

PascalVision demonstrates a working YOLO-based object-detection workflow
on Pascal VOC, including model training, quantitative evaluation, and
visual inference. The 5- and 10-epoch runs achieved similar results: the
longer run slightly improved recall and mAP50-95 but slightly reduced
precision and mAP50. Per-class results identify pottedplant and chair as
important areas for improvement. The current work is a useful baseline;
any claim of speed/size optimization needs additional measured evidence
if required by the assignment.
