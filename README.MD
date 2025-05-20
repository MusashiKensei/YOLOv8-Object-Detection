# YOLOv8 Object Detection: Signboard Detection

This project implements an object detection system using the YOLOv8 model to detect **Signboards** from images. The dataset was annotated using **CVAT**, preprocessed, and split into training, validation, and test sets. The model was trained using this custom-labeled data, evaluated with standard performance metrics, and tested on unseen images for real-world prediction.

---

## Key Features

- Trained **YOLOv8s** model for detecting signboards.
- Supports **training**, **validation**, and **inference**.
- Organized image data into YOLO-compliant structure: `images/` and `labels/` for `train`, `val`, and `test`.
- Visualizes key YOLOv8 outputs (confusion matrix, PR/P/R curves, batch predictions).
- Stores trained weights and predictions.

---

## Dataset & Model Files

The dataset consists of custom-labeled signboard images annotated using CVAT (Computer Vision Annotation Tool) and converted to YOLO format:
  - `images/` — Contains images split into `train/`, `val/`, and `test/`.
  - `labels/` — YOLO-format `.txt` files with bounding box coordinates.
  
**Note:** Due to large size, the `data/`, `runs/`, and `weights/` folders are hosted externally.

| Folder      | Download Link |
|-------------|--------------|
| **Main Folder (Object Detection)** | [Link](https://1024terabox.com/s/1hXJHMUzhMkVZY0ob63DHxA) |
| **weights** (e.g., `best.pt`)     | [Link](https://1024terabox.com/s/1UeX81l_LEbOM0zzLKwyccg) |
| **runs** (e.g., training results) | [Link](https://1024terabox.com/s/1gJ439zWsCZnfOGH5FlmBOA) |
| **data** (images and labels)      | [Link](https://1024terabox.com/s/1R0_VnsJRYmV6oAfUrtx6nw) |

---

## How to Run

1. **Install dependencies**:
    ```bash
    pip install ultralytics
    ```

2. **Set up your dataset**: Place the `data/`, `weights/`, and `runs/` folders in the root directory (or update paths accordingly in `data.yaml` and code).

3. **Train the model**:
    ```python
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    model.train(data="data.yaml", epochs=100, imgsz=640, batch=30, save=True)
    ```

4. **Validate the model**:
    ```python
    model = YOLO("runs/detect/train/weights/best.pt")
    model.val(data="data.yaml", imgsz=640, conf=0.6, iou=0.7)
    ```

5. **Run predictions**:
    ```python
    model = YOLO("weights/best.pt")
    model.predict(source="data/images/test", conf=0.6, save=True)
    ```

---

## Model Performance

Example metrics obtained after training and validation:

- **Precision**: 0.75  
- **Recall**: 0.71  
- **mAP@0.5**: 0.74  
- **mAP@0.5:0.95**: 0.59

Output visualizations include:

- Confusion matrix
- Precision-Recall (PR) curves
- Annotated validation batch images

---

## Notes

- The `data.yaml` file defines dataset paths, number of classes (`nc: 1`), and the class name (`Signboard`).
- Training and evaluation were performed on a CUDA-enabled GPU (`device=0`).
- Ensure paths in `data.yaml` and scripts match your local environment.
