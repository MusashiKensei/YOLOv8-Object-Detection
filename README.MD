# YOLOv8 Object Detection – Signboard Detection Project

This project leverages the YOLOv8 deep learning model to perform object detection on **signboards** from real-world images. The goal is to accurately detect and classify signboards using a custom dataset, with model training, validation, and prediction workflows implemented in Python and Jupyter Notebook.

---

## Project Structure

Object Detection/
├── .gitattributes
├── .gitignore
├── data.yaml
├── object_detection.ipynb
├── object_detection.py
├── yolov8s/                 # YOLO model folder (probably config or code)
├── Data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── runs/
│   └── detect/
│       └── predict/
├── weights/
    ├── best.pt
    ├── epoch1.pt
    ├── ...
    └── last.pt


---

## Key Features

- Uses the **Ultralytics YOLOv8** model.
- Trained on a **custom signboard dataset**.
- Organized image data into YOLO-compliant structure: `images/` and `labels/` for `train`, `val`, and `test`.
- Includes steps for:
  - Data preparation from CSVs and copying assets.
  - Model training with configuration tuning (epochs, batch size, optimizer).
  - Validation and evaluation (PR curve, confusion matrix, etc.).
  - Inference and visualization on test data.

---

## Dataset & Model Files

Due to large file sizes, the `data/`, `runs/`, and `weights/` folders are hosted on Terabox. You can access them below:

| Folder      | Terabox Link |
|-------------|--------------|
| **Main Folder (Object Detection)** | [Link](https://1024terabox.com/s/1hXJHMUzhMkVZY0ob63DHxA) |
| **weights/** (e.g., `best.pt`)     | [Link](https://1024terabox.com/s/1UeX81l_LEbOM0zzLKwyccg) |
| **runs/** (e.g., training results) | [Link](https://1024terabox.com/s/1gJ439zWsCZnfOGH5FlmBOA) |
| **data/** (images and labels)      | [Link](https://1024terabox.com/s/1R0_VnsJRYmV6oAfUrtx6nw) |

> Download these folders and place them in the root directory of the project to run the code end-to-end.

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

> Confusion matrices and PR curves can be found inside the `runs/detect/` folder.

---

## Notes

- The `data.yaml` file defines dataset paths, number of classes (`nc: 1`), and the class name (`Signboard`).
- Training and evaluation were performed on a CUDA-enabled GPU (`device=0`).
- Ensure paths in `data.yaml` and scripts match your local environment.
