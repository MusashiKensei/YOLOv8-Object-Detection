📦 YOLOv8 Object Detection - Signboard Detector
This project implements an object detection system using the YOLOv8 model to detect signboards from images. The dataset was annotated using CVAT, preprocessed, and split into training, validation, and test sets. The model was trained using this custom-labeled data, evaluated with standard performance metrics, and tested on unseen images for real-world prediction.

📁 Project Structure
bash
Copy
Edit
YOLOv8-Object-Detection/
├── object_detection.ipynb      # Jupyter Notebook for training, validation, and prediction
├── object_detection.py         # Python script version of the notebook
├── data.yaml                   # Configuration file for YOLO training
├── yolov8s/                    # Model configuration/code (if modified)
├── Data/                       # [Uploaded to TeraBox] Contains images and labels
│   ├── images/
│   └── labels/
├── runs/                       # [Uploaded to TeraBox] YOLO training and prediction outputs
├── weights/                    # [Uploaded to TeraBox] Saved model weights (best.pt, last.pt, etc.)
🚀 Features
Trained YOLOv8s model for detecting signboards.

Supports training, validation, and inference.

Includes code to prepare dataset from CSV and copy relevant files.

Visualizes key YOLOv8 outputs (confusion matrix, PR/P/R curves, batch predictions).

Stores trained weights and predictions.

📦 Dataset Description
The dataset consists of custom-labeled signboard images annotated using CVAT (Computer Vision Annotation Tool) and converted to YOLO format:

images/ — Contains images split into train/, val/, and test/.

labels/ — YOLO-format .txt files with bounding box coordinates.

Note: Due to large size, the full dataset is hosted externally.

🔗 Download Dataset (TeraBox)

🛠 Model Training
Training is done using ultralytics YOLOv8:

python
Copy
Edit
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=30, device=0)
The model learns to detect signboards from scratch using a single class:

yaml
Copy
Edit
nc: 1
names: ['Signboard']
📈 Model Evaluation
Validation includes:

Precision: 75.71%

Recall: 71.75%

mAP@0.5: 74.84%

mAP@0.5:0.95: 59.92%

Output visualizations include:

Confusion matrix

PR / Precision / Recall curves

Annotated validation batch images

🔍 Prediction on Test Images
Inference is run using:

python
Copy
Edit
model.predict(source="data/images/test", conf=0.6, iou=0.7)
Predicted images are saved under:

bash
Copy
Edit
runs/detect/predict/
📥 Pre-trained Weights & Outputs
weights/best.pt — Final trained model.

runs/ — Contains training and prediction outputs.

🔗 Download Weights and Outputs (TeraBox)

✅ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/MusashiKensei/YOLOv8-Object-Detection
cd YOLOv8-Object-Detection
Install requirements

bash
Copy
Edit
pip install ultralytics pandas
Download dataset & weights from TeraBox

Place the Data/ folder into the root directory.

Place the weights/ and runs/ folders similarly.

🧠 Acknowledgements
YOLOv8 by Ultralytics

Labeled images were annotated using CVAT (Computer Vision Annotation Tool).
