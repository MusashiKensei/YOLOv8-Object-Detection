# %%
%cd "C:\Users\Rizvi\Desktop\Milestone_2"

# %% [markdown]
# # **Imports**

# %%
from pathlib import Path
import random
import pandas as pd

# %%
import os
import shutil
import pandas as pd

# Define paths
milestone_folder = "/content/drive/MyDrive/CSE428 Project/Milestone_2"
data_folder = os.path.join(milestone_folder, "data")
csv_folder = os.path.join(milestone_folder, "section2-group2")
global_images_folder = os.path.join(milestone_folder, "global_images")
global_annotations_folder = os.path.join(milestone_folder, "global_annotations")

# Create data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Function to copy images and annotations
def copy_files(csv_file, src_images_folder, src_annotations_folder, dest_images_folder, dest_annotations_folder):
    df = pd.read_csv(os.path.join(csv_folder, csv_file))
    for index, row in df.iterrows():
        image_file = row[0]
        annotation_file = row[1]
        shutil.copy(os.path.join(src_images_folder, image_file), os.path.join(dest_images_folder, image_file))
        shutil.copy(os.path.join(src_annotations_folder, annotation_file), os.path.join(dest_annotations_folder, annotation_file))

# Create train, validation, and test folders
for folder in ["train", "validation", "test"]:
    folder_path = os.path.join(data_folder, folder)
    images_folder = os.path.join(folder_path, "images")
    annotations_folder = os.path.join(folder_path, "annotations")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

# Copy files for train, validation, and test sets
copy_files("train.csv", global_images_folder, global_annotations_folder, os.path.join(data_folder, "train", "images"), os.path.join(data_folder, "train", "annotations"))
copy_files("val.csv", global_images_folder, global_annotations_folder, os.path.join(data_folder, "validation", "images"), os.path.join(data_folder, "validation", "annotations"))
copy_files("test.csv", global_images_folder, global_annotations_folder, os.path.join(data_folder, "test", "images"), os.path.join(data_folder, "test", "annotations"))

print("Folder structure created and files copied successfully.")


# %%
data_folder = "/content/drive/MyDrive/CSE428 Project/Milestone_2/data"

# Function to count files in a folder
def count_files(folder_path):
    num_files = len(os.listdir(folder_path))
    return num_files

# Function to count image and annotation files in train, validation, and test folders
def count_files_in_folders(data_folder):
    for folder in ["train", "validation", "test"]:
        images_folder = os.path.join(data_folder, folder, "images")
        annotations_folder = os.path.join(data_folder, folder, "annotations")
        num_images = count_files(images_folder)
        num_annotations = count_files(annotations_folder)
        print(f"Folder: {folder}")
        print(f"Number of image files: {num_images}")
        print(f"Number of annotation files: {num_annotations}")
        print()

# Count files in train, validation, and test folders
count_files_in_folders(data_folder)


# %%
import torch
print(torch.cuda.is_available())

# %% [markdown]
# # **Install Model**

# %%
from IPython import display
display.clear_output()


# %%

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# %%
model = YOLO("yolov8s.pt")

# %% [markdown]
# # **Training Dataset**

# %%
%cd "C:\Users\Rizvi\Desktop\Milestone_2"

#!yolo task=detect mode=train model=yolov8s.pt data= data.yaml epochs=100 imgsz=640 batch = 30 plots=True optimizer = Adam save=True device = 0

results = model.train(data= "data.yaml", epochs=100, imgsz=640, batch = 30, plots=True, save_period =1, save=True, device = 0)

# %%
#!ls runs/detect/train/

# %%
from IPython.display import display, Image
Image(filename='runs/detect/train/confusion_matrix_normalized.png', width=1000)

# %%
Image(filename='runs/detect/train/results.png', width=1000)

# %%
Image(filename='runs/detect/train/P_curve.png', width=1000)

# %%
Image(filename='runs/detect/train/PR_curve.png', width=1000)

# %%
Image(filename='runs/detect/train/R_curve.png', width=1000)

# %%
Image(filename='runs/detect/train/train_batch0.jpg', width=1000)

# %%
Image(filename='runs/detect/train/train_batch1.jpg', width=1000)

# %%
Image(filename='runs/detect/train/train_batch2.jpg', width=1000)

# %% [markdown]
# # **Validation**

# %%
#!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

model = YOLO("runs/detect/train/weights/best.pt")
model.val(data = 'data.yaml', imgsz = 640, batch = 30, conf = 0.6, iou = 0.7, device = 0, plots = True, save = True)

# %% [markdown]
# **results_dict: {'metrics/precision(B)': 0.7571523619427811, 'metrics/recall(B)': 0.7175283732660782, 'metrics/mAP50(B)': 0.7484993829651984, 'metrics/mAP50-95(B)': 0.5992028009264534, 'fitness': 0.614132459130328}**

# %%
Image(filename='runs/detect/val/confusion_matrix_normalized.png', width=1000)

# %%
Image(filename='runs/detect/val/P_curve.png', width=1000)

# %%
Image(filename='runs/detect/val/PR_curve.png', width=1000)

# %%
Image(filename='runs/detect/val/R_curve.png', width=1000)

# %%
Image(filename='runs/detect/val/val_batch0_pred.jpg', width=1000)

# %%
Image(filename='runs/detect/val/val_batch1_pred.jpg', width=1000)

# %%
Image(filename='runs/detect/val/val_batch2_pred.jpg', width=1000)

# %% [markdown]
# # **Prediction**

# %%
#!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=data/images/test save=True   show_conf=True show_labels=True show_boxes=True

model = YOLO("weights/best.pt")
model.predict(data = 'data.yaml', imgsz = 640, batch = 30, conf = 0.6, source = "data/images/test", iou = 0.7, plots = True, save = True)


# %%
Image(filename='runs/detect/predict/2703.png', width=1000)

# %%
Image(filename='runs/detect/predict/ID_34.jpg', width=1000)

# %%



