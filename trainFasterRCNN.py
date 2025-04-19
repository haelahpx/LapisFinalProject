from pathlib import Path
import yaml

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from torch.amp import autocast, GradScaler

### GRY ###
train_dir = Path("datasets/_HD_GRY_CarAndPlateDetection/train/images")
val_dir = Path("datasets/_HD_GRY_CarAndPlateDetection/valid/images")
train_labels_dir = Path("datasets/_HD_GRY_CarAndPlateDetection/train/labels")
val_labels_dir = Path("datasets/_HD_GRY_CarAndPlateDetection/valid/labels")

# dataset configuration from data.yaml
with open("datasets/_HD_GRY_CarAndPlateDetection/data.yaml", "r") as file:
    data_config = yaml.safe_load(file)


### RGB ###
# train_dir = Path("datasets/_HD_RGB_CarAndPlateDetection/train/images")
# val_dir = Path("datasets/_HD_RGB_CarAndPlateDetection/valid/images")
# train_labels_dir = Path("datasets/_HD_RGB_CarAndPlateDetection/train/labels")
# val_labels_dir = Path("datasets/_HD_RGB_CarAndPlateDetection/valid/labels")

# with open("datasets/_HD_RGB_CarAndPlateDetection/data.yaml", "r") as file:
#     data_config = yaml.safe_load(file)

# label
num_classes = data_config["nc"]
class_names = data_config["names"]

print(num_classes)
print(class_names)


# class_names = ["car", "plate"] 

class_to_idx = {name: idx for idx, name in enumerate(class_names)}

class CustomDetectionDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transforms = transforms
        self.img_paths = list(img_dir.glob("*.jpg"))
        
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __getitem__(self, idx):

        # load image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)

        # Annotations (txt files)
        annotation_file = self.annotations_dir / (img_path.stem + ".txt")
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        boxes = []
        labels = []
        
        with open(annotation_file, 'r') as f:
            for line in f:
                # each line is expected to contain: class_idx x_center y_center width height
                parts = line.strip().split()  # BECAUSE space-separated values
                class_idx = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # convert the COCO-style bbox to (xmin, ymin, xmax, ymax)
                xmin = (x_center - width / 2)
                ymin = (y_center - height / 2)
                xmax = (x_center + width / 2)
                ymax = (y_center + height / 2)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_idx)

        # convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.img_paths)


# DataLoader
train_dataset = CustomDetectionDataset(train_dir, train_labels_dir, transforms=None)
val_dataset = CustomDetectionDataset(val_dir, val_labels_dir, transforms=None)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


### Hardware Configuration ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Model Declaration ###

# Load model
# model = fasterrcnn_resnet50_fpn(pretrained=True)

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # or COCO_V1 if you want exactly what pretrained=True used to give
model = fasterrcnn_resnet50_fpn(weights=weights)

# idk what is this
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


### Fine Tunning ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler('cuda')

# Model
for epoch in range(20):
    model.train()
    for images, targets in train_dataloader:
        # Skip images with no boxes
        valid_data = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]

        if len(valid_data) == 0:
            continue

        images, targets = zip(*valid_data)

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch [{epoch+1}/20], Train Loss: {losses.item():.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
    if len(val_dataloader) > 0:
        val_loss /= len(val_dataloader)
        print(f"Epoch [{epoch+1}/20], Val Loss: {val_loss:.4f}")
    else:
        print("Warning: Validation dataloader is empty. Skipping val loss calculation.")
    print(f"Epoch [{epoch+1}/20], Val Loss: {val_loss:.4f}")
    model.train()

    # CHECKPOINT SAVING
    if (epoch + 1) % 10 == 0:
        os.makedirs("vec-FasterRCNN", exist_ok=True)
        checkpoint_path = os.path.join("vec-FasterRCNN", f"fasterrcnn_checkpoint_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
