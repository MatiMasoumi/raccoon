from google.colab import drive
drive.mount('/content/drive')
#!pip install torchmetrics
import torch as tc
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision as tv
from torchvision import models, datasets, transforms

from tqdm import tqdm

import matplotlib.pyplot as plt

from torchmetrics.detection import MeanAveragePrecision

from PIL import Image

from matplotlib.patches import Rectangle

import pandas as pd

import os
device = 'cuda' if tc.cuda.is_available() else 'cpu'

device
model = models.detection.fasterrcnn_resnet50_fpn(True)
model = model.to(device)
image = Image.open('/content/drive/MyDrive/DL_NonProfit/R-CNN/Vignoble-Reims©www.mkb_.photos-Coll.-ADT-Marne9.jpg').convert('RGB')
image

image_tensor = transforms.functional.to_tensor(image).to(device)
model.eval()
with tc.no_grad():
  predict = model([image_tensor])
predict
boxes = predict[0]['boxes'].cpu()
labels = predict[0]['labels'].cpu()
scores = predict[0]['scores'].cpu()
plt.imshow(np_arr)

ax = plt.gca()
for box, label, score in zip(boxes, labels, scores):
  if score>0.5:
    rect = Rectangle((box[0], box[1]),
                     (box[2] - box[0]),
                     (box[3] - box[1]),
                     fill = False,
                     edgecolor = (1, 0, 0),
                     linewidth = 2)
    ax.add_patch(rect)

plt.show()
#!git clone https://github.com/experiencor/raccoon_dataset
class CustomDataset(Dataset):
  def __init__(self, root, phase):
    super(CustomDataset, self).__init__()
    self.root = root
    self.targets = pd.read_csv(root + f'/data/{phase}_labels.csv')
    self.images = os.listdir(os.path.join(self.root, 'images'))

  def __getitem__(self, index):
    image_path = os.path.join(self.root, 'images', self.images[index])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.functional.to_tensor(image)

    box_list = self.targets[self.targets['filename']==self.images[index]]
    boxes = box_list[['xmin','ymin','xmax','ymax']].values
    box_tensor = tc.tensor(boxes)

    labels = tc.ones(size=(box_tensor.shape[0], ), dtype=tc.int64)

    targets = {}
    targets['boxes'] = box_tensor
    targets['labels'] = labels

    return image_tensor, targets
  def __len__(self):
    return self.targets.shape[0]
  
train_dataset = CustomDataset('/content/raccoon_dataset', 'train')
test_dataset = CustomDataset('/content/raccoon_dataset', 'test')

def new_concat(batch):
   return tuple(zip(*batch)) # [(img1, lab1), (img2, lab2) ,...]

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=new_concat)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=new_concat)
next(iter(train_loader))
len(train_dataset), len(test_dataset)
model
model.roi_heads.box_predictor
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train_one_epoch(model, train_loader, loss_func, optimizer, schedular, epoch):
    model.train()

    train_loss = AverageMeter()

    with tqdm(train_loader, unit='batch') as tepoch:
        for idx, (images, targets) in enumerate(tepoch):
           if epoch is not None:
               tepoch.set_description(f'Epoch:{epoch}')
        optimizer.zero_grad()
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(error for error in loss_dict.values())

        #grd
        loss.backward()
        optimizer.step()
        schedular.step()

        train_loss.update(loss.item())

        tepoch.set_postfix(loss=train_loss.avg)
    return model, train_loss.avg
def validation(model, valid_loaedr, loss_func, device):
   model.eval()
   model = model.to(device)
   map = MeanAveragePrecision('xyxy') # Mean Average Precision (mAP)
   for batch in valid_loaedr:
    images  = [image.to(device) for image in batch[0]]
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
    with tc.no_grad():
        pred = model(images)


        map.update(pred, targets)
   print('========================================================================================')
   print(f"|| mAP               = {map.compute()['map']:0.2}      ||    mAP|50     = {map.compute()['map_50']:0.2}    ||   mAP|75    ={ map.compute()['map_75']:0.2}  ||")
   print('========================================================================================')
   print(f"|| mAP|small         = {map.compute()['map_small']:0.2}      ||    mAP|medium = {map.compute()['map_medium']:0.2}    ||   mAP|large = {map.compute()['map_large']:0.2} ||")
   print('========================================================================================')
   print(f"|| mAR|1             = {map.compute()['mar_1']:0.2}      ||    mAR|10     = {map.compute()['mar_10']:0.2}    ||   mAR|100   = {map.compute()['mar_100']:0.2} ||")
   print('========================================================================================')
   print(f"|| mAR|small         = {map.compute()['mar_small']:0.2}      ||    mAR|medium = {map.compute()['mar_medium']:0.2}    ||   mAR|large = {map.compute()['mar_large']:0.2} ||")
   print('========================================================================================')
   print(f"|| mAP|per_class     = {map.compute()['map_per_class']:0.2}      ||                         ||                    ||")
   print('========================================================================================')
   print(f"|| mAR|100_per_class = {map.compute()['mar_100_per_class']:0.2}      ||                         ||                    ||")
   print('========================================================================================')
   return map

model = models.detection.fasterrcnn_resnet50_fpn(True)
model.roi_heads.box_predictor = tv.models.detection.faster_rcnn.FastRCNNPredictor(in_channels = 1024, num_classes = 2)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 1e-4)
shedular  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1000, eta_min = 1e-6)
start, end = 0, 10
for i in range(start, end):

  model, train_loss = train_one_epoch(model, train_loader, None,  optimizer, shedular, i)
  map = validation(model, test_loader, None, device)
image = Image.open('/content/drive/MyDrive/DL_NonProfit/R-CNN/raccoon-117.jpg').convert('RGB')
image
image_tensor = transforms.functional.to_tensor(image).to(device)
model.eval()
with tc.no_grad():
  predict = model([image_tensor])

boxes = predict[0]['boxes'].cpu()
labels = predict[0]['labels'].cpu()
scores = predict[0]['scores'].cpu()
# np_arr = image_tensor.permute(1, 2, 0).cpu()
plt.imshow(np_arr)

ax = plt.gca()
for box, label, score in zip(boxes, labels, scores):
  if score>0.7:
    rect = Rectangle((box[0], box[1]), #xmin, ymin
                     (box[2] - box[0]),
                     (box[3] - box[1]),
                     fill = False,
                     edgecolor = (1, 0, 0),
                     linewidth = 2)
    ax.add_patch(rect)

plt.show()
