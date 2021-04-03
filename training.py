from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import PSRoIAlign
#from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict
import warnings
from torch.nn.modules.utils import _pair
from torch.jit.annotations import List
from torch.autograd import Variable
from torch.nn import init
import math
from tqdm.autonotebook import tqdm
import sys
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2
from skimage import io
import torch.optim as optim
import torchvision
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align
from torchvision.models.detection import _utils as det_utils
import traceback
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as patches

from thundernet import ThunderNet

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Running on the GPU")
else:
  device = torch.device("cpu")
  print("Running on the CPU")

EPOCHS = 2
EPOCH_NUMBER = -1
TRAIN_FROM_ZERO = True
REBUILD_DATA = False
TRAINING_DATA_IDS_PATH = '/content/drive/My Drive/GMM/training_data_ids.npy'
TRAINING_DATA_IMAGES_PATH = '/content/drive/My Drive/GMM/training_data_images.npy'
TRAINING_DATA_ANNOTATIONS_PATH = '/content/drive/My Drive/GMM/training_data_annotations.npy'
VALIDATION_DATA_IDS_PATH = '/content/drive/My Drive/GMM/validation_data_ids.npy'
VALIDATION_DATA_IMAGES_PATH = '/content/drive/My Drive/GMM/validation_data_images.npy'
VALIDATION_DATA_ANNOTATIONS_PATH = '/content/drive/My Drive/GMM/validation_data_annotations.npy'
MOCK_TARGETS_PATH = '/content/drive/My Drive/GMM/mock_targets.npy'

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, common_size=320):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))      # image resize

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale      # resize boxes, [x1, y1, x2, y2]

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]   # flip

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class OpenImagesDataset(Dataset):
  def __init__(self, files, annotations, transforms = None):
    self.files = files
    self.annotations = annotations
    self.transforms = transforms

  def __getitem__(self, idx):
    img_path = self.files[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    annots = np.zeros((0, 5))
    param_annots = self.annotations[idx]
    for i, a in enumerate(param_annots):
      annot = np.zeros((1, 5))
      annot[0, 0] = a[0]
      annot[0, 1] = a[1]
      annot[0, 2] = a[2]
      annot[0, 3] = a[3]
      annot[0, 4] = a[4]
      annots = np.append(annots, annot, axis = 0)

    sample = {'img': img, 'annot': annots}
    if self.transforms is not None:
        sample = self.transforms(sample)

    #permute
    sample['img'] = sample['img'].permute(2, 0, 1)

    return sample

  def __len__(self):
    return len(self.files)

class DataBuilder():
  scissors_count = 0
  pandas_count = 0
  snakes_count = 0

  def __init__(self, datamode, maximagecount):
    self.data_mode = datamode
    self.max_image_count = maximagecount
    self.SCISSORS = '/content/drive/My Drive/GMM/Data/' + self.data_mode + '/scissors'
    self.PANDAS = '/content/drive/My Drive/GMM/Data/' + self.data_mode + '/panda'
    self.SNAKES = '/content/drive/My Drive/GMM/Data/' + self.data_mode + '/snake'
    self.LABELS = {self.SCISSORS: 0, self.PANDAS: 1, self.SNAKES: 2}
    self.files = []
    self.annotations = []
    self.ids = []

  def make_data(self):
    self.scissors_count = 0
    self.pandas_count = 0
    self.snakes_count = 0

    self.files = []
    self.annotations = []
    self.ids = []

    for label in self.LABELS:
      current_count = 0
      label_number = self.LABELS[label]
      for f in tqdm(os.listdir(label)):
        if current_count < self.max_image_count:
          if 'jpg' in f:
            try:
              img_path = os.path.join(label, f)
              self.files.append(img_path)
              id = Path(img_path).stem
              self.ids.append(id)
              annot_path = os.path.join(label, 'labels', id + '.txt')
              annot_file = open(annot_path, 'r')
              Lines = annot_file.readlines()
              annots = []
              for line in Lines:
                line = line.strip()
                words = line.split()
                annot_line = []
                annot_line.append(float(words[1]))
                annot_line.append(float(words[2]))
                annot_line.append(float(words[3]))
                annot_line.append(float(words[4]))
                annot_line.append(label_number)
                annots.append(annot_line)
              self.annotations.append(annots)

              current_count += 1
              if label == self.SCISSORS:
                self.scissors_count += 1
              elif label == self.PANDAS:
                self.pandas_count += 1
              elif label == self.SNAKES:
                self.snakes_count += 1

            except Exception as e:
              print(e)
              traceback.print_exc()
              #pass

    if self.data_mode == 'train':
      print('Train')
      print(len(self.ids))
      print(len(self.files))
      print(len(self.annotations))
      np.save(TRAINING_DATA_IDS_PATH, np.array(self.ids))
      print('Saved ids')
      np.save(TRAINING_DATA_IMAGES_PATH, np.array(self.files))
      print('Saved images')
      np.save(TRAINING_DATA_ANNOTATIONS_PATH, np.array(self.annotations))
      print('Saved annotations')
    elif self.data_mode == 'validation':
      print('Validation')
      print(len(self.ids))
      print(len(self.files))
      print(len(self.annotations))
      np.save(VALIDATION_DATA_IDS_PATH, np.array(self.ids))
      np.save(VALIDATION_DATA_IMAGES_PATH, np.array(self.files))
      np.save(VALIDATION_DATA_ANNOTATIONS_PATH, np.array(self.annotations))

if REBUILD_DATA:
  traindatabuilder = DataBuilder('train', 292)
  traindatabuilder.make_data()
  validationdatabuilder = DataBuilder('test', 6)
  validationdatabuilder.make_data()

def load_data():
  t_ids = np.load(TRAINING_DATA_IDS_PATH, allow_pickle = True).tolist()
  t_files = np.load(TRAINING_DATA_IMAGES_PATH, allow_pickle = True).tolist()
  t_annotations = np.load(TRAINING_DATA_ANNOTATIONS_PATH, allow_pickle = True).tolist()

  v_ids = np.load(VALIDATION_DATA_IDS_PATH, allow_pickle = True).tolist()
  v_files = np.load(VALIDATION_DATA_IMAGES_PATH, allow_pickle = True).tolist()
  v_annotations = np.load(VALIDATION_DATA_ANNOTATIONS_PATH, allow_pickle = True).tolist()

  return t_ids, t_files, t_annotations, v_ids, v_files, v_annotations

def load_model(model, epoch_number = EPOCH_NUMBER, optimizer = None):
  model_path = '/content/drive/My Drive/GMM'
  model_name = '{}/thundernet_{}.pth.tar'.format(model_path, epoch_number)
  checkpoint = torch.load(model_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

def train():
  transform_train = transforms.Compose([
      Normalizer(),
      Augmenter(),
      Resizer()
  ])

  transform_test = transforms.Compose([
      Normalizer(),
      Resizer()
  ])

  t_ids, t_files, t_annotations, v_ids, v_files, v_annotations = load_data()

  t_dataset = OpenImagesDataset(t_files, t_annotations, transforms = transform_train)
  v_dataset = OpenImagesDataset(v_files, v_annotations, transforms = transform_test)

  t_data_loader = torch.utils.data.DataLoader(t_dataset, batch_size = 2)
  v_data_loader = torch.utils.data.DataLoader(v_dataset, batch_size = 2)

  model = ThunderNet(device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
  if not TRAIN_FROM_ZERO:
    load_model(model, optimizer = optimizer)

  milestones = [500, 800, 1200, 1500]
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=EPOCH_NUMBER)

  with open('/content/drive/My Drive/GMM/th_model.log', 'a') as f:
    for epoch in range(EPOCH_NUMBER + 1, EPOCHS + EPOCH_NUMBER + 1):
      train_loss = train_epoch(t_data_loader, model, optimizer, epoch, scheduler)
      detections, validation_loss = validate(v_data_loader, model)

      print("TEST")
      f.write(f"{'Model-1'},{epoch},{round(float(train_loss),4)},{round(float(validation_loss), 4)}\n")
      print(f"{'Model-1'},{epoch},{round(float(train_loss),4)},{round(float(validation_loss), 4)}\n")
      scheduler.step()

      model_path = '/content/drive/My Drive/GMM'
      model_name = '{}/thundernet_{}.pth.tar'.format(model_path, epoch)
      torch.save({
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
      }, model_name)

def create_loss_graph():
  contents = open('/content/drive/My Drive/GMM/th_model.log', 'r').read().split('\n')
  print(contents)
  epochs = []
  train_losses = []
  validation_losses = []

  for c in contents:
    try:
      name, epoch, tr_loss, val_loss = c.split(',')
      epochs.append(epoch)
      train_losses.append(float(tr_loss))
      validation_losses.append(float(val_loss))
    except Exception as e:
      pass

  #fig = plt.figure()
  #ax = plt.subplot2grid((2, 1), (0, 0))
  plt.plot(epochs, train_losses, label = 'train_loss', marker = 'o')
  plt.plot(epochs, validation_losses, label = 'val_loss', marker = 'o')
  plt.legend(loc = 2)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
  plt.show()

def train_epoch(t_data_loader, model, optimizer, epoch, scheduler):
  epoch_loss = []
  losses = {}
  progress_bar = tqdm(t_data_loader)

  for i, data in enumerate(progress_bar):
    input_data = data['img'].cuda().float()
    input_labels = data['annot'].cuda()

    detections, detector_losses, proposal_losses = model(input_data, input_labels)
    losses.update(detector_losses)
    losses.update(proposal_losses)
    total_loss = sum(loss for loss in losses.values())

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    epoch_loss.append(total_loss.item())

  train_loss = np.mean(epoch_loss)
  return train_loss

def validate(v_data_loader, model):
  all_loss = []
  losses = {}
  progress_bar = tqdm(v_data_loader)

  for i, data in enumerate(progress_bar):
    with torch.no_grad():
      print(data['img'].shape)
      print(data['annot'])
      input_data = data['img'].cuda().float()
      input_labels = data['annot'].cuda()

      detections, detector_losses, proposal_losses = model(input_data, input_labels)
      print("Val")
      print(detections)
      losses.update(detector_losses)
      losses.update(proposal_losses)
      total_loss = sum(loss for loss in losses.values())

      all_loss.append(total_loss.item())

  validation_loss = np.mean(all_loss)
  return detections, validation_loss


def to_coord(pred, shape):
    _, w, h = shape
    x0 = max(int(pred[0]), 0)
    x1 = min(int(pred[1]), w)
    y0 = max(int(pred[2]), 0)
    y1 = min(int(pred[3]), h)

    return [x0, y0, x1 - x0, y1 - y0]


def show_image_boxes(img, prediction):
    fig, ax = plt.subplots()
    ax.imshow(img.byte().numpy())
    for pred in prediction[0]['boxes']:
        pred = pred.cpu().detach().numpy()
        predicted = pred  # to_coord(pred, img.shape)
        rect = patches.Rectangle((predicted[0], predicted[1]), predicted[2], predicted[3], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()

def mock_targets():
  url = '/content/drive/My Drive/GMM/Data/test/panda'
  f = '534968ad2c836abd.jpg'
  img_path = os.path.join(url, f)
  id = Path(img_path).stem
  annot_path = os.path.join(url, 'labels', id + '.txt')
  annot_file = open(annot_path, 'r')
  Lines = annot_file.readlines()
  annots = []
  for line in Lines:
    line = line.strip()
    words = line.split()
    annot_line = []
    annot_line.append(float(words[1]))
    annot_line.append(float(words[2]))
    annot_line.append(float(words[3]))
    annot_line.append(float(words[4]))
    annot_line.append(1)
    annots.append(annot_line)
  np.save(MOCK_TARGETS_PATH, np.array(annots))

transform_test = transforms.Compose([
      Normalizer(),
      Resizer()
  ])

def testavimas():
  url = '/content/drive/My Drive/GMM/Data/test/panda'
  #f = '534968ad2c836abd.jpg'
  #f = '20aec5942def0e8b.jpg'
  f = '0bc65d083d184788.jpg'
  img_path = os.path.join(url, f)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  show_img = img

  targets = np.load(MOCK_TARGETS_PATH, allow_pickle = True).tolist()
  annots = np.zeros((0, 5))
  for i, a in enumerate(targets):
    annot = np.zeros((1, 5))
    annot[0, 0] = a[0]
    annot[0, 1] = a[1]
    annot[0, 2] = a[2]
    annot[0, 3] = a[3]
    annot[0, 4] = a[4]
    annots = np.append(annots, annot, axis = 0)
  sample = {'img': img, 'annot': annots}
  sample = transform_test(sample)
  sample['img'] = sample['img'].permute(2, 0, 1)
  imgList = []
  imgList.append(sample['img'])
  targList = []
  targList.append(sample['annot'])

  model = ThunderNet().to(device)
  load_model(model, epoch_number = 3)
  detections, detector_losses, proposal_losses = model(torch.stack(imgList).cuda().float(), torch.stack(targList))
  print(detections)
  print(detector_losses)
  print(proposal_losses)
  show_image_boxes(torch.from_numpy(show_img), detections)