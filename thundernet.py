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

from roi_heads_custom import RoIHeads

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def load_backbone():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)
    model.stage3.register_forward_hook(get_activation('c4'))
    model.conv5.register_forward_hook(get_activation('c5'))
    return model


class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.conv1 = nn.Conv2d(232, 245, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(1024, 245, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(10)
        self.conv3 = nn.Conv2d(1024, 245, kernel_size=1, stride=1, padding=0)

    def forward(self, c4_feature, c5_feature):
        c4 = c4_feature
        c4_lat = self.conv1(c4)  # output: [245, 20, 20]

        c5 = c5_feature
        c5_lat = self.conv2(c5)  # output: [245, 10, 10]

        # upsample x2
        c5_lat = F.interpolate(input=c5_lat, size=[20, 20], mode="nearest")  # output: [245, 20, 20]
        c_glb = self.avg_pool(c5)  # output: [512, 1, 1]
        c_glb_lat = self.conv3(c_glb)  # output: [245, 1, 1]

        out = c4_lat + c5_lat + c_glb_lat  # output: [245, 20, 20]
        return out


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(256, 245, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(245)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rpn_feature, cem_feature):
        cem = cem_feature  # feature map of CEM: [245, 20, 20]
        rpn = rpn_feature  # feature map of RPN: [256, 20, 20]

        sam = self.conv(rpn.to(device))
        sam = self.bn(sam)
        sam = self.sigmoid(sam)
        out = cem * sam  # output: [245, 20, 20]
        return out


class RCNNSubNetHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(RCNNSubNetHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)  # in_channles: 7*7*5=245  representation_size:1024

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        return x


class ThunderNetPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(ThunderNetPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):  # x: [1024, 1, 1]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def to_dict_list(annotations):
    result = []
    for i, annots in enumerate(annotations):
        boxes_list = []
        labels_list = []
        for i, a in enumerate(annots):
            box = torch.tensor([a[0], a[1], a[2], a[3]])
            label = torch.tensor(int(a[4]))
            boxes_list.append(box)
            labels_list.append(label)
        boxes = torch.stack(boxes_list)
        labels = torch.stack(labels_list)
        dictionary = {
            'boxes': boxes,
            'labels': labels
        }
        result.append(dictionary)
    return result


class DetectNet(nn.Module):
    def __init__(self, backbone, num_classes, device,
                 # RPN parameters
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=100,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,

                 rpn_mns_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

                 # Box parameters
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super(DetectNet, self).__init__()

        out_channels = backbone.out_channels  # 245

        self.backbone = backbone

        self.cem = CEM()
        self.sam = SAM()

        # rpn
        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 0.75, 1.0, 1.33, 2.0),)
        rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_mns_thresh)

        box_ps_roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        # R-CNN subnet
        resolution = box_ps_roi_align.output_size[0]  # size: (7, 7)
        representation_size = 1024
        box_out_channels = 5
        in_ch = 12005
        box_head = RCNNSubNetHead(in_ch, representation_size)

        # representation_size = 1024
        box_predictor = ThunderNetPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            box_ps_roi_align, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 320
        max_size = 320
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # backbone
        self.backbone(images)
        c4_feature = activation['c4']
        c5_feature = activation['c5']
        if targets is not None:
            targets = to_dict_list(targets)
        images, targets = self.transform(images, targets)

        # cem
        cem_feature = self.cem(c4_feature, c5_feature)
        cem_feature_output = cem_feature

        if isinstance(cem_feature, torch.Tensor):
            cem_feature = OrderedDict([('0', cem_feature)])

        # rpn
        if targets is not None:
            for i in range(len(targets)):
                targets[i]['boxes'] = targets[i]['boxes'].to(device)
                targets[i]['labels'] = targets[i]['labels'].to(device)
        self.rpn.eval()
        proposals, proposal_losses = self.rpn(images, cem_feature, targets)
        # sam
        # PROPOSALS IS LIST OF TENSORS OF DIFFERENT SHAPES
        # SAM WANTS TENSOR INSTEAD OF LIST
        # DID SOME PADDING
        # Determine maximum length
        max_len_0 = max([x.size(0) for x in proposals])
        max_len_1 = max([x.size(1) for x in proposals])

        # pad all tensors to have same length
        proposals = [torch.nn.functional.pad(x, [0, max_len_1 - x.size(1), 0, max_len_0 - x.size(0)]) for x in
                     proposals]

        # stack them
        proposals = torch.stack(proposals)

        # add fourth dimension as sam expects it
        save_proposals = proposals
        proposals = proposals[:, :, :, None]

        # cut variable length of channels to sam expected (256)
        new_proposals = torch.zeros(2, 256, 20, 20)
        cut_proposals = []
        cut_proposal = torch.zeros(256, 20, 20)
        for i in range(len(proposals)):
            for j in range(256):
                third_dim = torch.zeros(20, 20)
                for k in range(20):
                    try:
                        if (k == 0 or k == 1 or k == 2 or k == 3):
                            third_dim[k] = proposals[i][j][k]
                        else:
                            third_dim[k] = torch.zeros(20)
                    except Exception as e:
                        third_dim[k] = torch.zeros(20)
                cut_proposal[j] = third_dim
            cut_proposals.append(cut_proposal)
        for i in range(len(cut_proposals)):
            new_proposals[i] = cut_proposals[i]

        sam_feature = self.sam(new_proposals, cem_feature_output)

        if isinstance(sam_feature, torch.Tensor):
            sam_feature = OrderedDict([('0', sam_feature)])

        detections, detector_losses = self.roi_heads(sam_feature, save_proposals, images.image_sizes, targets)

        return detections, detector_losses, proposal_losses


def ThunderNet(device):
    backbone = load_backbone()
    backbone.out_channels = 245
    thundernet = DetectNet(backbone, num_classes=4, device=device)

    return thundernet