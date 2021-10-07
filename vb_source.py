# Importing Pytorch and initializing the device
import torch
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
# Force CPU
device = torch.device("cpu")
print('Using device', device)

import warnings
warnings.filterwarnings("ignore")

# fastrcnn import
from fastrcnn import *
from fastrcnn import FastRCNNOutputs

# More imports
import torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
from detectron2.modeling.roi_heads.fast_rcnn import *
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutput # Code doesn't have this
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers # Can import only this
# check this - https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from transformers import BertTokenizer, VisualBertForPreTraining, VisualBertModel

# Load the data
img_data = np.load("../data/image_array.npy")
txt_data = np.load("../data/text_array.npy")
labels_data = np.load("../data/labels.npy")
ids_data = np.load("../data/ids.npy")
# Printing the shapes
print(img_data.shape)
print(txt_data.shape)
print(labels_data.shape)
print(ids_data.shape)

# Reshape image to -> num_images, sources, num_channels, width, heigth
#NOTE: Can convert image data to tensor only in training loop with very less batch size
num_images, sources, width, height, num_channels = img_data.shape
img_data_reshape = np.reshape(img_data, newshape=(num_images, sources, width, height, num_channels))
img_data_target = img_data_reshape[:,0,:,:,:] # Don't convert to GPU
img_data_source = img_data_reshape[:,1,:,:,:] # Don't convert to GPU
print('New Target Shape', img_data_target.shape)
print('New Source Shape', img_data_source.shape)

# TODO: Create rgb cv2 batch
img_cv2_source = []
img_cv2_target = []
for i in range(len(img_data_source)):
    img_cv2_source.append(cv2.cvtColor(img_data_source[i], cv2.COLOR_RGB2BGR))
    img_cv2_target.append(cv2.cvtColor(img_data_target[i], cv2.COLOR_RGB2BGR))

# Loading Config and Model Weights
cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    # cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg
cfg = load_config_and_model_weights(cfg_path) 

def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

model = get_model(cfg).to(device)

# Convert Image to Model input
def prepare_image_inputs(cfg, img_list):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
    
    return images, batched_inputs

# Getting features
def get_features(model, images):
    model = model.to(device)
    features = model.backbone((images.tensor).to(device))
    return features

# Getting region proposals 
def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

# Get box features for proposal
def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)
    if box_features.shape[0] != features['p2'].shape[0]*1000: # Cause for error
        diff = features['p2'].shape[0]*1000 - box_features.shape[0]
        extra_array = torch.zeros(diff, 1024)
        box_features = torch.cat((box_features, extra_array), 0)
    box_features = box_features.reshape(features['p2'].shape[0], 1000, 1024) # depends on your config and batch size
    return box_features, features_list

# Get prediction logits and boxes
def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

# Getting box scores
def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, source_proposals):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        source_proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes

# Rescale boxes to the original size
def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

# Select boxes using NMS
def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    out_box_tensor = output_boxes.tensor.detach()
    if out_box_tensor.shape[0] != 80000: # Source for error
        diff = 80000 - output_boxes.tensor.detach().shape[0]
        extra_array = torch.zeros(diff, 4)
        out_box_tensor = torch.cat((out_box_tensor, extra_array), 0)
    cls_boxes = out_box_tensor.reshape(1000,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        if cls_scores.shape[0] != 1000:
            cls_scores = torch.cat((cls_scores, torch.zeros(1000-cls_scores.shape[0])), 0)
        det_boxes = cls_boxes[:,cls_ind,:]
        if det_boxes.shape[0] != 1000:
            det_boxes = torch.cat((det_boxes, torch.zeros(1000-det_boxes.shape[0])), 0)
        keep = np.array(nms(det_boxes.cpu(), cls_scores.cpu(), test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores.cpu()[keep] > max_conf.cpu()[keep], cls_scores.cpu()[keep], max_conf.cpu()[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

# Limit the number of embeddings
MIN_BOXES=10
MAX_BOXES=100
def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

# Getting the final visual embeddings
def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# trans_model = VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre').to(device) # this checkpoint has 1024 dimensional visual embeddings projection
trans_model = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre').to(device)
pool_layer = nn.MaxPool2d((200, 1))

#TODO: Batch Input Processing for VisualBERT
choice = 1 # 0 for target and 1 for source
for i in range(0, len(img_cv2_source), 32):
    print(i)
    start_range = i
    end_range = i +32
    if i+32>= len(img_cv2_source):
        end_range = len(img_cv2_source)
    source_images, source_batched_inputs = prepare_image_inputs(cfg, img_cv2_source[start_range:end_range])
    source_features = get_features(model, source_images)
    source_proposals = get_proposals(model, source_images, source_features)
    source_box_features, source_features_list = get_box_features(model, source_features, source_proposals)
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, source_features_list, source_proposals)
    source_boxes, source_scores, source_image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, source_proposals)
    output_boxes = [get_output_boxes(source_boxes[i], source_batched_inputs[i], source_proposals[i].image_size) for i in range(len(source_proposals))]
    temp = [select_boxes(cfg, output_boxes[i], source_scores[i]) for i in range(len(source_scores))]
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
    visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(source_box_features, keep_boxes)]
    # Brining visual BERT
    text_batch = txt_data[start_range:end_range,choice].tolist()
    tokens = tokenizer(text_batch, truncation=True, padding='max_length', max_length=100)
    input_ids = torch.tensor(tokens["input_ids"]).to(device)
    attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
    token_type_ids = torch.tensor(tokens["token_type_ids"]).to(device)
    visual_embeds = torch.stack(visual_embeds).to(device)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
    outputs = trans_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), token_type_ids=token_type_ids.to(device), visual_embeds=visual_embeds.to(device), visual_attention_mask=visual_attention_mask.to(device), visual_token_type_ids=visual_token_type_ids.to(device))
    # out_pool = outputs['hidden_states'][0][:,0].cpu().detach().numpy()
    out_pool = pool_layer(outputs['last_hidden_state']).squeeze(1).cpu().detach().numpy()
    if i == 0:
        output_array = np.array(out_pool)
        # new_shape = output_array.shape[0]*output_array.shape[1], output_array.shape[2]
        # output_array = output_array.reshape(new_shape)
        print(output_array.shape)
        np.save('../data/source_mm_vbert.npy', output_array)
    else:
        pre_array = np.load('../data/source_mm_vbert.npy')
        output_array = np.array(out_pool)
        concat_output_array = np.concatenate((pre_array, output_array), axis=0)
        print(concat_output_array.shape)
        np.save('../data/source_mm_vbert.npy', concat_output_array)
