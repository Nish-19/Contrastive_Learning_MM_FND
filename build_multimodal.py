import os
import sys
import numpy as np
from numpy import asarray,zeros
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import timm

device = torch.device("cpu") # Force CPU
print("Using device", device)

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
img_data_reshape = np.reshape(img_data, newshape=(num_images, sources, num_channels, width, height))
img_data_target = torch.tensor(img_data_reshape[:,0,:,:,:]) # Don't convert to GPU
img_data_source = torch.tensor(img_data_reshape[:,1,:,:,:]) # Don't convert to GPU
print('New Target Shape', img_data_target.shape)
print('New Source Shape', img_data_source.shape)

# Utility Models

# Vision Model
class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

# Vision Model
class ViTBottom(nn.Module):
    def __init__(self, original_model):
        super(ViTBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

# Text Model
class BERTModel(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(BERTModel, self).__init__()
        self.model_name = bert_model
        #  Instantiating BERT-based model object
        self.config = AutoConfig.from_pretrained(bert_model, output_hidden_states=False)
        self.bert_layer = AutoModel.from_pretrained(bert_model, config = self.config)
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''
        hidden_state  = self.bert_layer(input_ids, attn_masks, token_type_ids)
        pooler_output = hidden_state[0][:,0]

        return pooler_output

# Text Tokenizer
def get_transformer_model(modelname):
    trans_tokenizer = AutoTokenizer.from_pretrained(modelname, do_lower_case = True)
    print(trans_tokenizer)
    return trans_tokenizer

################ Tokenizer ####################
###############################################
def tokenize(model_name, data_list, tokenizer, MAX_LEN):
    print('Tokenizing')
    # add special tokens for BERT to work properly
    if model_name == 'bert-base-uncased':
        sentences = ["[CLS] " + data_list[i] + " [SEP]" for i in range(0,len(data_list))]
    elif model_name == 'roberta-base':
        sentences = ["<s> " + data_list[i] + " </s>" for i in range(0,len(data_list))]	
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # print ("Tokenize the first sentence:")
    # print (tokenized_texts[0])
    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # Finally convert this into torch tensors
    data_inputs = torch.tensor(input_ids)
    data_masks = torch.tensor(attention_masks)
    return data_inputs, data_masks

# Prepare text data
trans_model_name = 'bert-base-uncased' # for BERT
# trans_model_name = 'roberta-base' # for RoBERTa
trans_tokenizer = get_transformer_model(trans_model_name)
MAX_LEN = 200
# source_text_inputs, source_text_masks = tokenize(trans_model_name, txt_data[:,1], trans_tokenizer, MAX_LEN) # Data on CPU Need to convert to GPU
# target_text_inputs, target_text_masks = tokenize(trans_model_name, txt_data[:,0], trans_tokenizer, MAX_LEN) # Data on CPU Need to convert to GPU
source_text_inputs, source_text_masks = np.load("../data/tokenized/source_text_bert.npy"), np.load("../data/tokenized/source_mask_bert.npy")
target_text_inputs, target_text_masks = np.load("../data/tokenized/target_text_bert.npy"), np.load("../data/tokenized/target_mask_bert.npy")
source_text_inputs, source_text_masks = torch.tensor(source_text_inputs, device=device), torch.tensor(source_text_masks, device=device)
target_text_inputs, target_text_masks = torch.tensor(target_text_inputs, device=device), torch.tensor(target_text_masks, device=device)

#TODO: Multimodal (Image+Text) model
class MultimodalHead(nn.Module):
    def __init__(self):
        super(MultimodalHead, self).__init__()
        self.vision_base_model = timm.create_model('resnet18', pretrained=True)
        self.vision_model_head = ResNetBottom(self.vision_base_model)
        self.text_head = BERTModel('bert-base-uncased')
        self.normalize = nn.LayerNorm(1280)
    def forward(self, img_features, txt_features):
        with torch.no_grad():
            img_out = self.vision_model_head(img_features)
            txt_out = self.text_head(txt_features[0], txt_features[1], token_type_ids=None)
            multimodal_concat = self.normalize(torch.cat((img_out, txt_out), 1))
        return multimodal_concat

#TODO: Multimodal (Image+Text) model
class MultimodalHeadVit(nn.Module):
    def __init__(self, text_trans_name):
        super(MultimodalHeadVit, self).__init__()
        self.pretrained_v = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vision_model_head = ViTBottom(self.pretrained_v)
        self.text_head = BERTModel(text_trans_name)
    def forward(self, img_features, txt_features):
        with torch.no_grad():
            img_out = self.vision_model_head(img_features) # not working - dimension 3
            maxpool = nn.MaxPool2d((img_out.shape[1], 1))
            img_pooled_out = maxpool(img_out).squeeze(1) # 768 dimension
            # print(img_pooled_out.shape)
            txt_out = self.text_head(txt_features[0], txt_features[1], token_type_ids=None)
            multimodal_concat = F.normalize(torch.cat((img_pooled_out, txt_out), 1), dim=1)
        return multimodal_concat

# Create Multimodal model object
# multimodal_model = MultimodalHeadResnet().to(device) # For Resnet model
multimodal_model = MultimodalHeadVit(trans_model_name).to(device)
#TODO: Multimodal forward pass on the entire dataset (USE CPU)
for i in range(0, len(img_data_source), 32):
    print(i)
    start_range = i
    end_range = i + 32
    if i+32>= len(img_data_source):
        end_range = len(img_data_source)
    source_image_input = img_data_source[start_range:end_range,:,:,:].to(device)
    target_image_input = img_data_target[start_range:end_range,:,:,:].to(device)
    source_text_input, source_text_mask = source_text_inputs[start_range:end_range,:].to(device), source_text_masks[start_range:end_range,:].to(device)
    target_text_input, target_text_mask = target_text_inputs[start_range:end_range,:].to(device), target_text_masks[start_range:end_range,:].to(device)
    target_multimodal_out = multimodal_model(target_image_input, (target_text_input, target_text_mask))
    source_multimodal_out = multimodal_model(source_image_input, (source_text_input, source_text_mask))
    if i==0:
        source_all_out = source_multimodal_out
        target_all_out = target_multimodal_out
    else:
        source_all_out = torch.cat((source_all_out, source_multimodal_out), 0)
        target_all_out = torch.cat((target_all_out, target_multimodal_out), 0)
    print(source_all_out.shape)
    print(target_all_out.shape)

print(source_all_out.shape)
print(target_all_out.shape)

# Saving multimodal features
source_all_out = source_all_out.detach().numpy()
target_all_out = target_all_out.detach().numpy()
np.save('../data/source_multimodal_out_vit_bert.npy', source_all_out)
np.save('../data/target_multimodal_out_vit_bert.npy', target_all_out)