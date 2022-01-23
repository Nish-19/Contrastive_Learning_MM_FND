import os
os.environ['CUDA_ENVIRONMENT_DEVICES'] = "0"

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the data
target_data = np.load('saved_models/target_bert.npy')
source_data = np.load('saved_models/source_bert.npy')
# Printing the shapes
print('Tokenized Target Shape', target_data.shape)
print('Tokenized Source Shape', source_data.shape)

# Text Model
class BERTModel(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(BERTModel, self).__init__()
        #  Instantiating BERT-based model object
        self.config = AutoConfig.from_pretrained(bert_model, output_hidden_states=False)
        self.bert_layer = AutoModel.from_pretrained(bert_model, config = self.config)
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
    def forward(self, input_ids, attn_masks):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''
        hidden_state  = self.bert_layer(input_ids, attn_masks, token_type_ids=None)
        pooler_output = hidden_state[0][:,0]

        return pooler_output

# Get model
model = BERTModel().to(device)

for i in range(0, len(source_data), 16):
    print(i)
    start_range = i
    end_range = i + 16
    if i+16>= len(source_data):
        end_range = len(source_data)
    with torch.no_grad():
        output_tensor = model(torch.tensor(source_data[start_range:end_range,0,:], device=device), torch.tensor(source_data[start_range:end_range,1,:], device=device))
    output_tensor.detach().cpu()
    if i==0:
        all_out = output_tensor
    else:
        all_out = torch.cat((all_out, output_tensor), 0)
    del output_tensor
    torch.cuda.empty_cache()

print(all_out.shape)
all_out_np = all_out.detach().cpu().numpy()
np.save('../data/bert_text_only_source.npy', all_out_np)


