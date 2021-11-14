import numpy as np
import os
import pandas as pd 

#TODO: Load 300d pre-trained Glove embeddings
# Loading the pre-trained Glove embeddings
print('Extracting embeddings')
embeddings_dict = {}
with open("./glove.840B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            vector = np.asarray(values[1:], "float32")
        except:
            continue
        embeddings_dict[word] = vector
print('Finished extracting embeddings')

#TODO: Load data
text_data = np.load('../data/text_array.npy')
print(text_data.shape)
target_text_arr = text_data[:,0]
source_text_arr = text_data[:,1]
print(target_text_arr.shape)
print(source_text_arr.shape)
# Loading image2sen data
image2sen = pd.read_csv('./Show_and_Tell/test/results/our_captions.csv')['caption'].to_list()
print(len(image2sen))

#TODO: Get sentence embeddings
head_size = 30
body_size = 100
image_size = 20
zeros_vec = np.zeros(shape=(300,))
head_lst = []
body_lst = []
img_lst = [] 
for i in range(len(image2sen)):
    text_split = target_text_arr[i].split(' ')
    text_embed_lst = []
    for j, text in enumerate(text_split):
        if j >= head_size:
            break
        try:
            text_embed_lst.append(embeddings_dict[text])
        except:
            text_embed_lst.append(zeros_vec)
    if len(text_split) < head_size:
        for j in range(head_size - len(text_split)):
            text_embed_lst.append(zeros_vec)
    # NOTE: Body data
    src_text_split = source_text_arr[i].split(' ')
    src_text_embed_lst = []
    for j, text in enumerate(src_text_split):
        if j >= body_size:
            break
        try:
            src_text_embed_lst.append(embeddings_dict[text])
        except:
            src_text_embed_lst.append(zeros_vec)
    if len(src_text_split) < body_size:
        for j in range(body_size - len(src_text_split)):
            src_text_embed_lst.append(zeros_vec)
    # NOTE: Image Data
    img_split = image2sen[i].split(' ')
    img_embed_lst = []
    for j, text in enumerate(img_split):
        if j >= image_size:
            break
        try:
            img_embed_lst.append(embeddings_dict[text])
        except:
            img_embed_lst.append(zeros_vec)
    if len(img_split) < image_size:
        for j in range(image_size - len(img_split)):
            img_embed_lst.append(zeros_vec)
    head_lst.append(text_embed_lst)
    body_lst.append(src_text_embed_lst)
    img_lst.append(img_embed_lst)
head_vec = np.asarray(head_lst, dtype=np.float32)
img_vec = np.asarray(img_lst, dtype=np.float32)
body_vec = np.asarray(body_lst, dtype=np.float32)

print(head_vec.shape)
print(img_vec.shape)
print(body_vec.shape)

#TODO: Save
np.save('sent_data/head_arr', head_vec)
np.save('sent_data/img_arr', img_vec)
np.save('sent_data/body_arr', body_vec)