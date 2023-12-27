import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import sys
import pandas as pd
import json
from tqdm import tqdm 
from PIL import Image
import torch
from multiprocessing import Pool
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import clip
import sys
[sys.path.append(i) for i in [".",".."]]
from src.utils import load_data_for_training
import numpy as np
logging.set_verbosity_error()

data_dir = '/data/twkim/coco_images/'
features_dir = '/data/twkim/smallcap/features/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

data_ann ="/data/twkim/smallcap/data/dataset_coco.json"
cap_annotations = '/data/twkim/smallcap/data/retrieved_caps_resnet50x64.json'

# data = load_data_for_training(data_ann,cap_annotations)

def load_data():
    data = {'train': [], 'val': []}
    annotations = json.load(open(data_ann))['images']
    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
    return data



def encode_split(data, split):

    df = pd.DataFrame(data[split])

    cap_annotations = '/data/twkim/smallcap/data/retrieved_caps_resnet50x64.json'
    cap_ann = json.load(open(cap_annotations))

    template = open("/home/twkim/project/smallcap_sd/src/template.txt").read().strip() + ' '
    bs = 256
    h5py_file = h5py.File(features_dir + '{}_template.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        coco_bs = df['cocoid'][idx:idx + bs]

        tmp_bs = [clip.tokenize(clip_prefix(template=template, retrieved_caps=cap_ann[str(cb)], k=3)).to(device) for cb in coco_bs]
        prefix_ids = torch.cat(tmp_bs).to(device)
        with torch.no_grad():
            text_features = model.encode_text(prefix_ids).cpu().numpy()

        # re = {str(co):en for co, en in zip(coco_bs, text_features)}
        # result = dict(result,**re)

        for cocoid, encoding in zip(coco_bs, text_features):
            h5py_file.create_dataset(str(cocoid), (512), data=encoding)
    # np.savez(features_dir+"train_tamplate2.npz", **result)

def clip_prefix(template=None, retrieved_caps=None, k=None):

    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX
    return prefix



data = load_data()

encode_split(data, 'train')
# encode_split(data, 'val')
