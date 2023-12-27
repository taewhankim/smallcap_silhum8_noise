from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import os
import bisect
import numpy as np
import clip
import math
from tqdm import tqdm
CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25, args=None):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length

        if args.use_ret_img:
            self.ret_img_h5py = h5py.File(args.ret_img_h5py, 'r')
        self.args = args

        if args.use_template_feat:
            self.template_feat = h5py.File(args.template_feat, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k

        self.rag = rag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)

        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        ### 수정: template noise
        # load precomputed features
        if self.args.noise_projection:
            ###수정
            # template_feat = torch.tensor(self.template_feat[self.df['cocoid'][idx]][()],device=self.device)
            # template_noise_feat = clip_noise(template_feat = template_feat,
            #                        variance=self.args.noise_variance,device=self.device).to("cpu")
            # encoder_outputs = torch.concat((torch.tensor(encoder_outputs).unsqueeze(0),template_noise_feat.unsqueeze(0)), dim=0)
            template_feat = torch.tensor(self.template_feat[self.df['cocoid'][idx]][()])
            encoder_outputs = torch.concat((torch.tensor(encoder_outputs).unsqueeze(0),template_feat.unsqueeze(0)), dim=0)
        else:
            pass

        ## 수정 2
        if self.args.use_ret_img:
            ret = ret_enc(caps, self.ret_img_h5py)
            ret_output = np.array(ret)
            ###
            encoding = {"encoder_outputs": torch.tensor(encoder_outputs),
                        "decoder_input_ids": torch.tensor(decoder_input_ids),
                        "labels": torch.tensor(labels),
                        "ret_outputs": torch.tensor(ret_output),
                        "template_noise_feat":template_noise_feat
                        }
        else:
            encoding = {"encoder_outputs": encoder_outputs, # encoder_outputs,
                        "decoder_input_ids": torch.tensor(decoder_input_ids),
                        "labels": torch.tensor(labels),
                        }

        return encoding


def load_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])] ### image와 관련있는 retr cap 7개
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 

def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)

    return data      


def ret_enc(data,h5py_path):
    ret = []
    for cap in data:
        ret.append(h5py_path[cap][()][0])
    return ret


def clip_noise(template_feat=None,variance=0.0, device="cuda"):
    text_features = noise_injection(template_feat, variance=variance, device=device)
    return text_features

def clip_prefix(template=None, tokenizer = None, caps=None, k=None, device=None):
    input_ids_list = []
    for retrieved_caps in caps:
        if retrieved_caps is not None:
            infix = '\n\n'.join(retrieved_caps[:k]) + '.'
            prefix = template.replace('||', infix)
        else:
            prefix = SIMPLE_PREFIX
    # prefix_input_ids = tokenizer(prefix).to(device)
    # return prefix_input_ids
        input_ids_list.append(tokenizer(prefix).to(device))
    return torch.stack(input_ids_list,dim=0).squeeze(1)



def noise_injection(x, variance = 0.001, device = 'cuda') -> torch.Tensor:
    """
    Args:
        x: tensor with a shape of (batch_size, clip_hidden_size), prefix
        variance: the variance of noise
    Return:
        prefix with noise
    """
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    # normalization
    # x = torch.nn.functional.normalize(x, dim = -1)
    # adding noise
    x = x + (torch.randn(x.shape, device = device) * std)

    return torch.nn.functional.normalize(x, dim = -1)