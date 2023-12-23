import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import json
import numpy as np
import torch
import h5py
from PIL import Image
from tqdm import tqdm

from transformers import CLIPFeatureExtractor, CLIPVisionModel

re_cap = json.load(open("/home/twkim/project/smallcap/datastore/sd_captions.json"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name)
clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)

data_dir = "/data/twkim/coco_sd/images"

def encode_split(data, data_dir):
    df = list(data.values())
    id = list(data.keys())
    features_dir = "../features/"
    bs = 128
    result = {}
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format("ret_enc_cls"), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = id[idx:idx + bs]
        file_names = df[idx:idx + bs]
        images = [Image.open(os.path.join(data_dir, file_name+".png")).convert("RGB") for file_name in file_names]
        with torch.no_grad():
            pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
            encodings = clip_encoder(pixel_values=pixel_values).pooler_output.cpu().numpy()
        re = {co:en for co, en in zip(cocoids, encodings)}
        result = dict(result,**re)
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(cocoid, (1, 768), data=encoding)

    np.savez("/home/twkim/project/smallcap_sd/datastore/ret_cap_cls.npz", **result)


encode_split(re_cap,data_dir)



