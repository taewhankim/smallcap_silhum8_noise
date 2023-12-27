import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import argparse
import random
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
import numpy as np
import clip
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoProcessor, CLIPModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from src.utils import load_data_for_inference, prep_strings, postprocess_preds, ret_enc,clip_prefix, clip_noise

ImageFile.LOAD_TRUNCATED_IMAGES = True

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size

    for idx in tqdm(range(0, len(eval_df), bs)):
        file_names = eval_df['file_name'][idx:idx + bs]
        image_ids = eval_df['image_id'][idx:idx + bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test=True) for _ in range(len(image_ids))]

        # load image
        images = [Image.open(args.images_dir + file_name).convert("RGB") for file_name in file_names]
        pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
        with torch.no_grad():
            preds = model.generate(pixel_values.to(args.device),
                                   decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                                   **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)

        for image_id, pred in zip(image_ids, preds):
            pred = postprocess_preds(pred, tokenizer)
            out.append({"image_id": int(image_id), "caption": pred})

    return out


def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""
    template = open(args.template_path).read().strip() + ' '
    torch.cuda.empty_cache()
    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')
    if args.use_ret_img:
        ret_img_feat = h5py.File(args.ret_img_h5py, 'r')
    # if args.template_feat:
    #     template_feat = h5py.File(args.template_feat, 'r')


    out = []

    ### 미리 뽑기
    bs = 1000
    template_pre_feat = []
    for idx in tqdm(range(0, len(eval_df), bs)):
        caps = eval_df['caps'][idx:idx + bs]
        with torch.no_grad():
            temp_input_ids = clip_prefix(template=template, tokenizer=clip.tokenize, caps=caps, k=3,
                                         device=args.device)
            # template_noise_batch = clip_noise(template_feat=text_model.encode_text(temp_input_ids),
            #                                   variance=args.noise_variance, device=args.device)
            # template_pre_feat.append(template_noise_batch)
            template_pre_feat.append(text_model.encode_text(temp_input_ids))

    template_pre_feat = torch.concat(template_pre_feat)
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx]
        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                         k=int(args.k), is_test=True)
        ## 수정
        model.ret_value = None
        if args.use_ret_img:
            ret = ret_enc(caps[:args.k], ret_img_feat)
            ret_output = np.array(ret)
            model.ret_value = torch.tensor(ret_output).to(args.device)

        ## 수정 noise
        model.template_noise = None
        if args.template_feat:
            # template_feat = torch.tensor(template_feat[image_id][()], device=args.device)
            # with torch.no_grad():
            #     temp_input_ids = clip_prefix(template=template, tokenizer = clip.tokenize, retrieved_caps=caps, k=3, device=args.device)
            #     template_feat = text_model.encode_text(temp_input_ids)
            #
            # template_noise_feat = clip_noise(template_feat=template_feat,
            #                                   variance=args.noise_variance, device=args.device)
            template_noise_feat = template_pre_feat[idx].unsqueeze(0)
            if not args.noise_projection:
                pass
                ## 아직 코드 안짬
            else:
                model.template_noise= template_noise_feat

        # load image
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      **args.generation_kwargs)
        else:
            # 수정
            image = Image.open(args.images_dir + file_name).convert("RGB")
            pixel_values = feature_extractor(text=None, images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                pred = model.generate(pixel_values.to(args.device),
                                      decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                                      **args.generation_kwargs)

        pred = tokenizer.decode(pred[0])

        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    return out


def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model


def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)


def register_model_and_config():
    from transformers import AutoModelForCausalLM
    if args.w == "1":
        from src.vision_encoder_decoder_1 import SmallCap, SmallCapConfig
        print("#----- weight 1")
    if args.w == "2":
        from src.vision_encoder_decoder_2 import SmallCap, SmallCapConfig
        print("#----- weight 2")
    if args.w == "4":
        from src.vision_encoder_decoder_4 import SmallCap, SmallCapConfig
        print("#----- weight 4")
    if args.w == "8":
        from src.vision_encoder_decoder_8 import SmallCap, SmallCapConfig
        print("#----- weight 8")
    if args.w == "0":
        from src.vision_encoder_decoder_512 import SmallCap, SmallCapConfig
        print("#----- 512*2")


    if args.w == "1_1":
        from src.vision_encoder_decoder_c_loss_1 import SmallCap, SmallCapConfig
        print("#----- clip_loss_1")
    if args.w == "4_1":
        from src.vision_encoder_decoder_c_loss_4 import SmallCap, SmallCapConfig
        print("#----- clip_loss_4")

    if args.w == "noise":
        from src.vision_encoder_decoder_att_loss import SmallCap, SmallCapConfig
        print("#----- no noise norm")
    if args.w == "text":
        from src.vision_encoder_decoder_text_norm import SmallCap, SmallCapConfig
        print("#----- no text norm")
    # if args.w == "8":
    #     from src.vision_encoder_decoder_8 import SmallCap, SmallCapConfig
    #     print("#----- weight 8")

    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)


def main(args):
    set_seed(1)

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.infer_test or args.disable_rag:
        args.features_path = None

    if args.features_path is not None:
        feature_extractor = None
    else:
        ## 수정
        # feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
        global text_model
        text_model, _ = clip.load("ViT-B/32", device=args.device)
        feature_extractor = AutoProcessor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k = 0
        infer_fn = evaluate_norag_model
    else:
        infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'val'

    data = load_data_for_inference(args.annotations_path, args.captions_path)

    eval_df = pd.DataFrame(data[split])
    args.outfile_name = '{}_preds.json'.format(split)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    # configure generation
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="/data/twkim/coco_images/",
                        help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default='/data/twkim/smallcap/features/val.hdf5',
                        help="H5 file with cached input image features")
    parser.add_argument("--annotations_path", type=str, default="/data/twkim/smallcap/data/dataset_coco.json",
                        help="JSON file with annotations in Karpathy splits")

    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False,
                        help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix") ### 이후에 수정 3으로
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="/data/twkim/smallcap/data/retrieved_caps_resnet50x64.json",
                        help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size; only matter if evaluating a norag model")
    parser.add_argument("--use_ret_img", action="store_true", default=False, help="Directory where trained models will be saved")
    parser.add_argument("--ret_img_h5py", type=str, default="/data/twkim/smallcap/features/ret_enc_cls.hdf5", help="Directory where trained models will be saved")

    #### noise
    parser.add_argument("--use_template_feat", action="store_true", default=True, help="Directory where trained models will be saved")
    parser.add_argument("--template_feat", type=str, default="/data/twkim/smallcap/features/train_template.hdf5", help="Directory where trained models will be saved")

    parser.add_argument('--noise_projection', action="store_true", default=True, help = 'noise variance')
    parser.add_argument('--noise_variance', type = float, default = 0.016, help = 'noise variance')

    parser.add_argument("--w", type=str, default=4, help="s to use in prefix")

    args = parser.parse_args()

    main(args)