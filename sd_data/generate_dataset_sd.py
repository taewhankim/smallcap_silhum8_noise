import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import torch
from accelerate import PartialState
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers import AutoencoderKL
import sys
[sys.path.append(i) for i in [".",".."]]
from sd_data.sd_dataset import create_loader
import time
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


parts = "train"
output_dir = "/data/twkim/coco_sd"
# output_image = os.path.join(output_dir,"images",parts)
output_image = os.path.join(output_dir,"images")
os.makedirs(output_image, exist_ok=True)

batch_size = 256

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler",use_karras_sigmas=True, algorithm_type = "sde-dpmsolver++")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipe.vae = vae
#compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)


pipe = pipe.to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

def get_inputs(prompt,negative_prompt=None, batch_size=None):
  generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
  num_inference_steps = 50

  return {"prompt": prompt,
          "generator": generator,
          "negative_prompt": negative_prompt,
          "num_inference_steps": num_inference_steps}

# (masterpiece)), (top quality), (best quality)
# prompt = ["a horse with short legs standing in water",
#           "a cat against the wall",
#           "an astronaut riding a horse",
#           "a scientist sitting on at table and working at his office",
#           "(Cinematic simple scenery:1.5), (blurry background:1.5), promotional photoshoot, promotional render, raw photo, (surrealistic, intricated detail:1.2), (best quality, masterpiece:1.2), ultra detailed, (photo realistic:1.4), (A stylish korean actress with a perfect body:1.3), (pureerosface_v1:0.7), ((black hair)), (classy updo hair:1.3), (huge breasts and black pantyhose under black simple business suit under black chesterfield coat wrapped tightly around the body:1.4), (narrow waist:1.5), (slim thighs:1.5), (skinny legs:1.5), (skinny arms:1.5) (super-model body:1.3), (long legs, long arms:1.5), (closed eyes:0.7), (happy smile:1.1), (five fingers on one hand, beautiful fingers:1.3), (two legs, perfect legs, beautiful legs:1.3)",
#           "1girl, (Nadia Hilker):1.5, Champagne color Gatsby 1920s Flapper Dress with Sequins and Fringes, South of France, Monaco, Monte Carlo, dynamic angle, seductive grin, smokey eye shadow, high detail skin, high detail eyes, seductive eyes, smokey makeup, slender body, toned body, perfect face, slim athletic body, (perky small breasts:0. 75), detailed clothing, intricate clothing, seductive pose, action pose, motion, casting pose,((masterpiece)), ((best quality)), extremely detailed cg 8k wallpaper, bright colors, Dramatic light, photoreal full body (wide-angle lens, Panoramic:1. 2),fantasy, hyper-realistic, amazing fine detail, rich colors, realistic texture, vignette, moody, dark, epic, gorgeous, film grain, grainy, beautiful lighting, rim lighting, magical, shallow depth of field, photography, neo noir, volumetric lighting, Ultra HD, raytracing, studio quality, octane render, <lora:add_detail:1>, <lora:paulnong-04:0. 6>",
#           "Beautiful pakistani girl, wearing high-quality modern dress, on a balcony with a beautiful nightscape background, beautiful smile on her face, short black hair, tall height, looking into the camera, very fair skin tone, photo captured with Canon EOS R5 camera with an 85mm f/1.4 prime lens",
#           "a battered clock hanging from the side of an abandoned bank building",
#           "a building with a clock and statues of men"
#           ]

# detail_prompt = "8k, highres, masterpiece, ultra detailed, ultra quality, photo captured with Canon EOS R5 camera with an 85mm f/1.4 prime lens"
#detail_prompt = "8k wallpaper, highres, masterpiece, ultra quality, perfect ligthing, physically-based rendering, photo taken by Canon EOS R5 camera with an 85mm prime lens, "
# detail_prompt = "8k dlsr, 8k textures, highres, masterpiece, perfect ligthing, a realistic representation, "
# detail_prompt = "8k dlsr, highres, masterpiece, photoism, perfect ligthing, "
# detail_prompt = "8k dlsr, highres, masterpiece, photoism, perfect ligthing, "


#
# prompt = [detail_prompt+i for i in prompt]
# ultra quality, physically-based rendering, a realistic representation,
## 얘도 나쁘지않음
# negative_prompt = ["EasyNegative, disfigured, low quality, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"]

# negative_prompt = ["lowres, low quality, artifacts, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, drawing, painting, crayon, sketch, graphite, impressionist, noisy, soft, blurry, 3d, render, sketch, cartoon, drawing, anime, overexposed, photoshop"]
## 아래놈 얘가 1
# negative_prompt = ["lowres, low quality, artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, painting, sketch, graphite, impressionist, noisy, soft, blurry, 3d, render, sketch, cartoon, drawing, anime, overexposed"]
# negative_prompt = ["out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehy drated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"]
## 아래놈 일단 2
# negative_prompt = ["EasyNegative, error, cropped, worst quality, low quality, out of frame, mutated hands and fingers, extra fingers, extra hands, extra legs, disconnected hands, disconnected limbs, amputation, semi-realistic, cgi, blurry, 3d, render, sketch, cartoon, drawing, anime, doll, overexposed, photoshop"]
# negative_prompt = ["ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo,cgi, blurry, 3d, render, sketch, cartoon, drawing, anime, doll, overexposed, photoshop"]
# negative_prompt = [""]
# negative_prompt *= len(prompt)
# for idx,i in enumerate(prompt):
#     prompt[idx] =prompt[idx]+ " hard rim lighting photography--beta --ar 2:3 --beta --upbeta"

# prompt_compel = compel(prompt)

# t1 = time.time()
# images = pipe(**get_inputs(prompt,negative_prompt=negative_prompt,batch_size=9)).images # ,guidance_scale=10
# for idx,i in enumerate(images):
#     i.save(os.path.join(output_image,"{}.png".format(idx)))
# t2 = time.time()
# print(t2-t1)


re_cap = json.load(open("/home/twkim/project/smallcap/datastore/coco_index_captions.json"))

# X_train, X_test= train_test_split(re_cap, test_size=0.3, random_state=1)
#
# X_train, X_val = train_test_split(X_train, test_size=0.5, random_state=1)

# train_dataloader, val_dataloader, test_dataloader = create_loader([X_train,X_val, X_test], batch_size, 8)
train_dataloader= create_loader([re_cap], batch_size, 8)

sd_json = {}
cnt = 0
'''
set loader@@@@@
train : 62003
val : 62003
test: 53146
'''
ori_cap = re_cap
negative_prompt = [
    "lowres, low quality, artifacts, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, drawing, painting, crayon, sketch, graphite, impressionist, noisy, soft, blurry, 3d, render, sketch, cartoon, drawing, anime, overexposed, photoshop"]
# negative_prompt = ["EasyNegative, disfigured, low quality, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"]

for idx, batch in enumerate(tqdm(train_dataloader,total = len(train_dataloader),leave = True)):
    # output = {i:"sd_{0:06d}".format(cnt) for i in ori_cap}
    images = pipe(**get_inputs(batch,negative_prompt=negative_prompt*batch_size, batch_size=batch_size)).images
    for i in images:
        sd_json[ori_cap[cnt]]="sd_{0:06d}".format(cnt)
        i.save(os.path.join(output_image, "sd_{0:06d}.png".format(cnt)))
        cnt+=1

# json.dump(output, open('/home/twkim/project/smallcap/datastore/sd_captions_{}.json'.format(parts), 'w')) ## 모든 캡션
json.dump(sd_json, open('/home/twkim/project/smallcap/datastore/sd_captions.json', 'w')) ## 모든 캡션

