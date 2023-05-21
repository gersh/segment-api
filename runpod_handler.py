import torch
import cv2
import runpod
import urllib
import numpy as np
from runpod.serverless.utils import rp_upload

from diffusers import StableDiffusionPipeline

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint="/src/sam_vit_h_4b8939.pth")
sam.to(device='cuda')
mask_generator = SamAutomaticMaskGenerator(sam)


def handler(job):
    url_response = urllib.request.urlopen(job['input']['url'])
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    json_masks = [list(x['bbox']) for x in masks]
    return {'url':job['input']['url'],"masks": json_masks}


runpod.serverless.start({"handler": handler})