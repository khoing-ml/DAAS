from PIL import Image
import os
from pathlib import Path
import io
import numpy 
import torch 
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image
from importlib import resources
ASSETS_PATH = resources.files("assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize to [0,255]
            images = ((images/2) + 0.5).clamp(0,1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0,2,3,1) # NCHW -> NHWC: PIL expects (H,W,C)
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images] # create in-memory buffers
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95) # save to buffer with JPEG compression
        sizes = [buffer.tell()/1000  for buffer in buffers] # get size in KB
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device) # convert to tensor
        rewards = -1 * loss
        return loss, rewards
    return loss_fn

def clip_score(inference_dtype=None, device=None, return_loss=False):
    from daas.scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(inference_dtype, device)
    scorer.requires_grad_(False)
    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0:
                images = ((images/2) + 0.5).clamp(0,1)
            scores = scorer(images, prompts)
            return scores
        
        return _fn
    
    else:
        def loss_fn(iamges, prompts):
            if images.min() < 0:
                images = ((images/2) + 0.5).clamp(0,1)
            scores = scorer(images, prompts)
            loss = -1 * scores
            return loss, scores
        return loss_fn
    

def aesthetic_score(
  torch_dtype=None,
  device=None,
  aesthetic_target=None,
  grad_scale=0,
  return_loss=False      
):
    from daas.scorers.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(torch_dtype, device)
    scorer.requires_grad_(True)

    if not return_loss:
        def _fn(images, prompts=None):
            if images.min() < 0:
                images = ((images/2) + 0.5).clamp(0,1)
            scores = scorer(images)
            return scores
        
        return _fn
    
    else:
        def loss_fn(images, prompts=None):
            if images.min() < 0:
                images = ((images/2) + 0.5).clamp(0,1)
            scores = scorer(images)
            if aesthetic_target is None:
                loss = -1 * scores
            else:
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores
        return loss_fn


