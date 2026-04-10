from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torchvision

ASSETS_PATH = resources.files("assets")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.Dropout(0.1),
            nn.Linear(64,16),
            nn.Linear(16,1)
        )

    def forward(self, embed):
        return self.layers(embed)
    

class AestheticScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device


        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=self.device) # load the state dict for the MLP
        self.mlp.load_state_dict(state_dict)
        self.target_size = 224 # CLIP's ViT-L/14 model expects 224x224 images
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP normalization 

        self.eval()

    def __call__(self,images): 
        inputs = torchvision.transforms.Resize(self.target_size)(images) # resize images to CLIP's expected input size
        inputs = self.normalize(inputs).to(self.dtype) # normalize and convert to the correct dtype
        embed = self.clip.get_image_features(pixel_values=inputs) # get CLIP image embeddings, shape (N, 768)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)  # normalize embeddings to unit length

        return self.mlp(embed).squeeze(1) # return a single score per image, shape (N,)
    
