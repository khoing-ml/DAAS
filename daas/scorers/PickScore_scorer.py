import os
import torch
from PIL import Image
from transformers import AutoModel, CLIPProcessor


def _as_pil_images(images):
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, torch.Tensor):
        tensor = images.detach().cpu()
        if tensor.ndim != 4:
            raise ValueError("images tensor must have shape (batch, channels, height, width)")
        if tensor.dtype.is_floating_point:
            if tensor.min() < 0:
                tensor = ((tensor / 2) + 0.5).clamp(0, 1)
            tensor = (tensor * 255).round().clamp(0, 255).to(torch.uint8)
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return [Image.fromarray(sample.numpy()) for sample in tensor]
    return [image if isinstance(image, Image.Image) else Image.fromarray(image) for image in images]


class PickScoreScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

    def __call__(self, images, prompts):
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_embeds = self.model.get_text_features(**text_inputs)
        # Extract tensor if model returns a BaseModelOutputWithPooling object
        if hasattr(text_embeds, 'pooler_output'):
            text_embeds = text_embeds.pooler_output
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

        pil_images = _as_pil_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
        image_embeds = self.model.get_image_features(pixel_values=inputs)
        # Extract tensor if model returns a BaseModelOutputWithPooling object
        if hasattr(image_embeds, 'pooler_output'):
            image_embeds = image_embeds.pooler_output
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores
