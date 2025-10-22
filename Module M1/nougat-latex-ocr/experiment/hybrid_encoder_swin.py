import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from transformers import VisionEncoderDecoderModel

class HybridResNetSwinEncoder(nn.Module):
    def __init__(self, pretrained_swin_name: str = "Norm/nougat-latex-base"):
        super().__init__()

        # Swin encoder from HuggingFace
        nougat_swin_model = VisionEncoderDecoderModel.from_pretrained(pretrained_swin_name)
        self.swin_encoder = nougat_swin_model.encoder
        self.config = self.swin_encoder.config
        hidden_size = self.config.hidden_size  # e.g., 768 or 1024

        # ResNet-101 as global image summarizer
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Identity()

        # Project ResNet output to match Swin hidden size
        self.proj = nn.Linear(2048, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.zeros(1))  # starts at 0 (no fusion at start)

        self.main_input_name = "pixel_values"

    def forward(self, pixel_values, **kwargs):
        # Swin forward: get (B, L, hidden_size)
        swin_outputs = self.swin_encoder(pixel_values, **kwargs)
        swin_hidden = swin_outputs.last_hidden_state

        # ResNet global token: (B, 2048) → (B, hidden_size)
        resnet_feat = self.resnet(pixel_values)
        proj_feat = self.proj(resnet_feat)
        normed_feat = self.norm(proj_feat)
        gated_feat = self.alpha * normed_feat  # shape: (B, hidden_size)

        # Append as a new token
        resnet_token = gated_feat.unsqueeze(1)  # (B, 1, hidden_size)
        hybrid_hidden = torch.cat([resnet_token, swin_hidden], dim=1)  # (B, L+1, hidden_size)

        return type(swin_outputs)(
            last_hidden_state=hybrid_hidden,
            hidden_states=swin_outputs.hidden_states,
            attentions=swin_outputs.attentions,
        )

    def get_output_embeddings(self):
        return None
