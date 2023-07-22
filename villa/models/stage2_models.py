import torch
from torchvision.ops import roi_align
from villa.utils.utils import build_vision_encoder


class Stage2_RN50(torch.nn.Module):
    def __init__(self, emb_dim: int, clip_weights: str):
        """
        Initialize model for ViLLA Stage 2.

        Parameters:
            emb_dim (int): Embedding dimension
            clip_weights (str): Filepath for CLIP RN50 state dict
        """
        super().__init__()
        state_dict = torch.load(clip_weights)
        self.model = build_vision_encoder(state_dict, emb_dim)
        self.model_backbone = torch.nn.Sequential(
            self.model.visual.conv1,
            self.model.visual.bn1,
            self.model.visual.relu1,
            self.model.visual.conv2,
            self.model.visual.bn2,
            self.model.visual.relu2,
            self.model.visual.conv3,
            self.model.visual.bn3,
            self.model.visual.relu3,
            self.model.visual.avgpool,
            self.model.visual.layer1,
            self.model.visual.layer2,
            self.model.visual.layer3,
        )
        self.layer4 = self.model.visual.layer4
        self.attnpool = self.model.visual.attnpool

    def forward(self, sample: dict):
        """
        Run model forward pass for ViLLA Stage 2.

        Parameters:
            sample (dict): Data associated with each sample
        Returns:
            out_dict (dict): Consists of embeddings for each region
        """
        image = sample["image"]
        region_coords = sample["region"]
        num_regions = sample["num_regions"]
        features = self.model_backbone(image)

        idx = torch.repeat_interleave(
            torch.arange(image.shape[0]), num_regions.cpu()
        ).cuda()
        rois = torch.cat((idx.float().unsqueeze(1), region_coords), 1).to(
            dtype=features.dtype
        )
        scale_factor = features.shape[-1] / image.shape[-1]
        x = roi_align(features, rois, (14, 14), scale_factor, 0, True)
        x = self.layer4(x)
        x = self.attnpool(x)
        img_embs = x / x.norm(dim=1, keepdim=True)

        return {"img_emb": img_embs}
