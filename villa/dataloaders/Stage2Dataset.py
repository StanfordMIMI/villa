import abc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import itertools
import clip
from pathlib import Path


class Stage2Dataset(Dataset):
    __metaclass__ = abc.ABC

    def __init__(self, split, data_dir, stage_1_ckpt_dir):
        """
        Initialize dataloader for Stage 2.

        Parameters:
            split (str): Indicates the split (e.g. "train", "val")
            data_dir (str): Directory where data is stored
            stage_1_ckpt_dir (str): Directory where checkpoints from stage 1 are stored
        """
        super().__init__()

        self.split = split
        self.data_dir = Path(data_dir)
        self.stage_1_ckpt_dir = Path(stage_1_ckpt_dir)

        self.examples = pd.read_feather(
            self.stage_1_ckpt_dir / f"mapping_{split}.feather"
        )

    @abc.abstractmethod
    def create_train_dataset(self):
        return

    def get_text_embs(self):
        """
        Precompute text embeddings.

        Returns:
            text_emb (list): Text embeddings associated with each region
            valid_regions (pd.Series): List of valid regions (i.e. >= 1 assigned attribute)
                                       in each image
        """
        text_split = self.examples["assigned_text"].apply(
            lambda x: [[a.strip() for a in b.split(".") if len(a) > 0] for b in x]
        )
        text = itertools.chain(*list(itertools.chain(*text_split)))
        text = list(set(text))
        sent_to_emb = self.get_clip_text_embs(text)

        text_emb = []
        for t in text_split:
            emb = [
                np.mean(np.stack([sent_to_emb[x] for x in split]), 0)
                for split in t
                if len(split) > 0
            ]
            text_emb.append(np.stack(emb))
        return text_emb, self.examples["assigned_text"].apply(
            lambda x: [k for k in range(len(x)) if len(x[k]) > 0]
        )

    def get_clip_text_embs(self, sents: list, model: str = "RN50"):
        """
        Use CLIP to generate text embeddings.

        Parameters:
            sents (list): List of sentences
            model (str): CLIP model variant

        Returns:
            text_to_emb (dict): Dictionary mapping each element in sents to its
                                corresponding CLIP text embedding.
        """
        model, _ = clip.load("RN50", "cuda")
        text = clip.tokenize(sents, truncate=True).to("cuda")
        with torch.no_grad():
            text_features = model.encode_text(text).detach().cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_to_emb = dict(zip(sents, text_features.tolist()))

        return text_to_emb
