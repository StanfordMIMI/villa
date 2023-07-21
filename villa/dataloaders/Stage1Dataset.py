import abc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from pathlib import Path


class Stage1Dataset(Dataset):
    __metaclass__ = abc.ABC

    def __init__(self, split, data_dir):
        super().__init__()

        self.split = split
        self.data_dir = Path(data_dir)

        self.ann = pd.read_feather(self.data_dir / "annotations.feather")
        self.examples = self.ann[self.ann["split"] == split]

    def encode_attributes(self, attributes):
        attr_binary_vec = []
        for idx, row in self.examples.iterrows():
            vec = np.zeros(len(attributes))
            for a in row["attributes"]:
                vec[attributes.index(a)] = 1
            attr_binary_vec.append(vec)
        return attr_binary_vec

    def get_region_embs(self):
        region_embs = {}
        print(f"Loading region embeddings from {self.data_dir}/region_embs")
        emb_df = pd.read_feather(
            f"{self.data_dir}/region_embs/region_emb_mapping.feather"
        )
        curr_open_npz = None
        valid_samples = set(self.examples["image_id"].values.tolist())
        for idx, row in tqdm(emb_df.iterrows()):
            image_id = row["image_id"]
            if image_id not in valid_samples:
                continue
            if row["file"] != curr_open_npz:
                curr_open_npz = row["file"]
                embs = np.load(
                    f"{self.data_dir}/region_embs/{curr_open_npz}.npz",
                    allow_pickle=True,
                )["arr_0"]
            region_embs[str(image_id)] = embs[row["file_id"]].reshape(-1, 1024)
        return region_embs

    def getInputs(self, example, image_id):
        out_dict = {}
        reg_emb = self.region_embs[image_id]
        out_dict["num_regions"] = torch.tensor(reg_emb.shape[0])
        out_dict["img"] = torch.tensor(reg_emb)
        out_dict["attr_labels"] = torch.tensor(
            example["attr_binary_vec"].values[0]
        ).unsqueeze(0)
        return out_dict
