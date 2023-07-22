from rich import print
import clip
import torch
import pyrootutils
import os
import argparse
import pandas as pd
import copy
from tqdm import tqdm
import sparse
import numpy as np
from torchvision.ops import roi_align
from pathlib import Path
import shutil

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


ATTRIBUTES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "rectangle",
    "circle",
    "small",
    "medium",
    "large",
]


def generate_attribute_embs(out_dir):
    """
    Generate embeddings for each attribute.

    Parameters:
        out_dir: Directory for storing attribute embeddings
    """

    def get_prompts(attr):
        if attr in [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]:
            prompts = [
                f"the image shows a {attr}",
                f"the digit appears to be {attr}",
                f"there is an image showing a {attr}",
                f"the number is a {attr}",
            ]
        elif attr in ["red", "yellow", "green", "blue", "purple"]:
            prompts = [
                f"the color is {attr}",
                f"the digit appears to be {attr}",
                f"there is a {attr} image",
                f"the image is {attr}",
            ]
        elif attr in ["rectangle", "circle"]:
            prompts = [
                f"the shape is a {attr}",
                f"the shape appears to be a {attr}",
                f"there is a {attr}",
                f"the image has a {attr}",
            ]
        elif attr in ["small", "medium", "large"]:
            prompts = [
                f"the shape size is {attr}",
                f"the size of the shape is {attr}",
                f"the shape is {attr}",
            ]
        return prompts

    model, preprocess = clip.load("RN50", "cuda")

    # Obtain prompts for each attribute
    prompt_list = []
    for attr in ATTRIBUTES:
        prompt_list.append(get_prompts(attr))

    # Compute each attribute embedding as the average of its associated prompt embeddings
    attr_embs = []
    with torch.no_grad():
        for prompt in prompt_list:
            text = clip.tokenize(prompt, truncate=True).to("cuda")
            txt_emb = model.encode_text(text).detach().cpu()
            txt_emb = txt_emb.mean(dim=0, keepdim=True)
            txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
            attr_embs.append(txt_emb)
        attr_embs = torch.stack(attr_embs).squeeze().detach().cpu().numpy()

    attr_to_emb = dict(zip(ATTRIBUTES, attr_embs))

    torch.save(attr_to_emb, f"{out_dir}/attr_embs.pth")
    print(f"Saved {len(attr_to_emb)} attribute embeddings to {out_dir}/attr_embs.pth")


def generate_region_embs(out_dir):
    """
    Generate embeddings for each region.

    Parameters:
        out_dir: Directory for storing region embeddings
    """
    ann = pd.read_feather(f"{out_dir}/annotations.feather")
    components = {}
    out_dir = Path(out_dir) / f"region_embs"
    if out_dir.exists() and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Load CLIP vision encoder
    model, _ = clip.load("RN50", "cuda")
    model_backbone, _ = clip.load("RN50", "cuda")
    components["model_backbone"] = model_backbone.to(torch.float32)
    for c in ["attnpool", "layer4"]:
        components[c] = copy.deepcopy(eval(f"model_backbone.visual.{c}"))
        setattr(model_backbone.visual, c, torch.nn.Identity())
    for c in ["transformer", "token_embedding", "ln_final"]:
        setattr(model_backbone, c, torch.nn.Identity())

    # Generate embeddings for each region
    reg_emb_map = {"image_id": [], "file": [], "file_id": []}
    all_reg_embs = []
    for idx, row in tqdm(ann.iterrows()):
        image_id, filepath = row["image_id"], row["image_filepath"]
        image = torch.tensor(
            (sparse.load_npz(filepath).todense() / 255).astype(np.float32)
        )
        image = torch.stack([image]).cuda()
        regions = np.stack(row["region_coord"].tolist())

        with torch.no_grad():
            features = components["model_backbone"].encode_image(image)
            rois = (
                torch.cat((torch.zeros((len(regions), 1)), torch.tensor(regions)), 1)
                .to(torch.float32)
                .cuda()
            )
            x = roi_align(
                features,
                rois.to(dtype=features.dtype),
                (14, 14),
                features.shape[-1] / image.shape[-1],
                0,
                True,
            )
            x = components["layer4"](x)
            x = components["attnpool"](x)
            reg_embs = x

        if len(all_reg_embs) == 10000:
            all_reg_embs = np.array(all_reg_embs, dtype=object)
            np.savez_compressed(out_dir / f"embs_{idx}", all_reg_embs)
            reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
            reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))
            all_reg_embs = []
        reg_embs = reg_embs.cpu().numpy()
        all_reg_embs.append(
            reg_embs.reshape(
                -1,
            )
        )
        reg_emb_map["image_id"].append(image_id)

    all_reg_embs = np.array(all_reg_embs, dtype=object)
    np.savez_compressed(out_dir / f"embs_{idx}", all_reg_embs)
    reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
    reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))

    pd.DataFrame(reg_emb_map).to_feather(out_dir / "region_emb_mapping.feather")
    print(f"Saved region embeddings to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing functions for DocMNIST."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Directory where DocMNIST data is stored (e.g. docmnist_30000_15.2)",
    )
    args = parser.parse_args()

    print("Generating attribute embeddings")
    generate_attribute_embs(os.path.join(root, "data", args.data_dir))
    print(f"-----------")

    print("Generating region embeddings")
    generate_region_embs(os.path.join(root, "data", args.data_dir))
    print(f"-----------")


if __name__ == "__main__":
    main()
