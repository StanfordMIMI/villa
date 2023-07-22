import torch
import numpy as np
from tqdm import tqdm
from rich import print
import pandas as pd
import copy


def mapping(model, dataloader, split, one_proj, data_dir, checkpoint_dir, epsilon):
    """
    Compute region-attribute mappings.

    Parameters:
        model (torch.nn.Module): Model
        dataloader (torch.utils.data.DataLoader): Dataloader
        split (str): Data split (e.g. "train", "val")
        one_proj (bool): True if using one projection head, False otherwise
        data_dir (str): Filepath to directory with data
        checkpoint_dir (str): Directory for storing model weights
        epsilon (float): Threshold parameter for generating region-attribute mappings
    """

    print(f"=> Generating region-attribute mappings on split {split}")
    model.eval()

    # Load attribute embeddings
    attr_to_embs = torch.load(f"{data_dir}/attr_embs.pth")
    attr = []
    attr_embs = []
    for a in attr_to_embs:
        attr.append(a)
        attr_embs.append(attr_to_embs[a])
    attr_embs = torch.tensor(np.stack(attr_embs)).cuda().to(torch.float32)

    with torch.no_grad():
        image_to_regattr_map = {}
        for step, sample in enumerate(tqdm(dataloader)):
            pred = model(sample)

            img_embs = torch.nn.functional.normalize(
                pred["region_proj_embs"].float(), dim=2
            )

            # Compute similarity scores between each attribute and each region
            attr_to_reg_scores = {}
            attr_ids = sample["attr_labels"].squeeze().nonzero()
            for a in attr_ids.flatten().tolist():
                if one_proj:
                    img_emb = img_embs[:, 0, :]
                else:
                    img_emb = img_embs[:, a, :]
                txt_emb = attr_embs[a, :]
                scores = img_emb @ txt_emb
                attr_to_reg_scores[attr[a]] = scores.tolist()

            # Store computed similarity scores
            image_id = sample["image_id"][0]
            image_to_regattr_map[image_id] = attr_to_reg_scores

    # Save region-attribute mappings
    save_reg_attr_maps(image_to_regattr_map, split, epsilon, data_dir, checkpoint_dir)


def save_reg_attr_maps(attr_to_reg_scores, split, epsilon, data_dir, checkpoint_dir):
    """
    Save region-attribute mappings.

    Parameters:
        attr_to_reg_scores (dict): Similarity scores between attributes and regions
        split (str): Data split (e.g. "train", "val")
        epsilon (float): Threshold parameter for generating region-attribute mappings
        data_dir (str): Filepath to directory with data
        checkpoint_dir (str): Directory for storing model weights
    """

    def assign_sents_to_attributes(text, a):
        text = [t.lower() for t in text]
        selected_sents = []
        for i in range(len(text)):
            if f" {a}" in text[i]:
                selected_sents.append(i)
        return np.random.choice(selected_sents)

    dataset = pd.read_feather(f"{data_dir}/annotations.feather")
    dataset = dataset[dataset["split"] == split]

    all_assigned_attributes = []
    all_assigned_text = []
    for idx, row in dataset.iterrows():
        image_id = str(row["image_id"])

        # Assign each attribute to the region(s) with the highest similarity scores
        attributes = row["attributes"]
        assigned_attributes = [[] for x in range(row["num_regions"])]
        for a in attributes:
            max_score = max(attr_to_reg_scores[image_id][a]) - epsilon
            valid_img = np.where(np.array(attr_to_reg_scores[image_id][a]) > max_score)[
                0
            ]
            for i in valid_img:
                assigned_attributes[i].append(a)
        all_assigned_attributes.append(copy.deepcopy(assigned_attributes))

        # Identify sentences from the original description corresponding to each attribute
        attribute_to_text = {}
        text = [x.strip() for x in row["text"].split(".") if len(x) > 3]
        for a in row["attributes"]:
            assigned_sents = assign_sents_to_attributes(text, a)
            attribute_to_text[a] = text[assigned_sents]

        # Assign the appropriate segments of the original description to each region
        for j in range(len(assigned_attributes)):
            for k in range(len(assigned_attributes[j])):
                assigned_attributes[j][k] = attribute_to_text[assigned_attributes[j][k]]
            assigned_attributes[j] = " . ".join(assigned_attributes[j])
        all_assigned_text.append(assigned_attributes)

    # Save region-attribute mappings
    dataset.insert(dataset.shape[1], "assigned_attributes", all_assigned_attributes)
    dataset.insert(dataset.shape[1], "assigned_text", all_assigned_text)
    dataset.reset_index(inplace=True, drop=True)
    dataset.to_feather(f"{checkpoint_dir}/mapping_{split}.feather")

    compute_mapping_performance(dataset, split)


def compute_mapping_performance(dataset, split):
    """
    Evaluate region-attribute mappings.

    Parameters:
        dataset (pd.DataFrame): Dataframe with dataset annotations
        split (str): Data split (e.g. "train", "val")
    """

    print(f"=> Evaluating region-attribute mappings for split {split}")
    mapping_recall = [0, 0]
    mapping_precision = [0, 0]
    for idx, row in dataset.iterrows():
        for r in range(row["num_regions"]):
            assigned_attr = row["assigned_attributes"][r]
            true_attr = row["reg_to_attr"][r]
            for attr in assigned_attr:
                mapping_recall[0] += attr in true_attr
                mapping_precision[0] += attr in true_attr
            mapping_precision[1] += len(assigned_attr)
            mapping_recall[1] += len(true_attr)
    mapping_precision = (mapping_precision[0] / mapping_precision[1]) * 100
    mapping_recall = (mapping_recall[0] / mapping_recall[1]) * 100

    print("Mapping Precision:", np.round(mapping_precision, 2))
    print("Mapping Recall:", np.round(mapping_recall, 2))
    print(
        "Mapping F1:",
        np.round(
            2
            * mapping_precision
            * mapping_recall
            / (mapping_precision + mapping_recall),
            2,
        ),
    )
