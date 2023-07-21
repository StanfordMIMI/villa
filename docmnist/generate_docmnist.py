import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import shutil
import pyrootutils
import os

from utils import (
    load_mnist_data,
    apply_color,
    apply_shape,
    get_caption,
    save_images,
    save_ann,
)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def parse_args():
    """
    Parse input argments

    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Script for generating a DocMNIST dataset."
    )

    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument(
        "--attribute_budget",
        type=int,
        default=None,
        required=True,
        help="Attribute budget.",
    )
    parser.add_argument(
        "--target_sample_complexity",
        type=int,
        default=None,
        required=True,
        help="Max sample complexity per image.",
    )

    args = parser.parse_args()
    return args


def construct_docmnist_dataset(
    data: torch.Tensor,
    targets: torch.Tensor,
    attribute_budget: int,
    target_sample_complexity: int,
):
    """
    Generate a DocMNIST dataset.

    Parameters:
        data (torch.Tensor): Tensor of size n x 3 x 28 x 28 containing MNIST digit images
        targets (torch.Tensor): Tensor of size n containing MNIST digit labels
        attribute_budget (int): Indicates the total number of allowed attributes in the dataset
        target_sample_complexity (int): Indicates the max sample complexity of an image-text pair
    Returns:
        dataset (dict): Contains the images, individual regions, text, metadata, and region box coordinates
        avg_sample_complexity (float): Indicates the average sample complexity of the constructed dataset
    """

    dataset = {"img": [], "regions": [], "text": [], "metadata": [], "boxes": []}
    quadrant_coord = np.array(
        [
            (0, 0),
            (28, 0),
            (56, 0),
            (0, 28),
            (28, 28),
            (56, 28),
            (0, 56),
            (28, 56),
            (56, 56),
        ]
    )

    total_attr = 0
    while True:
        regions = []
        captions = []
        metadata = {"color": [], "digit": [], "shape": [], "size": []}
        num_reg_attr_pairs = 0

        while len(regions) < 9:
            # Randomly sample digit from set
            idx = np.random.choice(data.shape[0])
            reg = torch.clone(data[idx])
            digit_label = targets[idx].item()

            # Assign a random color, shape, and size to the digit
            reg, color_label = apply_color(reg)
            reg, shape_label, size_label = apply_shape(reg)

            # Check that the number of attributes in the image does not exceed the desired sample complexity
            num_reg_attr_pairs += sum(
                [
                    k is not None
                    for k in [digit_label, color_label, shape_label, size_label]
                ]
            )
            if num_reg_attr_pairs > target_sample_complexity:
                break

            # Update metadata
            metadata["digit"].append(digit_label)
            metadata["color"].append(color_label)
            metadata["shape"].append(shape_label)
            metadata["size"].append(size_label)

            # Generate caption for region
            txt = get_caption(digit_label, color_label, shape_label, size_label)

            # Store region and text
            regions.append(reg)
            captions.extend(txt)

        # Increment attribute count
        a = sum([(np.array(metadata[k]) != None).sum() for k in metadata.keys()])
        if total_attr + a > attribute_budget:
            break
        total_attr += a

        # Add metadata, regions, and the generated caption to dataset
        dataset["metadata"].append(metadata)
        dataset["regions"].append([r.numpy() for r in regions])
        captions = list(set(captions))
        np.random.shuffle(captions)
        dataset["text"].append(" . ".join(captions))

        # Create image with generated regions
        idx = quadrant_coord[
            sorted(
                np.random.choice(
                    range(len(quadrant_coord)), size=len(regions), replace=False
                )
            )
        ]
        new_im = Image.new("RGB", (84, 84))
        region_boxes = []
        for t in range(len(regions)):
            region = regions[t]
            new_im.paste(
                Image.fromarray(
                    np.transpose(region.numpy().astype("uint8"), (1, 2, 0)), "RGB"
                ),
                tuple(idx[t]),
            )
            region_boxes.append([idx[t][0], idx[t][1], idx[t][0] + 28, idx[t][1] + 28])
        new_im = np.array(new_im).transpose(2, 0, 1)
        dataset["img"].append(new_im)
        dataset["boxes"].append(region_boxes)

    # Compute the average sample complexity of the final dataset
    avg_sample_complexity = total_attr / len(dataset["img"])
    return dataset, avg_sample_complexity


def main():
    args = parse_args()
    data_dir = os.path.join(root, "data")

    np.random.seed(args.seed)
    data, targets = load_mnist_data(data_dir)
    target_comp = args.target_sample_complexity
    budget = args.attribute_budget

    # Generate train split
    print(
        f"Generating train split: attribute budget = {budget}, targeted sample complexity = {target_comp}"
    )
    train_dataset, train_complexity = construct_docmnist_dataset(
        data=data["train"],
        targets=targets["train"],
        attribute_budget=budget,
        target_sample_complexity=target_comp,
    )

    train_complexity = np.round(train_complexity, 1)
    print(
        f"Completed train split: {len(train_dataset['img'])} images, avg. sample complexity = {train_complexity}"
    )
    print(f"-----------")

    # Generate validation split
    print(
        f"Generating val split: attribute budget = 10000, targeted sample complexity = {target_comp}"
    )
    val_dataset, val_complexity = construct_docmnist_dataset(
        data=data["val"],
        targets=targets["val"],
        attribute_budget=10000,
        target_sample_complexity=target_comp,
    )
    val_complexity = np.round(val_complexity, 1)
    print(
        f"Completed val split: {len(val_dataset['img'])} images, avg. sample complexity = {val_complexity}"
    )
    print(f"-----------")

    # Save images and annotations to disk
    out_dir = Path(data_dir) / f"docmnist_{args.attribute_budget}_{train_complexity}"
    if out_dir.exists() and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    (out_dir / "images").mkdir()
    img_paths = save_images(
        images=train_dataset["img"] + val_dataset["img"], out_dir=out_dir / "images"
    )
    save_ann(
        train_data=train_dataset,
        val_data=val_dataset,
        img_paths=img_paths,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
