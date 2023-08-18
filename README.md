# ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data
This repository contains the PyTorch implementation for ViLLA (ICCV 2023).

![Overview](assets/img.png "")

## 🌴 Overview
Vision-language models (VLMs) are generally trained on datasets consisting of image-caption pairs obtained from the web. However, real-world multimodal datasets (e.g. healthcare data) are significantly more complex: each image is often paired with text (e.g. physician report) that describes many distinct attributes occurring in fine-grained regions of the image. We refer to these samples as exhibiting high image-text sample complexity, since each image-text pair can be decomposed into a large number of region-attribute pairings. Our work involves two key contributions:

- We introduce a synthetic dataset **DocMNIST**, which allows the average image-text sample complexity to be directly controlled by altering the number of region-attribute pairs per sample. We use DocMNIST to demonstrate that as the image-text sample complexity of the training dataset increases, standard VLMs struggle to learn region-attribute relationships.
- We present **Vi**sion-**L**anguage **L**earning with **A**ttributes (ViLLA), which leverages self-supervised learning in order to capture fine-grained region-attribute relationships from complex datasets. ViLLA involves two components:  (a) a lightweight, self-supervised mapping model to decompose image-text samples into region-attribute pairs, and (b) a contrastive VLM to learn representations from generated region-attribute pairs.

## ⚡️ Installation
Use the following commands to clone and install this repository. Confirm that PyTorch and torchvision are installed on your system.

```python
git clone https://github.com/StanfordMIMI/villa.git
cd villa
pip install -e .
pre-commit install
pre-commit
```

Then, create a file ```.env``` with the path to the package root (refer to ```.env_example``` for an example).

## 🔢 Using DocMNIST
DocMNIST is a synthetic vision-language training dataset designed to enable controlled evaluations of VLMs.

The ```docmnist/generate_docmnist.py``` script can be used to create a DocMNIST dataset. The following dataset-level variables can be controlled when generating a DOCMNIST dataset: the set of possible attributes (which can be modified in ```docmnist/generate_docmnist.py```), the attribute budget (specified by the ```attribute_budget``` parameter), and the targeted sample complexity for each image-text pair (specified by the ```target_sample_complexity``` parameter). Note that the true average sample complexity will vary slightly from the targeted value.

```python
python3 docmnist/generate_docmnist.py \
    --attribute_budget=30000 \
    --target_sample_complexity=16
```
The ```docmnist/visualize_data.ipynb``` notebook can be used to visualize generated images.

## ⚙️ Train ViLLA Models
### Preprocessing
First, preprocess the dataset by precomputing embeddings for all regions and attributes. We provide an example preprocessing script for DocMNIST (```preprocess/preprocess_docmnist.py```), which can be run as follows. Replace the parameter ```data_dir``` with the name of the directory where data is stored.

```python
python3 preprocess/preprocess_docmnist.py \
    --data_dir=docmnist_30000_15.2 \
```

The preprocessing script generates two outputs in the ```data_dir``` directory: (1) ```attr_embs.pth```, which contains text embeddings for all attributes, and (2) ```region_embs```, which contains image embeddings for all regions.

### Stage 1: Mapping Model
The lightweight, self-supervised mapping model decomposes image-text samples into region-attribute pairs. We provide an example config (```villa/configs/experiment/docmnist_stage1.py```) and training code (```villa/stage1.py```) for the mapping model, which can be run as follows.

```python
python3 -m villa.stage1 experiment=docmnist_stage1
```

Config parameters can be overridden from the command line with the following format:

```python
python3 -m villa.stage1 experiment=docmnist_stage1 epochs=10
```

This script generates a checkpoint for the mapping model (```last.pkl```) and region-attribute mappings (```mapping.feather```), which are stored in  ```villa/checkpoints/docmnist_stage1/```.

Note that this script does not currently support multi-GPU training. If your compute environment includes multiple GPUs, we recommend prepending ```CUDA_VISIBLE_DEVICES=0``` to the training command.

### Stage 2: Vision-Language Model
Download pretrained weights for the CLIP image encoder (```clip.pth```) from [this link](https://drive.google.com/drive/u/1/folders/1luymCKnHZ86xFFsGZAbW2nKYylVLTi-g). Store these weights in ```villa/checkpoints/```.

Given the results from Stage 1, a contrastive vision-language model can be trained to learn representations from generated region-attribute pairs. We provide an example config (```villa/configs/experiment/docmnist_stage2.py```) and training code (```villa/stage2.py```) , which can be run as follows.

```python
python3 -m villa.stage2 experiment=docmnist_stage2
```

This script generates checkpoints for the VLM, which are stored in ```villa/checkpoints/docmnist_stage2/```.


## 📎 Citation
If you find this repository useful for your work, please cite the following paper:

```
@inproceedings{varma2023villa,
  title={ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data},
  author={Varma, Maya and Delbrouck, Jean-Benoit and Hooper, Sarah and Chaudhari, Akshay and Langlotz, Curtis},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
This repository was inspired by [ViLMedic](https://github.com/jbdel/vilmedic), [CLIP](https://github.com/openai/CLIP), [RegionCLIP](https://github.com/microsoft/regionclip), and [GLoRIA](https://github.com/marshuang80/gloria).
