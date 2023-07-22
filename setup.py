from setuptools import setup

_REQUIRED = [
    "pandas==1.3.5",
    "numpy>=1.18.0",
    "tqdm>=4.49.0",
    "hydra-core==1.3.2",
    "matplotlib==3.5.3",
    "Pillow==9.5.0",
    "pyrootutils==1.0.4",
    "sparse==0.13.0",
    "rich==13.4.2",
    "ftfy==6.1.1",
    "regex==2023.6.3",
    "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33",
    "scikit-image==0.19.3",
    "pyarrow",
]

setup(
    name="villa",
    version="0.0.1",
    description="ViLLA: Fine-grained vision-language representation learning from real-world data",
    author="Maya Varma",
    author_email="mvarma2@stanford.edu",
    url="https://github.com/maya124/villa",
    packages=["villa"],
    install_requires=_REQUIRED,
)
