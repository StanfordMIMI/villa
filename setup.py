from setuptools import find_packages, setup

_REQUIRED = []

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