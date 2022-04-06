from setuptools import find_packages, setup

setup(
    name="estorch",
    version="0.0.1",
    author="Kohei Shinohara",
    license="MIT",
    description="PyTorch implementation for charge transfer and electronstatic interactions",
    packages=find_packages(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "estorch-train = estorch.scripts.train:main",
            "estorch-requeue = estorch.scripts.requeue:main",
        ]
    },
    install_requires=[
        "torch>=1.8.0",
        "torch-geometric>=1.7.1",
        "nequip>=0.3.3",
    ],
)
