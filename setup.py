from setuptools import setup, find_packages

setup(
    name="sword",
    version="0.1.0",
    description="High-Throughput Pure-PyTorch Attention & Generation Speed Engine",
    author="TweVids",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "transformers>=5.3.0",
        "accelerate",
        "bitsandbytes",
        "numpy",
    ],
    python_requires=">=3.10",
)
