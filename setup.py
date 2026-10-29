from setuptools import setup, find_packages

setup(
    name="sword",
    version="1.0.1",
    description="High-Throughput Pure-PyTorch Attention & Generation Speed Engine",
    author="TweVids",
    packages=find_packages(),
    py_modules=["train_math_grpo"],
    entry_points={
        "console_scripts": [
            "sword-train=train_math_grpo:main",
            "sword-grpo=train_math_grpo:main",
            "train-math-grpo=train_math_grpo:main",
        ],
    },
    install_requires=[
        "transformers>=4.45.0",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "einops",
        "datasets",
        "huggingface_hub",
        "peft",
        "requests>=2.31.0",
    ],
    python_requires=">=3.10",
    extras_require={"logger": ["gradio>=4.44,<7", "pandas>=2.0"]},
)
