"""
Setup script for Wav2Vec2 ESC-50 project
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wav2vec2-esc50",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tuning Wav2Vec2 for ESC-50 environmental sound classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wav2vec2-esc50",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wav2vec2-train=train:main",
            "wav2vec2-inference=inference:main",
        ],
    },
)

