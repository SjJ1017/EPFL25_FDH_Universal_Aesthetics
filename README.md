# Universal Aesthetics in AI Representations
[![Wiki](https://img.shields.io/badge/wiki-documentation-blue)](https://fdh.epfl.ch/index.php/Universal_Aesthetics_(Multimodal_Focus))

A research project investigating representational convergence in Large Language Models (LLMs) and Vision Transformers (ViTs) on aesthetic data.

**Course**: Foundation of Digital Humanities (DH-405), EPFL  
**Authors**: Jiajun Shen, Yibo Yin, Yifan Zhou


---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data and Features](#data-and-features)
- [Acknowledgments](#acknowledgments)
- [Wiki](#Wikipage)

---

## Overview

This project investigates the concept of **"Universal Aesthetics"** within the context of Artificial Intelligence models, specifically Large Language Models (LLMs) and Vision Transformers (ViTs). Our central hypothesis is inspired by Tolstoy's famous opening line: *"All happy families are alike; each unhappy family is unhappy in its own way" — but applied to AI representations. 

### Key Findings

This study validates the **Platonic Representation Hypothesis** on aesthetic data and provides insights into how aesthetic factors influence model alignment and representation space. Our findings suggest that aesthetic attributes, rather than being complex or high-level features, may be **fundamental elements mastered relatively early** in a model's training process.

---

## Project Structure

```
.
├── image2poem/           # Modified from external work for poem generation
├── platonic/             # Modified from Platonic Representation codebase
│   ├── data.py          # Custom data loading modifications
│   ├── models.py        # Model list configurations
│   └── ...
├── poems/               # Dataset creation and processing
│   ├── dataset.py       # Poem loading and filtering
│   ├── images/          # Image dataset processing
│   └── ...
├── representations/     # All data analysis and visualization code
│   ├── main.py         # Main script for computing alignment scores
│   ├── plots/          # Plotting utilities
│   │   ├── correlation.py
│   │   └── utils.py
│   ├── perplexity.py   # Perplexity calculations
│   ├── semantic.py     # Semantic encoding
│   └── ...
├── results/            # Generated alignment scores and figures
└── README.md
```

### Key Components

#### `image2poem/` and `platonic/`
These directories contain modified versions of external codebases:
- **image2poem**: Adapted for poem generation from images
- **platonic**: Modified Platonic Representation code with custom:
  - Data loading mechanisms
  - Model lists for LLMs and ViTs
  - Feature extraction pipelines

**Note**: These are based on existing works with modifications specific to our experiments.

#### `poems/`
Contains all dataset creation code:
- Data collection and preprocessing
- Poem filtering and selection
- Aesthetic scoring
- Image-text/image-poem pairing

#### `representations/`
All analysis and visualization code:
- **`main.py`**: Central script to compute alignment scores with different configurations
- Feature extraction and analysis
- Correlation analysis
- Visualization generation (heatmaps, boxplots, scatter plots)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)
- 6GB+ free disk space for features

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/SjJ1017/EPFL25_FDH_Universal_Aesthetics.git
cd EPFL25_FDH_Universal_Aesthetics
```

2. **Create a virtual environment**
```bash
conda create -n aesthetics python=3.11
conda activate aesthetics
```

3. **Install dependencies**
```bash
cd platonic
pip install -e .
cd ..
```

---

## Data and Features

### Datasets

Our project uses the following datasets (available on HuggingFace):

1. **`SHENJJ1017/poem_aesthetic_eval`**: poems with aesthetic scores (~800 or more, depending on the branch)
2. **`SHENJJ1017/Image-Text`**: ~1,000 image-text/image-poem pairs with aesthetic scores

### Pre-extracted Features

All extracted features (~5.7GB) are available for download:

**Google Drive**: `https://drive.google.com/drive/folders/1vB3KCPKarg2pindNvrX6wXs_rq0IOT9D?usp=sharing`

Features include:
- LLM embeddings for 7 language models using `pool-avg`
- ViT embeddings for 7 vision models using `pool-cls`

### Models Used

**Language Models (LLMs)**:
- Meta-Llama-3-8B
- Mistral-7B-v0.1
- Gemma-7b
- Gemma-2b
- Bloomz-1b7
- OLMo-1B-hf
- Bloomz-560m

**Vision Models (ViTs)**:
- vit_tiny_patch16_224.augreg_in21k
- vit_small_patch16_224.augreg_in21k
- vit_base_patch16_224.mae
- vit_base_patch14_dinov2.lvd142m
- vit_large_patch14_dinov2.lvd142m
- vit_large_patch14_clip_224.laion2b
- vit_huge_patch14_clip_224.laion2b_ft_in12k

---

## Acknowledgments

This project builds upon and modifies code from:

1. **Platonic Representation Hypothesis**
   - Original work on model convergence
   - Modified for aesthetic data analysis

2. **Image2Poem**
   - Image-to-poem generation framework
   - Adapted for our multimodal experiments

We thank the authors of these works for making their code publicly available.

**References**:
- Platonic Representation Hypothesis: [https://arxiv.org/pdf/2405.07987]
- Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training: [https://arxiv.org/abs/1804.08473]


---

## Wikipage

- Wiki: [https://github.com/SjJ1017/EPFL25_FDH_Universal_Aesthetics](https://fdh.epfl.ch/index.php/Universal_Aesthetics_(Multimodal_Focus))

---

**Last Updated**: December 2024
