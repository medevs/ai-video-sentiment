# Multimodal Video Sentiment Analysis

This project aims to build and train a multimodal AI model for sentiment and emotion analysis using video, audio, and text data from the MELD (Multimodal EmotionLines Dataset).

## Project Overview

The goal is to create a model that can analyze emotions and sentiment from TV show clips by processing:
- Video frames (visual features)
- Audio features (speech tone, volume, etc.)
- Text transcriptions (dialogue content)

## Dataset

This project uses the [MELD dataset](https://affective-meld.github.io/), which contains:
- Multimodal data from the TV show "Friends"
- 13,000+ utterances from 1,400+ dialogues
- 7 emotion classes (anger, disgust, fear, joy, neutral, sadness, surprise)
- 3 sentiment classes (positive, negative, neutral)

### Dataset Structure

```
sentiment-model/dataset/
├── train/                  # Training data
│   ├── train_splits/       # Video clips for training
│   └── train_sent_emo.csv  # Labels and metadata for training
├── dev/                    # Validation data
│   ├── dev_splits_complete/# Video clips for validation
│   └── dev_sent_emo.csv    # Labels and metadata for validation
└── test/                   # Test data
    ├── output_repeated_splits_test/ # Video clips for testing
    └── test_sent_emo.csv   # Labels and metadata for testing
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/medevs/ai-video-sentiment.git
cd ai-video-sentiment
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Due to the large size of the dataset, it's not included in this repository. Follow these steps to set it up:

1. Download the MELD dataset from the [official website](https://affective-meld.github.io/)
2. Extract the downloaded files
3. Create the following directory structure:
```
sentiment-model/dataset/
├── train/
├── dev/
└── test/
```
4. Move the extracted files to their respective directories:
   - Move `train_splits` and `train_sent_emo.csv` to the `train/` directory
   - Move `dev_splits_complete` and `dev_sent_emo.csv` to the `dev/` directory
   - Move `output_repeated_splits_test` and `test_sent_emo.csv` to the `test/` directory

## Model Architecture

The model uses a multimodal approach:
- **Video**: Processes video frames using a CNN architecture
- **Audio**: Extracts mel spectrogram features from audio
- **Text**: Uses BERT for text embedding
- These features are then fused together for final classification

## Usage

### Training

```bash
# Code for training will be added
```

### Evaluation

```bash
# Code for evaluation will be added
```

### Inference

```bash
# Code for inference will be added
```

## Project Structure

```
ai-video-sentiment/
├── sentiment-model/
│   ├── dataset/           # Dataset files (not included in repo)
│   └── training/          # Training code
│       ├── meld_dataset.py # Dataset loading and processing
│       └── meld_dataset.md # Documentation
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Future Work

- Implement model training code
- Create a front-end interface for demo
- Experiment with different fusion techniques
- Fine-tune hyperparameters

## License

[Add appropriate license information]

## Acknowledgements

- [MELD Dataset](https://affective-meld.github.io/)
- [Friends TV Show](https://www.warnerbros.com/tv/friends)
