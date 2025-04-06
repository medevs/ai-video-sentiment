# Project Architecture

## Overview of the Project

This project analyzes emotions and sentiment in videos using AI. It combines three types of data:
1. **Text** (what people say)
2. **Audio** (how they say it)
3. **Video** (facial expressions)

## Project Structure

```
ai-video-sentiment/
├── sentiment-model/           # Main model code
│   ├── dataset/               # Dataset files
│   │   ├── train/             # Training data
│   │   ├── dev/               # Validation data
│   │   └── test/              # Testing data
│   └── training/              # Training code
│       ├── meld_dataset.py    # Dataset loading code
│       └── meld_dataset.md    # Documentation
├── theory-docs/               # Theory explanations (you are here)
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

## Data Flow

Here's how data moves through the system:

```
┌─────────────┐
│  Raw Data   │  (Video clips from MELD dataset)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocessing│  (Extract text, audio, video features)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Model Training│  (Train the multimodal model)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Inference  │  (Make predictions on new videos)
└─────────────┘
```

## Key Components

### 1. Data Loading (MELDDataset Class)

This component:
- Reads video files and CSV files with labels
- Extracts frames from videos
- Extracts audio from videos
- Processes text dialogue
- Returns all features in the right format for the model

### 2. Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Multimodal Model                     │
│                                                         │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐      │
│  │           │     │           │     │           │      │
│  │   Text    │     │   Audio   │     │   Video   │      │
│  │  Encoder  │     │  Encoder  │     │  Encoder  │      │
│  │  (BERT)   │     │           │     │  (CNN)    │      │
│  │           │     │           │     │           │      │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘      │
│        │                 │                 │            │
│        └────────┬────────┴────────┬────────┘            │
│                 │                 │                     │
│          ┌──────▼─────────────────▼──────┐              │
│          │                               │              │
│          │         Fusion Layer          │              │
│          │                               │              │
│          └───────────────┬───────────────┘              │
│                          │                              │
│                 ┌────────▼────────┐                     │
│                 │                 │                     │
│                 │  Classification │                     │
│                 │      Layer      │                     │
│                 │                 │                     │
│                 └────────┬────────┘                     │
│                          │                              │
└──────────────────────────┼──────────────────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │                   │
                 │    Predictions    │
                 │                   │
                 └───────────────────┘
```

### 3. Training Pipeline

The training process:
1. Loads data in batches
2. Passes data through the model
3. Calculates loss (error)
4. Updates model weights
5. Repeats until performance stops improving

### 4. Inference (Using the Model)

Once trained, the model can:
1. Take a new video clip
2. Extract text, audio, and video features
3. Process through the model
4. Output emotion and sentiment predictions

## Technologies Used

### Python Libraries

1. **PyTorch** - Main deep learning framework
   - Similar to TensorFlow but more flexible
   - Used for building and training neural networks

2. **Transformers** - For text processing
   - Contains pre-trained BERT models
   - Handles tokenization and text embedding

3. **OpenCV** - For video processing
   - Reads video frames
   - Resizes and normalizes images

4. **Torchaudio** - For audio processing
   - Creates mel spectrograms
   - Handles audio feature extraction

5. **Pandas** - For data handling
   - Reads CSV files with labels
   - Helps organize dataset information

### External Tools

1. **ffmpeg** - For audio extraction
   - Extracts audio tracks from video files
   - Converts to the right format for processing

## How JavaScript Developers Can Understand This

If you know JavaScript, here are some parallels:

1. **PyTorch ≈ TensorFlow.js**
   - Both are deep learning frameworks
   - PyTorch is to Python what TensorFlow.js is to JavaScript

2. **Dataset Loading ≈ fetch() + data processing**
   - Similar to fetching JSON data and transforming it

3. **Model Architecture ≈ Layers of functions**
   - Like having several processing functions that pass data to each other
   - Similar to functional programming concepts

4. **Training Loop ≈ for loop with updates**
   - Similar to iterating through data and updating state

## Future Improvements

The project could be extended with:

1. **Web Interface**
   - Upload videos for analysis
   - Display results visually
   - This could use your JavaScript skills!

2. **Real-time Processing**
   - Analyze live video streams
   - Provide immediate feedback

3. **More Emotions**
   - Detect more subtle emotions
   - Recognize mixed emotions

4. **Better Fusion Methods**
   - Improve how text, audio, and video information is combined
   - Use attention mechanisms to focus on the most important features
