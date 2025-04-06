# Detailed Model Architecture

This document explains the specific architecture of our multimodal sentiment analysis model.

## Overall Architecture

Our model has three main branches that process different types of data:
1. Text branch
2. Audio branch
3. Video branch

These branches are then combined in a fusion layer to make the final prediction.

```
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│                   │   │                   │   │                   │
│   Text Branch     │   │   Audio Branch    │   │   Video Branch    │
│                   │   │                   │   │                   │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │                   │
                        │   Fusion Layer    │
                        │                   │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │                   │
                        │   Classification  │
                        │                   │
                        └───────────────────┘
```

## Video Branch

The video branch processes the visual information from videos.

```
┌───────────┐    ┌───────────┐    ┌───────┐    ┌───────┐    ┌───────┐
│ Extract   │    │ Resize &  │    │       │    │       │    │       │
│ frames    │───►│ normalize │───►│ CNN   │───►│ Pool  │───►│ Linear│
│ from video│    │ frames    │    │       │    │       │    │       │
└───────────┘    └───────────┘    └───────┘    └───────┘    └───────┘
```

### Detailed Steps:

1. **Extract frames from video**
   - Take 30 frames from each video clip
   - These frames capture facial expressions and movements

2. **Resize & normalize frames**
   - Resize each frame to 224×224 pixels
   - Convert pixel values from 0-255 to 0-1 range
   - This standardizes the input for the neural network

3. **CNN (Convolutional Neural Network)**
   - Uses ResNet-50 architecture
   - Extracts visual features from each frame
   - Detects patterns like facial expressions, gestures

4. **Pool**
   - Combines features across frames
   - Reduces dimensions while keeping important information

5. **Linear**
   - Final transformation of video features
   - Prepares them for fusion with other modalities

## Text Branch

The text branch processes the dialogue.

```
┌───────────┐    ┌───────────┐    ┌───────┐    ┌───────┐
│ Tokenize  │    │ BERT      │    │       │    │       │
│ text      │───►│ Encoder   │───►│ Pool  │───►│ Linear│
│ input     │    │           │    │       │    │       │
└───────────┘    └───────────┘    └───────┘    └───────┘
```

### Detailed Steps:

1. **Tokenize text input**
   - Break dialogue into tokens (words and parts of words)
   - Convert tokens to numbers that BERT can understand
   - Add special tokens like [CLS] and [SEP]

2. **BERT Encoder**
   - Pre-trained language model
   - 12 transformer layers
   - Processes tokens while considering context
   - Understands meaning, sentiment, and relationships between words

3. **Pool**
   - Takes output from BERT's [CLS] token
   - This token summarizes the entire text

4. **Linear**
   - Final transformation of text features
   - Prepares them for fusion with other modalities

## Audio Branch

The audio branch processes the speech sounds from the videos.

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│ Extract   │    │ Create    │    │           │    │           │    │           │
│ audio     │───►│ mel       │───►│ Conv1D    │───►│ MaxPool   │───►│ BatchNorm │
│ from video│    │ spectrogram│    │           │    │           │    │           │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘
                                                                          │
                                                                          ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│           │    │           │    │           │    │           │    │           │
│ Linear    │◄───│ Dropout   │◄───│ ReLU      │◄───│ Conv1D    │◄───│ ReLU      │
│           │    │           │    │           │    │           │    │           │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘
```

### Detailed Steps:

1. **Extract audio from video**
   - Separate audio track from video file
   - Convert to WAV format at 16kHz sample rate
   - This isolates the speech component

2. **Create mel spectrogram**
   - Convert audio waveform to visual representation
   - Shows how frequencies change over time
   - Highlights patterns in speech tone and emotion

3. **Conv1D (1D Convolution)**
   - Applies filters to detect patterns in the spectrogram
   - Captures features like tone changes, emphasis, pauses
   - Uses 64 filters with kernel size 3

4. **MaxPool**
   - Reduces dimensions by keeping only the strongest signals
   - Makes the model more efficient and robust

5. **BatchNorm (Batch Normalization)**
   - Normalizes the data to improve training stability
   - Helps the model learn faster

6. **ReLU (Rectified Linear Unit)**
   - Activation function that introduces non-linearity
   - Allows the model to learn complex patterns

7. **Conv1D (second layer)**
   - Further processes the audio features
   - Uses 128 filters with kernel size 3
   - Detects more complex audio patterns

8. **ReLU (second activation)**
   - Another non-linear activation

9. **Dropout**
   - Randomly turns off 30% of neurons during training
   - Prevents overfitting (memorizing instead of learning)

10. **Linear**
    - Final transformation of audio features
    - Prepares them for fusion with other modalities

## Fusion Layer

The fusion layer combines information from all three branches.

```
┌───────────────┐
│ Concatenate   │
│ features from │
│ all branches  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Linear layer  │
│ (1024 units)  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ ReLU          │
│ activation    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dropout       │
│ (0.5)         │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Linear layer  │
│ (512 units)   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ ReLU          │
│ activation    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Dropout       │
│ (0.5)         │
└───────────────┘
```

### Detailed Steps:

1. **Concatenate features**
   - Join text, audio, and video features into one large vector
   - This combines all the information from different sources

2. **Linear layer (1024 units)**
   - First dense layer that processes the combined features
   - Reduces dimensions and finds relationships between modalities

3. **ReLU activation**
   - Adds non-linearity to help learn complex patterns

4. **Dropout (0.5)**
   - Turns off 50% of neurons during training
   - Prevents the model from relying too much on any one feature

5. **Linear layer (512 units)**
   - Second dense layer that further processes the features
   - Creates a more compact representation

6. **ReLU activation**
   - Another non-linear activation

7. **Dropout (0.5)**
   - More regularization to prevent overfitting

## Classification Heads

Our model has two separate classification heads:

### Emotion Classification Head

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Linear layer  │    │ Softmax       │    │ 7 emotion     │
│ (512 → 7)     │───►│ activation    │───►│ probabilities │
│               │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

1. **Linear layer**
   - Takes the 512-dimensional fused features
   - Outputs 7 values (one for each emotion class)

2. **Softmax activation**
   - Converts raw outputs to probabilities
   - All values sum to 1.0

3. **Emotion classes**
   - anger
   - disgust
   - fear
   - joy
   - neutral
   - sadness
   - surprise

### Sentiment Classification Head

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Linear layer  │    │ Softmax       │    │ 3 sentiment   │
│ (512 → 3)     │───►│ activation    │───►│ probabilities │
│               │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

1. **Linear layer**
   - Takes the same 512-dimensional fused features
   - Outputs 3 values (one for each sentiment class)

2. **Softmax activation**
   - Converts raw outputs to probabilities
   - All values sum to 1.0

3. **Sentiment classes**
   - negative
   - neutral
   - positive

## Additional Components

### Optimizer

The optimizer controls how the model learns from errors.

```
┌───────────────┐
│ Adam          │
│ Optimizer     │
└───────────────┘
```

- **Adam optimizer**
  - Adaptive learning rate algorithm
  - Combines benefits of AdaGrad and RMSProp
  - Good default choice for deep learning
  - Learning rate: 0.0001 (1e-4)

### Loss Function

The loss function measures how wrong the model's predictions are.

```
┌───────────────┐
│ Cross Entropy │
│ Loss          │
└───────────────┘
```

- **Cross Entropy Loss**
  - Standard loss function for classification problems
  - Measures difference between predicted probabilities and true labels
  - Separate loss calculated for emotion and sentiment
  - Total loss = emotion loss + sentiment loss

### Learning Rate Scheduler

Controls how the learning rate changes during training.

```
┌───────────────┐
│ Reduce on     │
│ Plateau       │
└───────────────┘
```

- **ReduceLROnPlateau**
  - Reduces learning rate when model stops improving
  - Patience: 3 epochs (wait 3 epochs before reducing)
  - Factor: 0.1 (reduce learning rate by 10x)
  - This helps fine-tune the model in later stages of training

## Implementation Details

The model is implemented using PyTorch with the following structure:

```python
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel, self).__init__()
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_pool = nn.Linear(768, 512)
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128 * 150, 512)  # 300/2 = 150 after pooling
        )
        
        # Video encoder
        self.video_encoder = models.resnet50(pretrained=True)
        # Replace final layer
        self.video_encoder.fc = nn.Linear(2048, 512)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 1024),  # 512 from each modality
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classification heads
        self.emotion_classifier = nn.Linear(512, 7)  # 7 emotion classes
        self.sentiment_classifier = nn.Linear(512, 3)  # 3 sentiment classes
    
    def forward(self, text_inputs, audio_features, video_frames):
        # Process text
        text_outputs = self.text_encoder(**text_inputs)
        text_features = self.text_pool(text_outputs.pooler_output)  # [batch_size, 512]
        
        # Process audio
        audio_features = self.audio_encoder(audio_features)  # [batch_size, 512]
        
        # Process video
        video_features = self.video_encoder(video_frames)  # [batch_size, 512]
        
        # Concatenate features
        combined_features = torch.cat([text_features, audio_features, video_features], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)  # [batch_size, 512]
        
        # Classification
        emotion_output = self.emotion_classifier(fused_features)  # [batch_size, 7]
        sentiment_output = self.sentiment_classifier(fused_features)  # [batch_size, 3]
        
        return {
            'emotion': emotion_output,
            'sentiment': sentiment_output
        }
```

This architecture allows the model to effectively process and combine information from text, audio, and video to make accurate emotion and sentiment predictions.
