# Multimodal Sentiment Analysis Model Explanation

This document explains the `models.py` file which implements a multimodal sentiment and emotion analysis system. The model combines text, video, and audio data to predict emotions and sentiment in conversations.

## Overview

The multimodal sentiment analysis system processes three types of data simultaneously:
- **Text**: What people say (words, sentences)
- **Video**: How people look (facial expressions, gestures)
- **Audio**: How people sound (tone, pitch, volume)

By combining these three modalities, the model can achieve more accurate emotion and sentiment recognition than using any single modality alone.

## Main Components

### 1. TextEncoder

```python
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained BERT model that understands language
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters to use it as-is without modifying it
        for param in self.bert.parameters():
            param.requires_grad = False

        # Convert BERT's 768-dimensional output to a smaller 128-dimensional representation
        self.projection = nn.Linear(768, 128)
```

**What it does:**
- Processes the text part of the input (what people say)
- Uses a pre-trained BERT model to understand language context
- Converts the text into a 128-dimensional numerical representation

**Key techniques:**
1. **Transfer Learning**: Uses a pre-trained BERT model that already understands language
2. **Parameter Freezing**: Keeps BERT's parameters fixed to save computation and prevent overfitting
3. **Dimensionality Reduction**: Reduces the 768-dimensional BERT output to a more manageable 128 dimensions

### 2. VideoEncoder

```python
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained 3D video model
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # Freeze the video model parameters to use it as-is
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the final layer to output a 128-dimensional representation
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),  # Activation function that introduces non-linearity
            # Helps prevent overfitting by randomly dropping 20% of values
            nn.Dropout(0.2)
        )
```

**What it does:**
- Processes the visual part of the input (facial expressions, gestures)
- Uses a pre-trained 3D video model (R3D-18) to analyze motion and visual patterns
- Converts video frames into a 128-dimensional representation

**Key techniques:**
1. **3D Convolutional Network**: Uses R3D-18, which can process temporal information in videos
2. **Transfer Learning**: Leverages a pre-trained model that already understands visual patterns
3. **Dropout Regularization**: Randomly drops 20% of values during training to prevent overfitting

### 3. AudioEncoder

```python
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First layer extracts basic audio patterns
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),  # Normalizes the data for more stable learning
            nn.ReLU(),  # Activation function
            nn.MaxPool1d(2),  # Reduces the size by taking maximum values

            # Second layer extracts more complex audio patterns
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third layer for even more complex patterns
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Final layer for the most abstract audio features
            nn.Conv1d(256, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Adaptive pooling ensures fixed output size regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to 128-dimensional representation
        self.projection = nn.Linear(512, 128)
```

**What it does:**
- Processes the audio part of the input (tone, pitch, etc.)
- Uses convolutional layers to extract patterns from audio features
- Converts audio features into a 128-dimensional representation

**Key techniques:**
1. **Convolutional Neural Network**: Uses 1D convolutions to detect patterns in audio
2. **Batch Normalization**: Stabilizes and accelerates training
3. **Hierarchical Feature Extraction**: Progressively extracts more complex features through multiple layers
4. **Adaptive Pooling**: Ensures consistent output size regardless of input length

### 4. MultimodalSentimentModel

```python
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize the three encoders for different types of data
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer: combines the three types of information
        self.fusion_layer = nn.Sequential(
            # Combine the three 128-dim features into one 256-dim representation
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),  # Normalize for stable learning
            nn.ReLU(),  # Activation function
            # Prevent overfitting by randomly dropping 30% of values
            nn.Dropout(0.3)
        )

        # Emotion classifier: predicts specific emotions (7 categories)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 emotion categories
        )
        
        # Sentiment classifier: predicts positive/negative/neutral sentiment (3 categories)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 sentiment categories
        )
```

**What it does:**
- Combines the outputs from all three encoders (text, video, audio)
- Processes the combined information through a fusion layer
- Makes two separate predictions:
  - Emotion classification (7 categories: anger, disgust, fear, joy, neutral, sadness, surprise)
  - Sentiment classification (3 categories: negative, neutral, positive)

**Key techniques:**
1. **Multimodal Fusion**: Combines features from different modalities (text, video, audio)
2. **Multi-task Learning**: Simultaneously predicts both emotions and sentiment
3. **Hierarchical Classification**: Uses a shared representation for both tasks

### 5. MultimodalTrainer

```python
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to the appropriate device
        self.model.to(self.device)

        # Calculate class weights to handle imbalanced data
        self.emotion_weights = compute_class_weights(train_loader.dataset)[
            'emotion'].to(self.device)
        self.sentiment_weights = compute_class_weights(train_loader.dataset)[
            'sentiment'].to(self.device)

        # Set up optimizer (Adam) with learning rate and weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,  # Learning rate
            weight_decay=1e-5  # Weight decay for regularization
        )

        # Learning rate scheduler that reduces learning rate when progress plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=2,  # Wait 2 epochs before reducing learning rate
            verbose=True  # Print message when learning rate changes
        )

        # Set up loss functions for emotion and sentiment classification
        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=self.emotion_weights,  # Apply class weights
            label_smoothing=0.05  # Slight smoothing to prevent overconfidence
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,  # Slight smoothing to prevent overconfidence
            weight=self.sentiment_weights  # Apply class weights
        )

        # Set up TensorBoard for logging and visualization
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', current_time)
        self.writer = SummaryWriter(log_dir)
        self.current_epoch = 0
```

**What it does:**
- Manages the entire training process for the multimodal sentiment model
- Handles optimization, loss calculation, and performance evaluation
- Logs metrics for tracking progress

**Key techniques:**
1. **Class Weighting**: Handles imbalanced data by giving more importance to underrepresented classes
2. **Adam Optimizer**: Uses an adaptive learning rate optimization algorithm
3. **Learning Rate Scheduling**: Reduces learning rate when progress plateaus
4. **Label Smoothing**: Prevents the model from becoming overconfident
5. **TensorBoard Integration**: Logs metrics for visualization and analysis

## Training Process

The training process consists of two main methods:

### 1. train_epoch

```python
def train_epoch(self):
    # Set model to training mode
    self.model.train()
    losses = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    # Process each batch in the training data
    for batch in self.train_loader:
        # Move data to the appropriate device (GPU/CPU)
        text_inputs = {
            'input_ids': batch['text_inputs']['input_ids'].to(self.device),
            'attention_mask': batch['text_inputs']['attention_mask'].to(self.device)
        }
        video_frames = batch['video_frames'].to(self.device)
        audio_features = batch['audio_features'].to(self.device)
        emotion_labels = batch['emotion_label'].to(self.device)
        sentiment_labels = batch['sentiment_label'].to(self.device)

        # Zero gradients from previous batch
        self.optimizer.zero_grad()

        # Forward pass: get model predictions
        outputs = self.model(text_inputs, video_frames, audio_features)

        # Calculate losses for both emotion and sentiment
        emotion_loss = self.emotion_criterion(
            outputs["emotions"], emotion_labels)
        sentiment_loss = self.sentiment_criterion(
            outputs["sentiments"], sentiment_labels)
        total_loss = emotion_loss + sentiment_loss  # Combined loss

        # Backward pass: calculate gradients
        total_loss.backward()

        # Update weights
        self.optimizer.step()

        # Track predictions and true labels for metrics calculation
        all_emotion_preds.extend(
            outputs["emotions"].argmax(dim=1).cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_sentiment_preds.extend(
            outputs["sentiments"].argmax(dim=1).cpu().numpy())
        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

        # Track losses
        losses['total'] += total_loss.item()
        losses['emotion'] += emotion_loss.item()
        losses['sentiment'] += sentiment_loss.item()

    # Calculate average loss and metrics
    avg_loss = {k: v/len(self.train_loader) for k, v in losses.items()}
    
    # Calculate metrics: precision and accuracy
    emotion_precision = precision_score(
        all_emotion_labels, all_emotion_preds, average='weighted')
    emotion_accuracy = accuracy_score(
        all_emotion_labels, all_emotion_preds)
    sentiment_precision = precision_score(
        all_sentiment_labels, all_sentiment_preds, average='weighted')
    sentiment_accuracy = accuracy_score(
        all_sentiment_labels, all_sentiment_preds)
    
    # Log metrics
    self.log_metrics(avg_loss, {
        'emotion_precision': emotion_precision,
        'emotion_accuracy': emotion_accuracy,
        'sentiment_precision': sentiment_precision,
        'sentiment_accuracy': sentiment_accuracy
    })
    
    return avg_loss, metrics
```

**What it does:**
- Trains the model for one complete epoch (one pass through the entire dataset)
- Updates model weights to minimize the loss function
- Tracks and logs various metrics (loss, accuracy, precision)

**Key steps in training:**
1. **Forward Pass**: Generate predictions from input data
2. **Loss Calculation**: Compute how far predictions are from true labels
3. **Backward Pass**: Calculate gradients of the loss with respect to model parameters
4. **Parameter Update**: Adjust model weights to reduce the loss
5. **Metric Tracking**: Calculate and log performance metrics

### 2. evaluate

```python
def evaluate(self, data_loader, phase="val"):
    # Set model to evaluation mode
    self.model.eval()
    losses = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    # No gradient calculation needed for evaluation
    with torch.inference_mode():
        for batch in data_loader:
            # Process batch similar to training, but without gradient calculation
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(self.device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(self.device)
            }
            video_frames = batch['video_frames'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            sentiment_labels = batch['sentiment_label'].to(self.device)

            # Get model predictions
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Calculate losses
            emotion_loss = self.emotion_criterion(
                outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            # Track predictions and true labels
            all_emotion_preds.extend(
                outputs["emotions"].argmax(dim=1).cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_sentiment_preds.extend(
                outputs["sentiments"].argmax(dim=1).cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

            # Track losses
            losses['total'] += total_loss.item()
            losses['emotion'] += emotion_loss.item()
            losses['sentiment'] += sentiment_loss.item()

    # Calculate average loss and metrics
    avg_loss = {k: v/len(data_loader) for k, v in losses.items()}
    
    # Calculate metrics
    emotion_precision = precision_score(
        all_emotion_labels, all_emotion_preds, average='weighted')
    emotion_accuracy = accuracy_score(
        all_emotion_labels, all_emotion_preds)
    sentiment_precision = precision_score(
        all_sentiment_labels, all_sentiment_preds, average='weighted')
    sentiment_accuracy = accuracy_score(
        all_sentiment_labels, all_sentiment_preds)
    
    # Log metrics
    self.log_metrics(avg_loss, {
        'emotion_precision': emotion_precision,
        'emotion_accuracy': emotion_accuracy,
        'sentiment_precision': sentiment_precision,
        'sentiment_accuracy': sentiment_accuracy
    }, phase=phase)
    
    # Update learning rate scheduler if in validation phase
    if phase == "val":
        self.scheduler.step(avg_loss['total'])
    
    return avg_loss, metrics
```

**What it does:**
- Evaluates the model on validation or test data
- Calculates the same metrics as during training, but without updating weights
- Updates the learning rate scheduler based on validation performance

**Key differences from training:**
1. **No Gradient Calculation**: Uses `torch.inference_mode()` to disable gradient tracking
2. **No Weight Updates**: Doesn't call `optimizer.step()`
3. **Learning Rate Scheduling**: Updates the learning rate based on validation loss

## Advanced Techniques Used

### 1. Class Weighting

```python
def compute_class_weights(dataset):
    """
    Calculate weights for each emotion and sentiment class to handle imbalanced data.

    In real-world conversations, some emotions are rarer than others (e.g., surprise
    might be less common than neutral). This function gives more importance to
    underrepresented classes during training.
    """
    # Count samples per class
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        emotion_label = sample['emotion_label']
        sentiment_label = sample['sentiment_label']

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    # Calculate weights (less frequent classes get higher weights)
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts
    
    # Normalize weights so they sum to the number of classes
    emotion_weights = emotion_weights * (7 / emotion_weights.sum())
    sentiment_weights = sentiment_weights * (3 / sentiment_weights.sum())
    
    return {
        'emotion': emotion_weights,
        'sentiment': sentiment_weights
    }
```

**What it does:**
- Calculates weights for each class based on its frequency in the dataset
- Gives higher weight to underrepresented classes
- Helps the model pay more attention to rare emotions/sentiments

**Why it's important:**
- Real-world emotion data is often imbalanced (e.g., "neutral" is more common than "surprise")
- Without weighting, the model would optimize for common classes and perform poorly on rare ones
- Class weighting improves overall performance by balancing the importance of all classes

### 2. Learning Rate Scheduling

```python
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer,
    patience=2,  # Wait 2 epochs before reducing learning rate
    verbose=True  # Print message when learning rate changes
)
```

**What it does:**
- Monitors validation loss and reduces the learning rate when progress plateaus
- Waits for 2 epochs of no improvement before reducing the rate
- Helps the model converge to a better solution

**Why it's important:**
- A high learning rate helps the model learn quickly at first
- As training progresses, a smaller learning rate helps fine-tune the weights
- Automatic scheduling removes the need to manually adjust the learning rate

### 3. Label Smoothing

```python
self.emotion_criterion = nn.CrossEntropyLoss(
    weight=self.emotion_weights,  # Apply class weights
    label_smoothing=0.05  # Slight smoothing to prevent overconfidence
)
```

**What it does:**
- Slightly "softens" the target labels (e.g., instead of [0,1,0,0], uses [0.0125,0.95,0.0125,0.0125])
- Prevents the model from becoming too confident in its predictions

**Why it's important:**
- Models can become overconfident and assign probabilities close to 1.0
- Overconfident models generalize poorly to new data
- Label smoothing encourages more reasonable probability distributions

## Conclusion

The multimodal sentiment analysis model demonstrates several advanced deep learning techniques:

1. **Multimodal Learning**: Combining different types of data (text, video, audio) for better predictions
2. **Transfer Learning**: Using pre-trained models as starting points for each modality
3. **Multi-task Learning**: Simultaneously predicting both emotions and sentiment
4. **Regularization Techniques**: Using dropout and weight decay to prevent overfitting
5. **Class Weighting**: Handling imbalanced data by giving more importance to rare classes
6. **Adaptive Learning Rate**: Using learning rate scheduling to improve convergence
7. **Label Smoothing**: Preventing overconfidence for better generalization

This architecture shows how combining multiple sources of information can lead to more robust emotion and sentiment recognition, which is particularly valuable for understanding human communication in its full context.
