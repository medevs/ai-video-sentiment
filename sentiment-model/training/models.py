import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from meld_dataset import MELDDataset


class TextEncoder(nn.Module):
    """
    Text Encoder: Processes the text part of the input (what people say).

    This component takes the words from conversations and converts them into
    a format that the computer can understand better for sentiment analysis.
    It uses BERT, which is a pre-trained model that understands language context.
    """

    def __init__(self):
        super().__init__()
        # Load a pre-trained BERT model that understands language
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters to use it as-is without modifying it
        # This saves training time and prevents overfitting
        for param in self.bert.parameters():
            param.requires_grad = False

        # Convert BERT's 768-dimensional output to a smaller 128-dimensional representation
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        """
        Process text data through the encoder.

        Parameters:
            input_ids: The numeric representation of words in the text
            attention_mask: Tells the model which words to pay attention to and which to ignore

        Returns:
            A 128-dimensional representation of the text content
        """
        # Extract meaningful information from the text using BERT (embeddings)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the special [CLS] token which summarizes the entire text
        pooler_output = outputs.pooler_output

        # Convert to our smaller representation size
        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    """
    Video Encoder: Processes the visual part of the input (what people look like).

    This component analyzes the facial expressions and visual cues from video frames
    to help understand the emotional context of the conversation.
    It uses a pre-trained 3D video model (R3D-18) that can understand motion and visual patterns.
    """

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

    def forward(self, x):
        """
        Process video data through the encoder.

        Parameters:
            x: Video frames in format [batch_size, frames, channels, height, width]

        Returns:
            A 128-dimensional representation of the video content
        """
        # Rearrange dimensions to match what the video model expects
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    """
    Audio Encoder: Processes the audio part of the input (how people sound).

    This component analyzes the tone, pitch, and other audio characteristics
    to help understand the emotional context from how words are spoken.
    It uses convolutional layers to extract meaningful patterns from audio features.
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First layer extracts basic audio patterns (like tone changes)
            nn.Conv1d(64, 64, kernel_size=3),  # Analyzes 3 time steps at once
            nn.BatchNorm1d(64),  # Normalizes the data for more stable learning
            nn.ReLU(),  # Activation function
            nn.MaxPool1d(2),  # Reduces the size by taking maximum values

            # Second layer extracts more complex audio patterns
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Averages all time steps into one value
        )

        # Freeze the convolutional layers to use them as-is
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        # Final processing to get a 128-dimensional representation
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)  # Prevents overfitting
        )

    def forward(self, x):
        """
        Process audio data through the encoder.

        Parameters:
            x: Audio features

        Returns:
            A 128-dimensional representation of the audio content
        """
        x = x.squeeze(1)  # Remove unnecessary dimension

        # Extract audio features
        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        # Remove the last dimension and apply final processing
        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    """
    Main Sentiment Analysis Model: Combines text, video, and audio to understand emotions.

    This model takes all three types of information (what people say, how they look,
    and how they sound) and combines them to predict both the emotion (like happy, sad)
    and sentiment (positive, negative, neutral) of a conversation.

    This combined approach is called "multimodal" because it uses multiple modes of information.
    """

    def __init__(self):
        super().__init__()

        # Create encoders for each type of input
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
            nn.Linear(256, 64),  # Reduce dimensions
            nn.ReLU(),  # Activation function
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 7)  # Output 7 scores, one for each emotion category
        )

        # Sentiment classifier: predicts overall sentiment (3 categories)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),  # Reduce dimensions
            nn.ReLU(),  # Activation function
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 3)  # Output 3 scores: negative, neutral, positive
        )

    def forward(self, text_inputs, video_frames, audio_features):
        """
        Process all three types of input data and predict emotions and sentiment.

        Parameters:
            text_inputs: Dictionary containing text data (words spoken)
            video_frames: Visual data (facial expressions, gestures)
            audio_features: Audio data (tone, pitch, speaking style)

        Returns:
            Dictionary with two keys:
            - 'emotions': Scores for 7 emotion categories
            - 'sentiments': Scores for 3 sentiment categories (negative, neutral, positive)
        """
        # Process each type of input through its specialized encoder
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Combine all features into one representation
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)  # [batch_size, 128 * 3]

        # Process the combined features through the fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Get emotion and sentiment predictions
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        # Return both sets of predictions
        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }

# if __name__ == "__main__":
#     dataset = MELDDataset('../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

#     sample = dataset[0]

#     model = MultimodalSentimentModel()
#     model.eval()

#     text_inputs = {
#         'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
#         'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
#     }

#     video_frames = sample['video_frames'].unsqueeze(0)
#     audio_features = sample['audio_features'].unsqueeze(0)

#     with torch.inference_mode():
#         outputs = model(text_inputs, video_frames, audio_features)

#         empotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
#         sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

#     emotion_map = {
#         0: 'anger',
#         1: 'disgust',
#         2: 'fear',
#         3: 'joy',
#         4: 'neutral',
#         5: 'sadness',
#         6: 'surprise'
#     }

#     sentiment_map = {
#         0: 'negative',
#         1: 'neutral',
#         2: 'positive'
#     }

#     for i, prob in enumerate(empotion_probs):
#         print(f"{emotion_map[i]}: {prob:.2f}")

#     for i, prob in enumerate(sentiment_probs):
#         print(f"{sentiment_map[i]}: {prob:.2f}")

#     print("Predictions for utterance")


def compute_class_weights(dataset):
    """
    Calculate weights for each emotion and sentiment class to handle imbalanced data.

    In real-world conversations, some emotions are rarer than others (e.g., surprise
    might be less common than neutral). This function gives more importance to
    underrepresented classes during training.

    Parameters:
        dataset: The dataset containing emotion and sentiment labels

    Returns:
        Dictionary with two keys:
        - 'emotion': Weights for emotion classes
        - 'sentiment': Weights for sentiment classes
    """
    # Count occurrences of each emotion and sentiment
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)

    # Loop through the dataset to count each class
    for i in range(len(dataset)):
        sample = dataset[i]
        emotion_label = sample['emotion_label']
        sentiment_label = sample['sentiment_label']

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    # Calculate weights (less frequent classes get higher weights)
    # The formula gives higher weight to classes with fewer samples
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts

    # Normalize weights so they sum to the number of classes
    emotion_weights = emotion_weights * (7 / emotion_weights.sum())
    sentiment_weights = sentiment_weights * (3 / sentiment_weights.sum())

    print("Emotion class weights:", emotion_weights)
    print("Sentiment class weights:", sentiment_weights)

    return {
        'emotion': emotion_weights,
        'sentiment': sentiment_weights
    }


class MultimodalTrainer:
    """
    Trainer: Handles the training process for the sentiment analysis model.

    This class manages the entire training process, including:
    - Running training loops
    - Calculating losses
    - Updating model weights
    - Evaluating model performance
    - Logging metrics for tracking progress

    It uses techniques like learning rate scheduling and class weighting to
    improve training effectiveness.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Initialize the trainer with a model and data loaders.

        Parameters:
            model: The MultimodalSentimentModel to train
            train_loader: DataLoader containing training data
            val_loader: DataLoader containing validation data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset sizes
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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
            mode='min',  # Minimize loss
            factor=0.5,  # Reduce learning rate by half when plateauing
            patience=2,  # Wait 2 epochs before reducing learning rate
            verbose=True  # Print message when learning rate changes
        )

        # Set up loss functions for emotion and sentiment classification
        # CrossEntropyLoss is standard for classification tasks
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
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to {log_dir}")

    def log_metrics(self, losses, metrics=None, phase="train"):
        """
        Log training metrics to TensorBoard for visualization.

        Parameters:
            losses: Dictionary of loss values
            metrics: Dictionary of evaluation metrics (accuracy, precision)
            phase: Either "train" or "val" to indicate which phase is being logged
        """
        # Get current step for logging
        step = self.current_epoch if phase == "train" else self.current_epoch + 0.5

        # Log all losses
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(
                f'{phase}/{loss_name}_loss', loss_value, step)

        # Log all metrics if provided
        if metrics:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(
                    f'{phase}/{metric_name}', metric_value, step)

        # Print a summary of the metrics
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])

        if metrics:
            metrics_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(
                f"{phase.capitalize()} Epoch {self.current_epoch}: {loss_str}, {metrics_str}")
        else:
            print(f"{phase.capitalize()} Epoch {self.current_epoch}: {loss_str}")

    def train_epoch(self):
        """
        Train the model for one complete epoch.

        An epoch means going through the entire training dataset once.
        This function:
        1. Sets the model to training mode
        2. Processes each batch of data
        3. Calculates losses
        4. Updates model weights
        5. Tracks and logs metrics

        Returns:
            Tuple of (average losses, metrics)
        """
        self.model.train()  # Set model to training mode
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

            # Calculate losses for both emotion and sentiment (using raw logits)
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

        # Calculate average loss over all batches
        avg_loss = {k: v/len(self.train_loader) for k, v in losses.items()}

        # Calculate metrics: precision and accuracy for both tasks
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_accuracy = accuracy_score(
            all_emotion_labels, all_emotion_preds)
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels, all_sentiment_preds)

        # Log all metrics
        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        })

        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }

    def evaluate(self, data_loader, phase="val"):
        """
        Evaluate the model on validation or test data.

        This function:
        1. Sets the model to evaluation mode (no weight updates)
        2. Processes each batch of data
        3. Calculates losses and metrics
        4. Updates the learning rate scheduler if needed

        Parameters:
            data_loader: DataLoader containing evaluation data
            phase: Either "val" or "test" to indicate which phase

        Returns:
            Tuple of (average losses, metrics)
        """
        self.model.eval()  # Set model to evaluation mode
        losses = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        # No gradient calculation needed for evaluation
        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

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

        # Calculate average loss
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

        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }


if __name__ == "__main__":
    """
    This section runs when the script is executed directly.
    It demonstrates how to use the model with a sample from the dataset.
    """
    # Load a sample from the dataset
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]  # Get the first sample

    # Create and prepare the model
    model = MultimodalSentimentModel()
    model.eval()  # Set to evaluation mode

    # Prepare the input data
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),  # Add batch dimension
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)  # Add batch dimension
    audio_features = sample['audio_features'].unsqueeze(0)  # Add batch dimension

    # Run the model without calculating gradients
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        # Convert raw scores to probabilities
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    # Map numeric indices to emotion and sentiment names
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }

    # Print emotion probabilities
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    # Print sentiment probabilities
    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance")
