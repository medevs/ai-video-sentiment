"""
Multimodal Sentiment Analysis Models

This module contains neural network models for multimodal sentiment analysis,
processing text, video, and audio inputs to predict emotions and sentiment.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models


class TextEncoder(nn.Module):
    """
    Text encoder using BERT for feature extraction.
    
    Extracts semantic features from text using a frozen pre-trained BERT model
    and projects them to a 128-dimensional space.
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters to use as feature extractor only
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        """
        Process text inputs through BERT and projection layer.
        
        Args:
            input_ids: Tokenized text input IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Projected text features (batch_size, 128)
        """
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    """
    Video encoder using 3D ResNet for feature extraction.
    
    Processes video frames using a pre-trained R3D-18 model and
    projects features to a 128-dimensional space.
    """
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Process video frames through 3D CNN.
        
        Args:
            x: Video tensor (batch_size, frames, channels, height, width)
            
        Returns:
            Video features (batch_size, 128)
        """
        # [batch_size, frames, channels, height, width]->[batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    """
    Audio encoder using 1D convolutional layers.
    
    Extracts acoustic features from audio spectrograms using
    a series of 1D convolutions and projects to a 128-dimensional space.
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Freeze convolutional layers
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Process audio features through CNN.
        
        Args:
            x: Audio spectrogram tensor (batch_size, 1, 64, time)
            
        Returns:
            Audio features (batch_size, 128)
        """
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    """
    Multimodal sentiment analysis model.
    
    Combines text, video, and audio features to predict emotions and sentiment.
    Uses a late fusion approach where modality-specific features are extracted
    independently and then combined for final prediction.
    """
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # Sadness, anger
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Negative, positive, neutral
        )

    def forward(self, text_inputs, video_frames, audio_features):
        """
        Process multimodal inputs and predict emotions and sentiment.
        
        Args:
            text_inputs: Dictionary with 'input_ids' and 'attention_mask'
            video_frames: Video tensor (batch_size, frames, channels, height, width)
            audio_features: Audio spectrogram tensor (batch_size, 1, 64, time)
            
        Returns:
            Dictionary with 'emotions' and 'sentiments' prediction tensors
        """
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)  # [batch_size, 128 * 3]

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }
