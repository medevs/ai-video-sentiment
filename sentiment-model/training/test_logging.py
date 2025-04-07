"""
Test Logging Module for Multimodal Sentiment Analysis

This module provides a test function to verify the logging functionality
of the MultimodalTrainer class. It creates mock data and a model instance
to test whether metrics are properly logged to TensorBoard.

This is useful for:
1. Verifying that the logging system works correctly
2. Debugging TensorBoard integration issues
3. Testing the trainer without requiring real data

Usage:
    python test_logging.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
from models import MultimodalSentimentModel, MultimodalTrainer


# Create a mock dataset that matches the expected structure
class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return a dictionary with the expected keys
        return {
            'text_inputs': {
                'input_ids': torch.ones(1),
                'attention_mask': torch.ones(1)
            },
            'video_frames': torch.ones(1),
            'audio_features': torch.ones(1),
            'emotion_label': torch.zeros(1, dtype=torch.long).squeeze(),
            'sentiment_label': torch.zeros(1, dtype=torch.long).squeeze()
        }


def test_logging():
    """
    Test the logging functionality of the MultimodalTrainer.

    This function:
    1. Creates mock data batches with dummy tensor values
    2. Initializes a MultimodalSentimentModel and MultimodalTrainer
    3. Logs fake training metrics to verify TensorBoard integration
    4. Logs fake validation metrics with additional performance metrics

    No real training or validation is performed - this only tests the logging
    functionality in isolation.

    Returns:
        None
    """
    # Create a mock dataset that matches the expected structure
    mock_dataset = MockDataset(size=10)

    # Create a DataLoader with our mock dataset
    mock_loader = DataLoader(mock_dataset, batch_size=2)

    # Initialize the model and trainer
    model = MultimodalSentimentModel()

    try:
        # Create the trainer
        trainer = MultimodalTrainer(model, mock_loader, mock_loader)

        # Manually set the current_epoch attribute if it doesn't exist
        if not hasattr(trainer, 'current_epoch'):
            trainer.current_epoch = 1
            print(f"Set current_epoch to {trainer.current_epoch}")

        # Create fake training losses to log
        train_losses = {
            'total': 2.5,      # Combined loss
            'emotion': 1.0,    # Emotion classification loss
            'sentiment': 1.5   # Sentiment classification loss
        }

        # Log the training metrics (this should write to TensorBoard)
        print("Logging training metrics...")
        trainer.log_metrics(train_losses, phase="train")

        # Create fake validation losses and metrics
        val_losses = {
            # Combined loss (lower than training, as expected)
            'total': 1.5,
            'emotion': 0.5,    # Emotion classification loss
            'sentiment': 1.0   # Sentiment classification loss
        }

        # Additional performance metrics for validation
        val_metrics = {
            'emotion_precision': 0.65,    # Precision for emotion classification
            'emotion_accuracy': 0.75,     # Accuracy for emotion classification
            'sentiment_precision': 0.85,  # Precision for sentiment classification
            'sentiment_accuracy': 0.95    # Accuracy for sentiment classification
        }

        # Log the validation metrics (this should write to TensorBoard)
        print("Logging validation metrics...")
        trainer.log_metrics(val_losses, val_metrics, phase="val")

        print("Logging test completed. Check TensorBoard logs in the 'runs' directory.")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Execute the test function when this script is run directly
    test_logging()
