import os
import argparse
import torchaudio
import torch
from tqdm import tqdm  # Progress bar library
import json
import sys

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg

# AWS SageMaker environment variables - these help the script run in both local and cloud environments
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")  # Where to save the trained model
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")  # Training data location
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")  # Validation data location
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")  # Test data location

# Memory allocation setting for PyTorch on GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    """
    Parse command line arguments for training.
    
    Returns:
        argparse.Namespace: Object containing all training parameters
    """
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20)  # Number of training cycles
    parser.add_argument("--batch-size", type=int, default=16)  # Number of samples processed at once
    parser.add_argument("--learning-rate", type=float, default=0.001)  # How fast the model learns

    # Data directories - where to find training, validation and test data
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)  # Where to save the model

    return parser.parse_args()


def main():
    """
    Main training function that:
    1. Installs FFmpeg (needed for video processing)
    2. Prepares data loaders for training, validation and testing
    3. Creates and trains the multimodal sentiment model
    4. Evaluates the model on test data
    5. Saves the best model based on validation performance
    """
    # Install FFmpeg - required for video processing
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    # Check available audio backends
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    # Get command line arguments
    args = parse_args()
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Track GPU memory usage if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # Prepare data loaders - these will feed batches of data to our model
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),  # CSV with training data info
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),  # Directory with training videos
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),  # CSV with validation data info
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),  # Directory with validation videos
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),  # CSV with test data info
        test_video_dir=os.path.join(
            args.test_dir, 'output_repeated_splits_test'),  # Directory with test videos
        batch_size=args.batch_size
    )

    # Print paths to confirm data locations
    print(f"""Training CSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {
          os.path.join(args.train_dir, 'train_splits')}""")

    # Create model and trainer
    model = MultimodalSentimentModel().to(device)  # Move model to GPU if available
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')  # Track best validation loss (lower is better)

    # Dictionary to store metrics for later analysis
    metrics_data = {
        "train_losses": [],  # Training losses over time
        "val_losses": [],    # Validation losses over time
        "epochs": []         # Epoch numbers
    }

    # Training loop - runs for the specified number of epochs
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch and get training loss
        train_loss = trainer.train_epoch()
        
        # Evaluate on validation set and get validation loss and metrics
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Store metrics for later analysis
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics in SageMaker format for tracking in AWS
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        # Track GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save the model if it's the best one so far (based on validation loss)
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(
                args.model_dir, "model.pth"))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    # Log final test metrics
    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy",
                "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy",
                "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision",
                "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision",
                "Value": test_metrics["sentiment_precision"]},
        ]
    }))
    
    print("Training completed successfully!")


# Entry point of the script
if __name__ == "__main__":
    main()
