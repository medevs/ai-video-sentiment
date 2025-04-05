"""
MELD Dataset Module

This module provides functionality to load and process the MELD (Multimodal EmotionLines Dataset) 
for multimodal sentiment and emotion analysis. It handles video frames, audio features, and text data.
"""

# Import necessary libraries
# For creating dataset and data loading utilities
from torch.utils.data import Dataset, DataLoader
import pandas as pd  # For handling CSV data
import torch.utils.data.dataloader  # For data loading functionality
from transformers import AutoTokenizer  # For text tokenization
import os  # For file and directory operations
import cv2  # For video processing
import numpy as np  # For numerical operations
import torch  # PyTorch deep learning framework
import subprocess  # For running external commands (ffmpeg)
import torchaudio  # For audio processing

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDataset(Dataset):
    """
    Dataset class for the MELD dataset that handles multimodal data (text, video, audio).
    Inherits from PyTorch's Dataset class to enable data loading with DataLoader.
    """

    def __init__(self, csv_path, video_dir):
        """
        Initialize the MELD dataset.

        Args:
            csv_path (str): Path to the CSV file containing utterance data
            video_dir (str): Directory containing video files
        """
        # Load the CSV data into a pandas DataFrame
        self.data = pd.read_csv(csv_path)

        # Store the video directory path
        self.video_dir = video_dir

        # Initialize the BERT tokenizer for processing text
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Define mapping from emotion labels to numeric indices
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }

        # Define mapping from sentiment labels to numeric indices
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }

    def _load_video_frames(self, video_path):
        """
        Load and preprocess video frames from a video file.

        Args:
            video_path (str): Path to the video file

        Returns:
            torch.FloatTensor: Tensor of processed video frames with shape [frames, channels, height, width]

        Raises:
            ValueError: If video cannot be loaded or processed
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            # Check if video file was opened successfully
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try to read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Extract up to 30 frames from the video
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:  # End of video
                    break

                # Resize frame to 224x224 (standard size for many vision models)
                frame = cv2.resize(frame, (224, 224))
                # Normalize pixel values to range [0,1]
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            # Always release the video capture object
            cap.release()

        # Check if any frames were extracted
        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames to ensure exactly 30 frames
        if len(frames) < 30:
            # If fewer than 30 frames, pad with zero frames
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            # If more than 30 frames, keep only the first 30
            frames = frames[:30]

        # Convert frames to tensor and rearrange dimensions
        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width] (PyTorch format)
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        """
        Extract and process audio features from a video file.

        Args:
            video_path (str): Path to the video file

        Returns:
            torch.Tensor: Mel spectrogram features of the audio

        Raises:
            ValueError: If audio cannot be extracted or processed
        """
        # Define path for temporary WAV file
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            # Extract audio from video using ffmpeg
            subprocess.run([
                'ffmpeg',
                '-i', video_path,  # Input video
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono audio
                audio_path  # Output path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed to 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Create mel spectrogram transformer
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,  # Number of mel bands
                n_fft=1024,  # FFT window size
                hop_length=512  # Hop length
            )

            # Generate mel spectrogram
            mel_spec = mel_spectrogram(waveform)

            # Normalize the spectrogram (z-score normalization)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # Ensure consistent size (300 time steps)
            if mel_spec.size(2) < 300:
                # Pad if too short
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                # Truncate if too long
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            # Clean up temporary WAV file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing text, video, audio features and labels

        Note:
            Returns None if an error occurs during processing
        """
        # Convert tensor index to integer if needed
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        # Get the row from the DataFrame
        row = self.data.iloc[idx]

        try:
            # Construct video filename from dialogue and utterance IDs
            video_filename = f"""dia{row['Dialogue_ID']}_utt{
                row['Utterance_ID']}.mp4"""

            # Create full path to the video file
            path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(path)

            # Check if video file exists
            if video_path_exists == False:
                raise FileNotFoundError(f"No video found for filename: {path}")

            # Tokenize the utterance text for the language model
            text_inputs = self.tokenizer(row['Utterance'],
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,
                                         return_tensors='pt')

            # Load video frames and extract audio features
            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            # Convert emotion and sentiment labels to numeric indices
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            # Return a dictionary with all features and labels
            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            # Log error and return None for this sample
            print(f"Error processing {path}: {str(e)}")
            return None


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle None values in batch.

    Args:
        batch (list): List of samples from the dataset

    Returns:
        dict: Collated batch with None values filtered out
    """
    # Filter out None samples
    batch = list(filter(None, batch))
    # Use default collate function from PyTorch
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    """
    Prepare DataLoaders for training, validation, and testing.

    Args:
        train_csv (str): Path to training CSV file
        train_video_dir (str): Directory containing training videos
        dev_csv (str): Path to validation CSV file
        dev_video_dir (str): Directory containing validation videos
        test_csv (str): Path to test CSV file
        test_video_dir (str): Directory containing test videos
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.

    Returns:
        tuple: (train_loader, dev_loader, test_loader) - DataLoaders for each split
    """
    # Create dataset objects
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    # Create DataLoader for training data
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,  # Shuffle training data
                              collate_fn=collate_fn)  # Use custom collate function

    # Create DataLoader for validation data
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


# This block runs when the script is executed directly (not imported)
if __name__ == "__main__":
    # Create DataLoaders with default paths
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )

    # Test the first batch from the training loader
    for batch in train_loader:
        # Print the contents of the batch to verify
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break  # Only process one batch
