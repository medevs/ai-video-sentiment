"""
Multimodal Sentiment Analysis Inference Pipeline

This module provides the inference pipeline for the multimodal sentiment analysis model,
including video processing, audio processing, and model prediction functions.
The pipeline is designed to work with AWS SageMaker but can also be used locally.
"""

import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import sys
import json
import boto3
import tempfile

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def install_ffmpeg():
    """
    Install FFmpeg for video and audio processing.
    
    Attempts to install FFmpeg through pip and by downloading static binaries.
    Returns True if installation is successful, False otherwise.
    """
    print("Starting Ffmpeg installation...")

    # Upgrade pip to ensure compatibility with latest packages
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])

    # Upgrade setuptools to ensure compatibility with latest packages
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "setuptools"])

    try:
        # Try installing ffmpeg via pip
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        # If pip installation fails, try downloading and installing static binaries
        # Download static FFmpeg binary for Linux
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        # Extract the downloaded archive
        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        # Find the ffmpeg executable in the extracted files
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()

        # Copy the executable to a system path
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        # Make the executable file executable
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    try:
        # Verify FFmpeg installation by checking version
        result = subprocess.run(["ffmpeg", "-version"],
                                capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False


class VideoProcessor:
    """
    Process video files for model input.
    
    Extracts frames from videos, resizes them to 224x224, and normalizes pixel values.
    Ensures a fixed number of frames (30) through padding or truncation.
    """
    def process_video(self, video_path):
        """
        Extract and process frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tensor of processed video frames with shape [frames, channels, height, width]
            
        Raises:
            ValueError: If video cannot be read or no frames can be extracted
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Extract frames from the video (up to 30)
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to 224x224 (standard input size for many vision models)
                frame = cv2.resize(frame, (224, 224))
                # Normalize pixel values to [0, 1] range
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    """
    Process audio from video files for model input.
    
    Extracts audio from videos, converts to mel spectrograms,
    and ensures consistent dimensions through padding or truncation.
    """
    def extract_features(self, video_path, max_length=300):
        """
        Extract and process audio features from a video file.
        
        Args:
            video_path: Path to the video file
            max_length: Maximum spectrogram length (default: 300)
            
        Returns:
            Mel spectrogram tensor with shape [1, 64, max_length]
            
        Raises:
            ValueError: If audio extraction or processing fails
        """
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            # Extract audio from video using FFmpeg
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load audio waveform
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample audio to 16 kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Compute mel spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # Pad or truncate spectrogram to max_length
            if mel_spec.size(2) < max_length:
                padding = max_length - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_length]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    """
    Process video segments for utterance-level analysis.
    
    Extracts video segments based on timestamps and processes
    both video frames and audio features for model input.
    """
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir="/tmp"):
        """
        Extract a segment from a video based on start and end times.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            temp_dir: Directory for temporary files (default: /tmp)
            
        Returns:
            Path to the extracted video segment
            
        Raises:
            ValueError: If segment extraction fails
        """
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(
            temp_dir, f"segment_{start_time}_{end_time}.mp4")

        # Use FFmpeg to extract the segment from the video file
        # -i: input file
        # -ss: start time
        # -to: end time
        # -c:v libx264: use H.264 codec for video
        # -c:a aac: use AAC codec for audio
        # -y: overwrite output file if it exists
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Verify the segment was created successfully
        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed: " + segment_path)

        return segment_path


def download_from_s3(s3_uri):
    """
    Download a video file from S3.
    
    Args:
        s3_uri: S3 URI in the format s3://bucket/key
        
    Returns:
        Local path to the downloaded file
    """
    s3_client = boto3.client("s3")
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name


def input_fn(request_body, request_content_type):
    """
    SageMaker input function to process incoming requests.
    
    Args:
        request_body: Request body containing video path
        request_content_type: Content type of the request
        
    Returns:
        Dictionary with local video path
        
    Raises:
        ValueError: If content type is not supported
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        s3_uri = input_data['video_path']
        local_path = download_from_s3(s3_uri)
        return {"video_path": local_path}
    raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction, response_content_type):
    """
    SageMaker output function to format predictions.
    
    Args:
        prediction: Model prediction
        response_content_type: Content type of the response
        
    Returns:
        Formatted prediction as JSON string
        
    Raises:
        ValueError: If content type is not supported
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")


def model_fn(model_dir):
    """
    SageMaker model loading function.
    
    Loads the multimodal sentiment model, tokenizer, and transcriber.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Dictionary with model, tokenizer, transcriber, and device
        
    Raises:
        RuntimeError: If FFmpeg installation fails
        FileNotFoundError: If model file is not found
    """
    # Load the model for inference
    if not install_ffmpeg():
        raise RuntimeError(
            "FFmpeg installation failed - required for inference")

    # Determine whether to use GPU (CUDA) or CPU for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    # Try to find model file in different possible locations
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        # Alternative path for SageMaker deployment structure
        model_path = os.path.join(model_dir, "model", 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file not found in path " + model_path)

    print("Loading model from path: " + model_path)
    # Load pre-trained weights with compatibility for different devices
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    # Set model to evaluation mode (disables dropout, batch normalization behaves differently)
    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        # Load Whisper model for speech transcription, ensuring it's on the right device
        'transcriber': whisper.load_model(
            "base",
            device="cpu" if device.type == "cpu" else device,
        ),
        'device': device
    }


def predict_fn(input_data, model_dict):
    """
    Main prediction function for sentiment analysis.
    
    Transcribes video, processes segments, and runs inference on each segment.
    
    Args:
        input_data: Dictionary with video path
        model_dict: Dictionary with model and related components
        
    Returns:
        Dictionary with utterance-level predictions including:
        - Start and end times
        - Transcribed text
        - Top emotions with confidence scores
        - Top sentiments with confidence scores
    """
    # Extract components from the model dictionary
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    # Transcribe the entire video using Whisper, with word-level timestamps
    result = model_dict['transcriber'].transcribe(
        video_path, word_timestamps=True)

    # Initialize the processor for handling video segments
    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    # Process each utterance segment identified by the transcriber
    for segment in result["segments"]:
        try:
            # Extract the specific time segment from the video
            segment_path = utterance_processor.extract_segment(
                video_path,
                segment["start"],
                segment["end"]
            )

            # Process video frames from the segment
            video_frames = utterance_processor.video_processor.process_video(
                segment_path)
            # Extract audio features (mel spectrograms) from the segment
            audio_features = utterance_processor.audio_processor.extract_features(
                segment_path)
            # Tokenize the transcribed text for the BERT model
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",  # Pad to max_length for consistent tensor sizes
                truncation=True,       # Truncate if text is too long
                max_length=128,        # Maximum sequence length for the model
                return_tensors="pt"    # Return PyTorch tensors
            )

            # Move to device - transfers tensors to GPU if available, otherwise keeps on CPU
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            # Add batch dimension (1) and move to appropriate device
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Get predictions using inference mode to disable gradient calculation for efficiency
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                # Apply softmax to convert logits to probabilities (0-1 range)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(
                    outputs["sentiments"], dim=1)[0]

                # Get top 3 emotions and sentiments with their confidence scores
                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(
                    sentiment_probs, 3)

            # Create structured prediction output for this segment
            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                # Map numerical indices to human-readable emotion labels with confidence scores
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(emotion_indices, emotion_values)
                ],
                # Map numerical indices to human-readable sentiment labels with confidence scores
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })

        except Exception as e:
            print("Segment failed inference: " + str(e))

        finally:
            # Cleanup
            if os.path.exists(segment_path):
                os.remove(segment_path)
    return {"utterances": predictions}


# def process_local_video(video_path, model_dir="model_normalized"):
#     model_dict = model_fn(model_dir)

#     input_data = {'video_path': video_path}

#     predictions = predict_fn(input_data, model_dict)

#     for utterance in predictions["utterances"]:
#         print("\nUtterance:")
#         print(f"""Start: {utterance['start_time']}s, End: {
#               utterance['end_time']}s""")
#         print(f"Text: {utterance['text']}")
#         print("\n Top Emotions:")
#         for emotion in utterance['emotions']:
#             print(f"{emotion['label']}: {emotion['confidence']:.2f}")
#         print("\n Top Sentiments:")
#         for sentiment in utterance['sentiments']:
#             print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")
#         print("-"*50)


# if __name__ == "__main__":
#     process_local_video("./dia2_utt3.mp4")
