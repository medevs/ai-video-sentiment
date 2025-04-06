# SageMaker Training Process

## What is SageMaker?

Amazon SageMaker is a cloud service that makes it easy to build, train, and deploy machine learning models. It provides powerful computers with GPUs that can train models much faster than regular computers.

## Setting Up SageMaker for Training

### 1. Creating a Training Job Script

Before we can train on SageMaker, we need to create a script that sets up the training job:

```python
# Example of a SageMaker training job creation script
import boto3
import time

sagemaker_client = boto3.client('sagemaker')

job_name = f"meld-sentiment-{int(time.time())}"  # Unique name using timestamp

training_params = {
    'JobName': job_name,
    'AlgorithmSpecification': {
        'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38',
        'TrainingInputMode': 'File',
        'EnableSageMakerMetricsTimeSeries': True,
    },
    'RoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    'InputDataConfig': [
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://your-bucket/meld-dataset/train/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        },
        {
            'ChannelName': 'validation',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://your-bucket/meld-dataset/dev/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
    ],
    'OutputDataConfig': {
        'S3OutputPath': 's3://your-bucket/meld-model-output/'
    },
    'ResourceConfig': {
        'InstanceType': 'ml.p3.2xlarge',  # GPU instance
        'InstanceCount': 1,
        'VolumeSizeInGB': 100
    },
    'StoppingCondition': {
        'MaxRuntimeInSeconds': 86400  # 24 hours
    },
    'HyperParameters': {
        'epochs': '10',
        'batch-size': '16',
        'learning-rate': '0.0001'
    }
}

response = sagemaker_client.create_training_job(**training_params)
print(f"Training job created: {job_name}")
```

### 2. Preparing the Training Script

SageMaker expects a specific structure for the training script. Here's a simplified example:

```python
# train.py - SageMaker training script
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Parse arguments provided by SageMaker
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--learning-rate', type=float, default=0.0001)
# SageMaker specific arguments
parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
args = parser.parse_args()

# Set up TensorBoard for logging
writer = SummaryWriter(log_dir='/opt/ml/output/tensorboard')

# Load dataset
train_dataset = MELDDataset(os.path.join(args.train, 'train_sent_emo.csv'), 
                           os.path.join(args.train, 'train_splits'))
val_dataset = MELDDataset(os.path.join(args.validation, 'dev_sent_emo.csv'), 
                         os.path.join(args.validation, 'dev_splits_complete'))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                       collate_fn=collate_fn)

# Create model
model = MultimodalSentimentModel()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch['sentiment_label'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            loss = criterion(outputs, batch['sentiment_label'])
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch['sentiment_label'].size(0)
            correct += (predicted == batch['sentiment_label']).sum().item()
    
    # Log metrics
    writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
    writer.add_scalar('Loss/validation', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/validation', 100 * correct / total, epoch)
    
    print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*correct/total:.2f}%')

# Save model
torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
print('Training complete!')
```

## Class Weights for Imbalanced Data

One important aspect of training is handling class imbalances. In the MELD dataset, some emotions (like "neutral") appear much more often than others (like "fear" or "disgust").

To fix this, we use class weights:

```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get all labels
all_labels = [data['sentiment_label'] for data in train_dataset]
unique_labels = np.unique(all_labels)

# Compute weights (less frequent classes get higher weights)
class_weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Use weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This gives more importance to rare classes during training, helping the model learn them better.

## Monitoring Training with TensorBoard

TensorBoard is a tool that helps visualize training progress. It shows:

1. **Loss curves** - How the error decreases over time
2. **Accuracy** - How the model's predictions improve
3. **Model architecture** - Visual representation of the model
4. **Confusion matrix** - Which classes get confused with each other

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  SageMaker  │─────────►  TensorBoard Logs     │  │
│  │  Training   │         │                       │  │
│  │             │         │  ┌─────────────────┐  │  │
│  └─────────────┘         │  │ Loss Graph      │  │  │
│                          │  └─────────────────┘  │  │
│                          │                       │  │
│                          │  ┌─────────────────┐  │  │
│                          │  │ Accuracy Graph  │  │  │
│                          │  └─────────────────┘  │  │
│                          │                       │  │
│                          └───────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

To view TensorBoard logs from SageMaker:

1. Download the logs from S3
2. Run TensorBoard locally:
   ```
   tensorboard --logdir=./logs
   ```
3. Open http://localhost:6006 in your browser

## Counting Model Parameters

It's important to know how big your model is. More parameters means:
- More powerful model (can learn more complex patterns)
- More memory needed
- Longer training time
- Potential for overfitting

Here's how to count parameters:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f'Total trainable parameters: {total_params:,}')
```

A typical multimodal model might have:
- Text encoder: ~110 million parameters (BERT base)
- Video encoder: ~25 million parameters (ResNet-50)
- Audio encoder: ~5 million parameters
- Fusion layers: ~1 million parameters

## FFMPEG Installation on AWS

For audio processing, we need to install ffmpeg on our SageMaker instance. This is done in the Dockerfile or setup script:

```bash
# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
```

## Common Training Issues

### 1. Out of Memory Errors

**Problem:** GPU runs out of memory during training
**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Use a smaller model
- Use mixed precision training

### 2. Slow Training

**Problem:** Training takes too long
**Solutions:**
- Use a more powerful instance (e.g., ml.p3.8xlarge instead of ml.p3.2xlarge)
- Optimize data loading (use more workers)
- Use a smaller subset of data for initial experiments

### 3. Poor Performance

**Problem:** Model doesn't learn well
**Solutions:**
- Check for data issues
- Try different learning rates
- Use class weights for imbalanced data
- Add regularization to prevent overfitting
- Try different model architectures
