"""
SageMaker Training Script for Multimodal Sentiment Analysis

This script configures and launches a training job on AWS SageMaker cloud platform.
It sets up the necessary configuration for training our sentiment analysis model in the cloud,
including hardware specifications, hyperparameters, and data locations.
"""
from sagemaker.pytorch import PyTorch  # AWS SageMaker's PyTorch training container
# For visualizing training metrics
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    """
    Configure and start a training job on AWS SageMaker.

    This function:
    1. Sets up TensorBoard for monitoring training progress
    2. Configures the PyTorch estimator with hardware and software settings
    3. Specifies hyperparameters for the training job
    4. Starts the training process with data from S3 storage
    """
    # Configure TensorBoard to visualize training metrics
    # TensorBoard is a tool that helps you see how your model is learning
    tensorboard_config = TensorBoardOutputConfig(
        # Where to save TensorBoard logs in S3
        s3_output_path="s3://your-bucket-name/tensorboard",
        # Where logs are stored in the container
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    # Create a PyTorch estimator - this defines how SageMaker will run our training job
    estimator = PyTorch(
        # The training script to run (our local train.py)
        entry_point="train.py",
        source_dir="training",   # Directory containing the training code
        # IAM role with permissions to access AWS resources
        role="your-execution-role-arn",
        framework_version="2.5.1",  # PyTorch version
        py_version="py311",      # Python version
        instance_count=1,        # Number of training instances (machines)
        instance_type="ml.g5.xlarge",  # Type of machine (g5.xlarge has a GPU)
        hyperparameters={
            # Number of samples processed at once (larger than local training)
            "batch-size": 32,
            "epochs": 25         # Number of training cycles
        },
        tensorboard_config=tensorboard_config  # Configuration for TensorBoard
    )

    # Start the training job with data from S3 buckets
    # Each channel (training, validation, test) maps to a directory in the container
    estimator.fit({
        "training": "s3://your-bucket-name/dataset/train",     # Training data location
        "validation": "s3://your-bucket-name/dataset/dev",     # Validation data location
        "test": "s3://your-bucket-name/dataset/test"           # Test data location
    })


# Entry point of the script
if __name__ == "__main__":
    start_training()
