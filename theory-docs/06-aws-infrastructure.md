# AWS Infrastructure for AI Model Deployment

## What is AWS?

AWS (Amazon Web Services) is a cloud platform that provides services for building and deploying applications. In this project, we use AWS to train and deploy our AI model.

## Key AWS Services Used in This Project

### 1. Amazon S3 (Simple Storage Service)

**What it is:** Cloud storage for files
**How we use it:**
- Store our dataset (video files, CSV files)
- Store our trained model files
- Store uploaded videos for inference

```
┌─────────────┐
│  S3 Bucket  │
├─────────────┤
│ - Dataset   │
│ - Models    │
│ - Uploads   │
└─────────────┘
```

### 2. Amazon SageMaker

**What it is:** Service for building, training, and deploying machine learning models
**How we use it:**
- Train our multimodal model on powerful GPU instances
- Deploy our model as an endpoint for inference
- Monitor model performance

```
┌───────────────────────┐
│  SageMaker            │
├───────────────────────┤
│ ┌─────────────────┐   │
│ │ Training Job    │   │
│ └─────────────────┘   │
│                       │
│ ┌─────────────────┐   │
│ │ Model Endpoint  │   │
│ └─────────────────┘   │
└───────────────────────┘
```

### 3. Amazon EC2 (Elastic Compute Cloud)

**What it is:** Virtual servers in the cloud
**How we use it:**
- Set up instances to download and prepare our dataset
- Run preprocessing tasks that don't need GPUs
- Host our web application

### 4. IAM (Identity and Access Management)

**What it is:** Manages access to AWS services and resources
**How we use it:**
- Create roles for SageMaker to access S3
- Create users for our application to invoke the model endpoint
- Set up security policies

## Training Infrastructure

For training our model, we use this setup:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  S3 Bucket  │◄────────┤  SageMaker Training   │  │
│  │  (Dataset)  │         │  Job (GPU Instance)   │  │
│  │             │─────────►                       │  │
│  └─────────────┘         └───────────────────────┘  │
│         ▲                           │               │
│         │                           ▼               │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  EC2        │         │  S3 Bucket           │  │
│  │  Instance   │         │  (Trained Model)     │  │
│  │             │         │                       │  │
│  └─────────────┘         └───────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Training Process Steps:

1. **Upload Dataset to S3**
   - Use EC2 instance to download and upload the MELD dataset to S3

2. **Create Training Job**
   - Configure SageMaker training job with:
     - Instance type (e.g., ml.p3.2xlarge with GPU)
     - Input data location (S3 path)
     - Output model location (S3 path)
     - Training script location

3. **Run Training**
   - SageMaker pulls data from S3
   - Runs our training script
   - Saves the trained model back to S3

4. **Monitor Training**
   - Use TensorBoard for visualizing training progress
   - Check logs for errors or issues

## Deployment Infrastructure

For deploying our model, we use this setup:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  S3 Bucket  │◄────────┤  SageMaker Endpoint   │  │
│  │  (Model)    │─────────►                       │  │
│  │             │         └───────────────────────┘  │
│  └─────────────┘                   ▲                │
│                                    │                │
│  ┌─────────────┐         ┌────────┴──────────┐     │
│  │             │         │                   │     │
│  │  S3 Bucket  │◄────────┤  Web Application  │     │
│  │  (Uploads)  │─────────►  (Next.js)        │     │
│  │             │         │                   │     │
│  └─────────────┘         └───────────────────┘     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Deployment Process Steps:

1. **Create Model Endpoint**
   - Configure SageMaker endpoint with:
     - Model artifacts location (S3 path)
     - Instance type (e.g., ml.g4dn.xlarge)
     - Inference script

2. **Set Up Web Application**
   - Deploy Next.js application
   - Configure it to:
     - Generate S3 signed URLs for video uploads
     - Call SageMaker endpoint for inference
     - Display results to users

3. **Create IAM User for Endpoint Invocation**
   - Create a user with permissions to:
     - Upload to S3
     - Invoke SageMaker endpoint
   - Generate access keys for the web application

## Cost Considerations

AWS services cost money, with different pricing models:

1. **S3 Storage**
   - Pay for the amount of data stored
   - Pay for data transfer (downloading)

2. **SageMaker**
   - Training: Pay per hour of instance usage
   - Endpoint: Pay per hour the endpoint is running
   - GPU instances are much more expensive than CPU instances

3. **EC2**
   - Pay per hour of instance usage
   - Different instance types have different costs

To save money:
- Stop endpoints when not in use
- Use spot instances for training when possible
- Monitor usage and set up billing alerts

## Common Issues and Solutions

### 1. Timeout Issues

**Problem:** Long inference times can cause timeouts
**Solution:**
- Increase timeout settings in the web application
- Optimize inference code for faster processing
- Use asynchronous processing for long-running tasks

### 2. Memory Issues

**Problem:** Running out of memory during training or inference
**Solution:**
- Use larger instance types
- Reduce batch size
- Optimize data loading

### 3. Permission Issues

**Problem:** Access denied errors
**Solution:**
- Check IAM roles and policies
- Ensure proper cross-service permissions
- Use the principle of least privilege
