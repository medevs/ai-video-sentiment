# Web Application Architecture

## Overview

The web application is the user interface for our video sentiment analysis system. It allows users to:
1. Upload videos
2. Process them with our AI model
3. View the sentiment and emotion results

This application is built using modern web technologies that will be familiar to JavaScript developers.

## Technology Stack

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  Next.js 15 │         │  React                │  │
│  │  Framework  │─────────►  UI Components        │  │
│  │             │         │                       │  │
│  └─────────────┘         └───────────────────────┘  │
│         │                           │               │
│         │                           │               │
│         ▼                           ▼               │
│  ┌─────────────┐         ┌───────────────────────┐  │
│  │             │         │                       │  │
│  │  API Routes │         │  Tailwind CSS        │  │
│  │  (Backend)  │         │  (Styling)           │  │
│  │             │         │                       │  │
│  └─────────────┘         └───────────────────────┘  │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────────────────┐    │
│  │                                             │    │
│  │  Database (User data, video results)        │    │
│  │                                             │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1. Next.js 15

Next.js is a React framework that provides:
- Server-side rendering
- API routes (backend functionality)
- File-based routing
- Easy deployment

### 2. React

React is a JavaScript library for building user interfaces:
- Component-based architecture
- Virtual DOM for efficient updates
- Hooks for state management

### 3. Tailwind CSS

Tailwind is a utility-first CSS framework:
- Pre-defined classes for styling
- Responsive design
- Customizable themes

### 4. Database

The application uses a database to store:
- User information
- Video metadata
- Analysis results

## Application Structure

```
app/
├── api/                   # Backend API routes
│   ├── auth/              # Authentication endpoints
│   ├── s3/                # S3 signed URL generation
│   └── inference/         # Model inference endpoint
├── components/            # Reusable UI components
│   ├── ui/                # Basic UI elements
│   ├── dashboard/         # Dashboard components
│   └── upload/            # Video upload components
├── lib/                   # Utility functions
│   ├── aws.js             # AWS SDK helpers
│   ├── database.js        # Database connection
│   └── auth.js            # Authentication helpers
└── app/                   # Pages and routes
    ├── dashboard/         # Dashboard page
    ├── docs/              # Documentation pages
    └── auth/              # Authentication pages
```

## Authentication Flow

The application uses authentication to:
- Secure user data
- Track usage
- Limit API calls

```
┌──────────┐     ┌────────────┐     ┌──────────────┐
│          │     │            │     │              │
│  User    │────►│  Sign Up/  │────►│  Dashboard   │
│          │     │  Login     │     │              │
└──────────┘     └────────────┘     └──────────────┘
                       │                   │
                       ▼                   ▼
                 ┌────────────┐     ┌──────────────┐
                 │            │     │              │
                 │  Database  │◄────┤  Protected   │
                 │  (Users)   │     │  API Routes  │
                 │            │     │              │
                 └────────────┘     └──────────────┘
```

## Video Processing Flow

Here's how a video gets processed:

1. **Upload Process**
   ```
   ┌──────────┐     ┌────────────┐     ┌──────────────┐
   │          │     │            │     │              │
   │  User    │────►│  Frontend  │────►│  API Route   │
   │          │     │  Upload UI │     │  /api/s3     │
   └──────────┘     └────────────┘     └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │              │
                                        │  S3 Bucket   │
                                        │  (Videos)    │
                                        │              │
                                        └──────────────┘
   ```

2. **Inference Process**
   ```
   ┌──────────┐     ┌────────────┐     ┌──────────────┐
   │          │     │            │     │              │
   │  User    │────►│  Frontend  │────►│  API Route   │
   │          │     │  Dashboard │     │  /api/infer  │
   └──────────┘     └────────────┘     └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │              │
                                        │  SageMaker   │
                                        │  Endpoint    │
                                        │              │
                                        └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │              │
                                        │  Database    │
                                        │  (Results)   │
                                        │              │
                                        └──────────────┘
   ```

## Key API Endpoints

### 1. S3 Signed URL Endpoint

This endpoint generates a URL that allows direct upload to S3:

```javascript
// api/s3/index.js
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

export async function POST(request) {
  const { filename, contentType } = await request.json();
  
  // Create a unique file key
  const key = `uploads/${Date.now()}-${filename}`;
  
  // Create S3 client
  const s3Client = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    },
  });
  
  // Create command
  const command = new PutObjectCommand({
    Bucket: process.env.S3_BUCKET_NAME,
    Key: key,
    ContentType: contentType,
  });
  
  // Generate signed URL (valid for 5 minutes)
  const signedUrl = await getSignedUrl(s3Client, command, {
    expiresIn: 300,
  });
  
  return Response.json({
    uploadUrl: signedUrl,
    key: key,
  });
}
```

### 2. Inference Endpoint

This endpoint calls the SageMaker model to analyze a video:

```javascript
// api/inference/index.js
import { SageMakerRuntimeClient, InvokeEndpointCommand } from "@aws-sdk/client-sagemaker-runtime";
import { prisma } from "@/lib/prisma";

export async function POST(request) {
  const { videoKey } = await request.json();
  
  // Create SageMaker client
  const sagemakerClient = new SageMakerRuntimeClient({
    region: process.env.AWS_REGION,
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    },
  });
  
  // Create request payload
  const payload = {
    video_key: videoKey,
    bucket: process.env.S3_BUCKET_NAME,
  };
  
  // Create command
  const command = new InvokeEndpointCommand({
    EndpointName: process.env.SAGEMAKER_ENDPOINT_NAME,
    ContentType: "application/json",
    Body: JSON.stringify(payload),
  });
  
  try {
    // Call SageMaker endpoint
    const response = await sagemakerClient.send(command);
    
    // Parse response
    const responseBody = JSON.parse(Buffer.from(response.Body).toString());
    
    // Save result to database
    const result = await prisma.videoAnalysis.create({
      data: {
        videoKey,
        sentiment: responseBody.sentiment,
        emotion: responseBody.emotion,
        confidence: responseBody.confidence,
        userId: request.auth.userId,
      },
    });
    
    return Response.json(responseBody);
  } catch (error) {
    console.error("Inference error:", error);
    return Response.json({ error: "Failed to process video" }, { status: 500 });
  }
}
```

## Database Schema

The database stores information about users and their video analyses:

```
┌─────────────────┐       ┌─────────────────────┐
│ User            │       │ VideoAnalysis        │
├─────────────────┤       ├─────────────────────┤
│ id              │       │ id                  │
│ name            │       │ videoKey            │
│ email           │◄──────┤ userId              │
│ createdAt       │       │ sentiment           │
│ updatedAt       │       │ emotion             │
└─────────────────┘       │ confidence          │
                          │ createdAt           │
                          │ updatedAt           │
                          └─────────────────────┘
```

## Dashboard UI

The dashboard is where users see their video analyses:

```
┌─────────────────────────────────────────────────────┐
│ Dashboard                                          ▲ │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Video 1     │  │ Video 2     │  │ Video 3     │  │
│  │             │  │             │  │             │  │
│  │ Sentiment:  │  │ Sentiment:  │  │ Sentiment:  │  │
│  │ Positive    │  │ Negative    │  │ Neutral     │  │
│  │             │  │             │  │             │  │
│  │ Emotion:    │  │ Emotion:    │  │ Emotion:    │  │
│  │ Joy         │  │ Anger       │  │ Neutral     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │                                             │    │
│  │  Upload New Video                           │    │
│  │                                             │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Deployment

The web application can be deployed to various platforms:

1. **Vercel** (easiest for Next.js)
   - Connect to GitHub repository
   - Automatic deployments on push
   - Environment variables for AWS credentials

2. **AWS Amplify**
   - Integrated with other AWS services
   - CI/CD pipeline
   - Environment variables management

3. **AWS EC2**
   - More control over the server
   - Requires more setup
   - Can use Docker for containerization

## Common Issues and Solutions

### 1. Timeout Issues

**Problem:** SageMaker inference takes too long, causing API timeout
**Solutions:**
- Increase API timeout settings
- Use a background job queue
- Implement a webhook callback system

### 2. CORS Issues

**Problem:** Browser blocks requests due to Cross-Origin Resource Sharing
**Solutions:**
- Configure proper CORS headers on API routes
- Ensure S3 bucket has correct CORS configuration

### 3. Authentication Issues

**Problem:** Users can't log in or access protected routes
**Solutions:**
- Check authentication configuration
- Verify JWT tokens are properly validated
- Ensure database connections are working

### 4. Performance Issues

**Problem:** Dashboard loads slowly with many videos
**Solutions:**
- Implement pagination
- Use server-side rendering
- Optimize database queries
- Implement caching
