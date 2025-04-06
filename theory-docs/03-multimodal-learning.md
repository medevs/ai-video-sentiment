# Multimodal Learning

## What is Multimodal Learning?

Multimodal learning is when AI uses multiple types of information (or "modes") at the same time. In our project, we use three modes:

1. **Text** - The words people say (dialogue)
2. **Audio** - How people say those words (tone, volume)
3. **Video** - What people look like when speaking (facial expressions)

This is similar to how humans understand emotions - we don't just listen to words, we also pay attention to how someone sounds and what their face is doing.

## Why Use Multiple Modes?

Using multiple modes gives better results because:

1. **More Complete Information**
   - Text alone might miss sarcasm
   - Audio can reveal if someone sounds angry even with neutral words
   - Video can show facial expressions that contradict words

2. **Handling Missing Information**
   - If audio is noisy, the model can rely more on text and video
   - If someone is off-camera, the model can still use audio and text

## How Each Mode is Processed

### 1. Text Processing

**Input:** Dialogue text (what people say)
**Processing:**
- Text is broken into tokens (words and parts of words)
- BERT converts these tokens into number vectors
- These vectors capture the meaning of the words

**Example in JavaScript-like code:**
```javascript
// This is simplified - real NLP is more complex
function processText(text) {
  const tokens = tokenize(text);  // ["I", "am", "happy"]
  const vectors = bertModel.encode(tokens);
  return vectors;  // Returns number arrays
}
```

### 2. Audio Processing

**Input:** Sound from the video
**Processing:**
- Audio is extracted from video files
- Converted to a mel spectrogram (a visual representation of sound)
- This shows patterns in pitch, volume, and tone

**What a mel spectrogram looks like:**
```
Time →
↑
F  ██░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░
r  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░
e  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░
q  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
↓  ████████████░░░░░░░░░░░░░░░░░░░░░░
```
(Dark areas show where sound energy is stronger)

### 3. Video Processing

**Input:** Video frames (images)
**Processing:**
- Extract 30 frames from each video clip
- Resize all frames to the same size (224x224 pixels)
- Normalize pixel values (convert from 0-255 to 0-1)
- Use a CNN (Convolutional Neural Network) to analyze facial expressions

## Fusion: Combining the Modes

Fusion means mixing different types of information together. We need to combine text, audio, and video data to make one final prediction.

### 1. Early Fusion

**Simple explanation:** This is like mixing all ingredients at the beginning of cooking.

Imagine making a soup. Early fusion puts all ingredients (vegetables, meat, spices) in the pot at the same time and cooks them together.

In our AI project, this means combining the raw text, audio, and video data before processing them. The problem is that text, audio, and video are very different - like trying to mix oil and water.

### 2. Late Fusion

**Simple explanation:** This is like cooking different dishes separately and then serving them on the same plate.

Imagine making a meal with rice, chicken, and vegetables. Late fusion cooks each part separately and only puts them together on the plate at the end.

In our AI project, this means:
1. Process text with BERT to get text features
2. Process audio to get audio features
3. Process video to get video features
4. At the very end, combine their final results

This works better than early fusion, but it might miss connections between the different types of data.

### 3. Hybrid Fusion (What We Use)

**Simple explanation:** This is like partially cooking ingredients separately, then combining them to finish cooking together.

Imagine making a stir-fry. You cook the meat halfway, cook the vegetables separately, and then combine them to finish cooking together so the flavors mix.

In our AI project, this means:
1. Process each type of data (text, audio, video) partially
2. Combine their intermediate features
3. Process the combined features together for the final prediction

This gets the benefits of both early and late fusion.

## Architecture Diagram

```
┌─────────┐    ┌───────────────┐
│  Text   │───►│  BERT Model   │──┐
└─────────┘    └───────────────┘  │
                                  │  ┌─────────────┐    ┌─────────────┐
┌─────────┐    ┌───────────────┐  ├─►│             │    │             │
│  Audio  │───►│ Mel Spectrogram│──┤  │   Fusion   │───►│ Classifier  │
└─────────┘    └───────────────┘  │  │   Layer     │    │             │
                                  │  └─────────────┘    └─────────────┘
┌─────────┐    ┌───────────────┐  │
│  Video  │───►│  CNN Model    │──┘
└─────────┘    └───────────────┘
```

This diagram shows:
1. **Left side**: The three types of input data (text, audio, video)
2. **Middle-left**: Each type of data is processed by its own special model
   - Text goes through BERT (which understands language)
   - Audio is converted to a mel spectrogram (a visual representation of sound)
   - Video goes through a CNN (which understands images)
3. **Middle-right**: The Fusion Layer combines all the processed information
4. **Right side**: The Classifier makes the final decision about the emotion/sentiment

It's like having three experts (language expert, sound expert, image expert) who each analyze their part, then come together to make a final decision.
