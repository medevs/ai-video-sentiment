# Model Training Process

## What is Model Training?

Model training is teaching an AI system to make predictions by showing it many examples. It's like teaching a child to recognize animals by showing them pictures and saying "this is a cat" or "this is a dog" many times.

## The Training Process

### 1. Data Preparation

Before training starts, we need to prepare our data:

1. **Collect Data**
   - For our project: Video clips from the MELD dataset (from the TV show "Friends")
   - Each clip has a label (emotion and sentiment)

2. **Split Data**
   - Training set (70-80%): Used to teach the model
   - Validation set (10-15%): Used to check progress during training
   - Test set (10-15%): Used only at the end to evaluate the final model

3. **Preprocess Data**
   - Text: Tokenize and convert to vectors using BERT
   - Audio: Convert to mel spectrograms
   - Video: Extract frames and normalize them

### 2. Model Architecture

Our model has several parts:

1. **Text Encoder**
   - Uses BERT to understand dialogue
   - Outputs a vector representing text meaning

2. **Audio Encoder**
   - Analyzes mel spectrograms
   - Outputs a vector representing audio features

3. **Video Encoder**
   - Uses CNN to analyze video frames
   - Outputs a vector representing visual features

4. **Fusion Layer**
   - Combines outputs from all three encoders
   - Can be simple (concatenation) or complex (attention mechanism)

5. **Classification Head**
   - Takes the fused features
   - Outputs probabilities for each emotion/sentiment class

### 3. Training Loop

The actual training happens in a loop:

1. **Forward Pass**
   - Feed a batch of examples through the model
   - Get predictions

2. **Calculate Loss**
   - Compare predictions with actual labels
   - Compute how wrong the model is (loss)

3. **Backward Pass**
   - Calculate how to adjust the model to reduce error
   - Update model weights (parameters)

4. **Validation**
   - Periodically check performance on validation data
   - Make sure the model isn't just memorizing training data

```
┌────────────────┐
│ Start Training │
└───────┬────────┘
        ▼
┌───────────────┐         ┌─────────────┐
│ Load Batch of │         │ Validation  │
│ Training Data │◄────────┤ Every N     │
└───────┬───────┘         │ Epochs      │
        ▼                 └──────┬──────┘
┌───────────────┐                │
│ Forward Pass  │                │
│ (Predictions) │                │
└───────┬───────┘                │
        ▼                        │
┌───────────────┐                │
│ Calculate Loss│                │
└───────┬───────┘                │
        ▼                        │
┌───────────────┐                │
│ Backward Pass │                │
│ (Update Model)│◄───────────────┘
└───────┬───────┘
        ▼
┌───────────────┐     No    ┌─────────────┐
│ Finished All  ├───────────► Continue    │
│ Epochs?       │           │ Training    │
└───────┬───────┘           └─────────────┘
        │ Yes
        ▼
┌───────────────┐
│ Final Testing │
│ on Test Set   │
└───────┬───────┘
        ▼
┌───────────────┐
│ Save Model    │
└───────────────┘
```

### 4. Hyperparameter Tuning

These are settings we can adjust to improve training:

1. **Learning Rate**
   - How big steps the model takes when learning
   - Think of it like adjusting how much a student changes their understanding after each lesson
   - Too high: Like changing your mind completely after every new fact
   - Too low: Like barely changing your opinion even with strong evidence

2. **Batch Size**
   - How many examples the model sees at once
   - Like teaching a class of 16 students versus teaching one student at a time
   - Larger batches: More stable learning but needs more computer memory
   - Smaller batches: Less stable but works on smaller computers

3. **Number of Epochs**
   - How many times the model sees the entire dataset
   - Like how many times a student reviews all their flashcards
   - Too few: Student doesn't learn enough
   - Too many: Student memorizes the cards without understanding the concepts

### 5. Evaluation Metrics

Ways to measure how well our model is doing:

1. **Accuracy**
   - Percentage of correct predictions
   - Simple measurement: "The model got 75% of the answers right"

2. **Precision and Recall**
   - Precision: When the model says "this is happy," how often is it right?
   - Recall: Out of all the truly happy examples, how many did the model find?
   - These are important when some emotions are rare

3. **F1 Score**
   - Combines precision and recall into one number
   - Good for when the dataset has imbalanced classes (some emotions appear more than others)

4. **Confusion Matrix**
   - Shows which emotions get confused with each other
   - Helps identify specific problems (e.g., the model often mistakes surprise for fear)

## Common Challenges in Training

### 1. Overfitting

**Simple explanation:** The model memorizes the training examples instead of learning general patterns.

This is like a student who memorizes test answers but can't solve new problems. They do great on practice tests but fail the real exam.

**Solutions:**
- Add dropout (randomly turn off parts of the model during training)
  - Like forcing a student to solve problems without looking at certain notes
- Use regularization (penalize complex models)
  - Like giving lower grades for overly complicated solutions
- Get more training data
  - Like giving the student more varied practice problems
- Use data augmentation (create variations of training examples)
  - Like showing the same animal from different angles

### 2. Underfitting

**Simple explanation:** The model is too simple to learn the patterns in the data.

This is like trying to teach advanced math using only addition and subtraction. The tools are too basic for the complex task.

**Solutions:**
- Use a more complex model
  - Like giving the student more advanced tools and concepts
- Train longer
  - Like spending more time teaching
- Reduce regularization
  - Like allowing more complex solutions

### 3. Class Imbalance

**Simple explanation:** Some emotions appear much more often than others in the training data.

This is like having a classroom with 30 students who speak English and only 2 who speak French, then wondering why the class doesn't learn French well.

**Solutions:**
- Weight classes differently in loss function
  - Like giving extra credit for correctly answering French questions
- Oversample minority classes
  - Like repeating the French examples more often
- Undersample majority classes
  - Like using fewer English examples

## Transfer Learning

**Simple explanation:** Instead of training everything from scratch, we use pre-trained models.

This is like hiring experienced chefs who already know how to cook, rather than training people with no cooking experience. We just need to teach them our specific recipes.

In our project, we use:

1. **BERT** for text
   - Already trained on millions of documents
   - Knows grammar, context, and meaning
   - We just fine-tune it for emotion recognition

2. **ResNet** for video
   - Already trained to recognize objects in images
   - Knows about edges, shapes, and patterns
   - We fine-tune it to focus on facial expressions

3. **VGGish** for audio
   - Already trained on many audio samples
   - Knows about sound patterns and features
   - We fine-tune it to recognize emotional tones

This saves a lot of time and gives better results, because these models already have a strong foundation of knowledge.
