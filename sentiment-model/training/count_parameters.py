"""
Parameter Counter for Multimodal Sentiment Analysis Model

This module provides functionality to count and analyze the trainable parameters
in the MultimodalSentimentModel. It breaks down the parameter count by component
(text encoder, video encoder, audio encoder, etc.) to help understand the model's
complexity and resource requirements.

This is useful for:
1. Model complexity analysis
2. Identifying which components have the most parameters
3. Debugging model architecture issues
4. Reporting model specifications

Usage:
    python count_parameters.py
"""

from models import MultimodalSentimentModel


def count_parameters(model):
    """
    Count trainable parameters in a model, broken down by component.

    This function iterates through all parameters in the model and categorizes
    them based on their module names. Only parameters that require gradients
    (trainable parameters) are counted.

    Parameters:
        model: A PyTorch model (typically MultimodalSentimentModel)

    Returns:
        tuple: (params_dict, total_params)
            - params_dict: Dictionary mapping component names to parameter counts
            - total_params: Total number of trainable parameters in the model
    """
    # Initialize dictionary to store parameter counts for each component
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    # Initialize counter for total parameters
    total_params = 0

    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # Only count parameters that require gradients (trainable parameters)
        if param.requires_grad:
            # Count the number of elements in this parameter tensor
            param_count = param.numel()
            # Add to the total count
            total_params += param_count

            # Categorize the parameter based on its name
            if 'text_encoder' in name:
                params_dict['text_encoder'] += param_count
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += param_count
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += param_count
            elif 'fusion_layer' in name:
                params_dict['fusion_layer'] += param_count
            elif 'emotion_classifier' in name:
                params_dict['emotion_classifier'] += param_count
            elif 'sentiment_classifier' in name:
                params_dict['sentiment_classifier'] += param_count

    return params_dict, total_params


if __name__ == "__main__":
    # Create an instance of the model
    model = MultimodalSentimentModel()

    # Count parameters in the model
    param_dics, total_params = count_parameters(model)

    # Display parameter counts by component
    print("Parameter count by component")
    for component, count in param_dics.items():
        # Format with commas for readability (e.g., 1,234,567)
        print(f"{component:20s}: {count:,} parameters")

    # Display total parameter count
    print("\nTotal trainable parameters", f"{total_params:,}")
