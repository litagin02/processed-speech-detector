import torch
from models import AudioClassifier

dependencies = ["torch", "librosa"]


def processed_speech_detector(task="freq"):
    """
    Load the processed speech detection model.

    This function loads a model trained to detect processed speech. It supports two tasks:
    - 'freq': Detect whether the speech has been passed through low-pass, high-pass, or band-pass filters.
    - 'reverb': Detect whether the speech has reverb applied.

    In both cases, the model performs binary classification where 1 indicates processed speech and 0 indicates unprocessed speech.

    Args:
        task (str): The task for which the model is loaded. Options are 'freq' for filter detection and 'reverb' for reverb detection.

    Returns:
        torch.nn.Module: The loaded audio classifier model.
    """
    model = AudioClassifier()
    if task == "freq":
        model.load_state_dict(
            torch.load("pretrained/freq_model.pth", map_location="cpu")
        )
    elif task == "reverb":
        model.load_state_dict(
            torch.load("pretrained/reverb_model.pth", map_location="cpu")
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    model.eval()
    return model
