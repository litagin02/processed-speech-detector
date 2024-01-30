import torch
from models import AudioClassifier

dependencies = ["torch", "librosa"]

URLS = {
    "freq": "https://github.com/litagin02/processed-speech-detector/releases/download/1.0/freq_model.pth",
    "reverb": "https://github.com/litagin02/processed-speech-detector/releases/download/1.0/reverb_model.pth",
}


def processed_speech_detector(task="freq", progress=True):
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
        state_dict = torch.hub.load_state_dict_from_url(
            URLS["freq"], progress=progress, map_location="cpu"
        )
        model.load_state_dict(state_dict)
    elif task == "reverb":
        state_dict = torch.hub.load_state_dict_from_url(
            URLS["reverb"], progress=progress, map_location="cpu"
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown task: {task}")
    model.eval()
    return model
