# Processed Speech Detector

Detect whether the speech voice is manually processed.


## Usage

```python
import torch

model = torch.hub.load(
    "litagin02/processed-speech-detector",
    "processed_speech_detector",
    task="freq",  # or "reverb"
    device="cpu",  # or "cuda"
)
score = model.infer_from_file("test.wav")
print(f"Score: {score:.3f}")
```

Task:
- `freq`: Detect whether the speech has been passed through low-pass, high-pass, or band-pass filters.
- `reverb`: Detect whether the speech has reverb applied.

Score is a float number between 0 and 1. The higher the score, the more likely the speech has been processed.
