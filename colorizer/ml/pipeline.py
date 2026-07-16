from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from PIL import Image, UnidentifiedImageError
from skimage.color import lab2rgb

from .model_loader import ModelNotConfiguredError, get_model


class ColorizationError(RuntimeError):
    pass


def colorize_file(input_path: Path, output_path: Path) -> None:
    """Validate, preprocess, infer, post-process, and save a colorized image."""
    try:
        original = Image.open(input_path)
        original.verify()
        original = Image.open(input_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ColorizationError("Invalid image file. Please upload a valid JPG or PNG.") from exc

    try:
        model, device = get_model()
    except ModelNotConfiguredError as exc:
        raise ColorizationError(str(exc)) from exc

    model_input, original_l = _preprocess(original, device)

    try:
        import torch

        with torch.no_grad():
            prediction = model(model_input).detach().cpu().numpy()
    except Exception as exc:
        raise ColorizationError(
            "Model inference failed. This usually means the preprocessing shape does not "
            "match the PyTorch generator input. Expected input shape is (1, 1, 256, 256)."
        ) from exc

    colorized = _postprocess(original, original_l, prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colorized.save(output_path, format="PNG")


def _preprocess(image: Image.Image, device):
    import torch

    target_size = settings.COLORIZER_TARGET_SIZE
    resized = image.convert("L").resize(target_size, Image.Resampling.LANCZOS)
    gray = np.asarray(resized).astype("uint8")

    # Match the original inference notebook: equalize the grayscale image before
    # normalizing it to the model's single-channel input tensor.
    equalized = cv2.equalizeHist(gray)
    normalized_l = equalized.astype("float32") / 255.0
    tensor = torch.from_numpy(normalized_l).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor, normalized_l


def _postprocess(original: Image.Image, normalized_l: np.ndarray, prediction: np.ndarray) -> Image.Image:
    pred = np.asarray(prediction)
    if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 2:
        raise ColorizationError(f"Unsupported model output shape: {pred.shape}. Expected (1, 2, H, W).")

    ab = np.moveaxis(pred[0], 0, -1) * 118.0
    lab = np.zeros((ab.shape[0], ab.shape[1], 3), dtype="float32")
    lab[:, :, 0] = normalized_l * 100.0
    lab[:, :, 1:] = ab
    lab = _smooth_lab_image(lab)

    rgb = np.clip(lab2rgb(lab) * 255.0, 0, 255).astype("uint8")
    result = Image.fromarray(rgb, mode="RGB")
    return result.resize(original.size, Image.Resampling.LANCZOS)


def _smooth_lab_image(lab_image: np.ndarray) -> np.ndarray:
    lab_image[:, :, 1] = cv2.GaussianBlur(lab_image[:, :, 1], (3, 3), 0.5)
    lab_image[:, :, 2] = cv2.GaussianBlur(lab_image[:, :, 2], (3, 3), 0.5)
    return lab_image
