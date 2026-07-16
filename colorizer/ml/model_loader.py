from functools import lru_cache

from django.conf import settings

from .networks import UnetGenerator


class ModelNotConfiguredError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_model():
    """Load the PyTorch generator once per Django process."""
    model_path = settings.COLORIZER_MODEL_PATH
    if not model_path.exists():
        raise ModelNotConfiguredError(
            f"Model file not found at {model_path}. Place your .pth file there or update "
            "COLORIZER_MODEL_PATH in settings.py."
        )

    try:
        import torch
    except Exception as exc:
        raise ModelNotConfiguredError(
            "PyTorch is not installed or could not be imported. Install the "
            "packages from requirements.txt."
        ) from exc

    device = _select_device(torch)
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as exc:
        raise ModelNotConfiguredError(f"Could not load PyTorch model file: {exc}") from exc

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ModelNotConfiguredError("Unsupported .pth file: expected a state_dict checkpoint.")

    generator_state = {
        key.removeprefix("net_G."): value
        for key, value in state_dict.items()
        if key.startswith("net_G.")
    }
    if not generator_state:
        generator_state = state_dict

    model = UnetGenerator(input_nc=1, output_nc=2, num_downs=8, ngf=64)
    try:
        model.load_state_dict(generator_state, strict=True)
    except Exception as exc:
        raise ModelNotConfiguredError(
            "The .pth file did not match the expected pix2pix U-Net generator architecture. "
            f"Load error: {exc}"
        ) from exc

    model.to(device)
    # Match the original Kaggle inference script. It loads MainModel and calls
    # forward under torch.no_grad(), but does not switch the model to eval mode.
    # Keeping train mode preserves the BatchNorm/Dropout behavior that produced
    # the user's reference colorization.
    model.train()
    return model, device


def _select_device(torch):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
