# Image Colorization Django App

A complete Django frontend/backend for uploading grayscale images, running your trained PyTorch `.pth` colorization model, previewing the before/after result, and downloading the generated image.

## Install

Use the project virtual environment, then install the Django, image-processing, and PyTorch dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

## Add Your Model

Your model is already configured here:

```text
models/imagecolorizationmodel.pth
```

Or change `COLORIZER_MODEL_PATH` in `colorize_web/settings.py`.

The loader expects the supplied pix2pix-style checkpoint with generator weights under the `net_G.` prefix. The app reconstructs the matching U-Net generator in `colorizer/ml/networks.py`.

## Run

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Folder Structure

```text
.
├── colorize_web/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── colorizer/
│   ├── forms.py
│   ├── urls.py
│   ├── views.py
│   ├── ml/
│   │   ├── networks.py
│   │   ├── model_loader.py
│   │   └── pipeline.py
│   ├── static/colorizer/
│   │   ├── css/styles.css
│   │   └── js/app.js
│   └── templates/colorizer/index.html
├── media/
│   ├── uploads/
│   └── outputs/
├── models/
│   └── imagecolorizationmodel.pth
├── manage.py
├── requirements.txt
└── README.md
```

## Upload-to-Output Flow

1. The user selects or drags a JPG, JPEG, or PNG into the page.
2. The frontend validates file type and size, then shows a local preview.
3. Clicking **Colorize image** posts the image to `/colorize/`.
4. `ImageUploadForm` validates the upload again on the server.
5. The view saves the input under `media/uploads/`.
6. `model_loader.py` loads `models/imagecolorizationmodel.pth` once and caches the PyTorch generator.
7. `pipeline.py` opens the image, converts it to grayscale, resizes it to 256x256, applies histogram equalization, normalizes the L channel to `[0, 1]`, runs the generator, scales predicted AB channels back by `118`, smooths AB with a light Gaussian blur, converts Lab back to RGB, and saves a PNG under `media/outputs/`.
8. The frontend displays the colorized image and enables the download link.

## Validation and Failure Handling

- Unsupported extensions are rejected.
- Uploads above 8 MB are rejected.
- Invalid image files are caught before inference.
- Missing PyTorch or missing model files produce clear UI errors.
- Model architecture mismatches return an actionable loader error.

For production, set `DJANGO_SECRET_KEY`, set `DJANGO_DEBUG=0`, configure `ALLOWED_HOSTS`, and serve media/static files with your deployment stack.

## Model Notes

The current integration is tailored to the supplied checkpoint:

- Input tensor shape: `(1, 1, 256, 256)`
- Input image mode: grayscale
- Preprocessing: resize to 256x256, histogram equalization with OpenCV, normalize to `[0, 1]`
- Output tensor shape: `(1, 2, 256, 256)`
- Output channels: Lab AB
- L reconstruction: `input * 100`
- AB scaling: `prediction * 118`
- AB smoothing: Gaussian blur with kernel `(3, 3)` and sigma `0.5`
- Runtime mode: the model is intentionally kept in train mode during `torch.no_grad()` inference to match the provided Kaggle script, which does not call `model.eval()`.

If you train a different model later, keep `colorizer/ml/model_loader.py` and `colorizer/ml/pipeline.py` as the integration boundary.
