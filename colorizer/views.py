from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .forms import ImageUploadForm
from .ml.pipeline import ColorizationError, colorize_file


def index(request):
    return render(request, "colorizer/index.html")


@require_POST
def colorize_image(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"ok": False, "error": form.errors["image"][0]}, status=400)

    upload = form.cleaned_data["image"]
    upload_storage = FileSystemStorage(location=settings.MEDIA_ROOT / "uploads")
    output_storage = FileSystemStorage(location=settings.MEDIA_ROOT / "outputs")

    extension = Path(upload.name).suffix.lower()
    upload_name = upload_storage.save(f"{uuid4().hex}{extension}", upload)
    input_path = settings.MEDIA_ROOT / "uploads" / upload_name
    output_name = f"{Path(upload_name).stem}_colorized.png"
    output_path = settings.MEDIA_ROOT / "outputs" / output_name

    try:
        colorize_file(input_path=input_path, output_path=output_path)
    except ColorizationError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=422)
    except Exception:
        return JsonResponse(
            {"ok": False, "error": "Processing failed. Check the server logs for details."},
            status=500,
        )

    return JsonResponse(
        {
            "ok": True,
            "input_url": settings.MEDIA_URL + f"uploads/{upload_name}",
            "output_url": settings.MEDIA_URL + f"outputs/{output_name}",
            "download_url": settings.MEDIA_URL + f"outputs/{output_name}",
        }
    )
