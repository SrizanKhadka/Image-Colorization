from pathlib import Path

from django import forms
from django.conf import settings


class ImageUploadForm(forms.Form):
    image = forms.ImageField()

    def clean_image(self):
        image = self.cleaned_data["image"]
        extension = Path(image.name).suffix.lower()

        if extension not in settings.COLORIZER_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(settings.COLORIZER_ALLOWED_EXTENSIONS))
            raise forms.ValidationError(f"Unsupported file type. Use {allowed}.")

        if image.size > settings.COLORIZER_MAX_UPLOAD_BYTES:
            max_mb = settings.COLORIZER_MAX_UPLOAD_BYTES // (1024 * 1024)
            raise forms.ValidationError(f"Image is too large. Maximum size is {max_mb} MB.")

        return image
