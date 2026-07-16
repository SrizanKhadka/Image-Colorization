const form = document.querySelector("#colorize-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const previewImage = document.querySelector("#preview-image");
const resultImage = document.querySelector("#result-image");
const previewFrame = previewImage.closest(".image-frame");
const resultFrame = resultImage.closest(".image-frame");
const button = document.querySelector("#colorize-button");
const statusBox = document.querySelector("#status");
const fileName = document.querySelector("#file-name");
const resultState = document.querySelector("#result-state");
const downloadLink = document.querySelector("#download-link");
const loader = document.querySelector("#loader");

const maxBytes = 8 * 1024 * 1024;
const allowedTypes = ["image/jpeg", "image/png"];

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function resetResult() {
  resultImage.removeAttribute("src");
  resultFrame.classList.remove("has-image");
  resultState.textContent = "Waiting";
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
}

function validateFile(file) {
  if (!file) return "Choose an image first.";
  if (!allowedTypes.includes(file.type)) return "Unsupported file type. Use JPG, JPEG, or PNG.";
  if (file.size > maxBytes) return "Image is too large. Maximum size is 8 MB.";
  return "";
}

function selectFile(file) {
  const error = validateFile(file);
  resetResult();

  if (error) {
    input.value = "";
    button.disabled = true;
    setStatus(error, true);
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewFrame.classList.add("has-image");
  fileName.textContent = file.name;
  button.disabled = false;
  setStatus("Ready to colorize.");
}

input.addEventListener("change", () => {
  selectFile(input.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;

  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  input.files = dataTransfer.files;
  selectFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = input.files[0];
  const error = validateFile(file);
  if (error) {
    setStatus(error, true);
    return;
  }

  const body = new FormData(form);
  button.disabled = true;
  loader.hidden = false;
  resultState.textContent = "Processing";
  setStatus("Running model inference...");

  try {
    const response = await fetch("/colorize/", {
      method: "POST",
      body,
      headers: {
        "X-Requested-With": "XMLHttpRequest",
      },
    });

    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Processing failed.");
    }

    resultImage.src = `${payload.output_url}?t=${Date.now()}`;
    resultFrame.classList.add("has-image");
    downloadLink.href = payload.download_url;
    downloadLink.classList.remove("disabled");
    resultState.textContent = "Complete";
    setStatus("Colorization complete.");
  } catch (err) {
    resultState.textContent = "Failed";
    setStatus(err.message, true);
  } finally {
    loader.hidden = true;
    button.disabled = false;
  }
});
