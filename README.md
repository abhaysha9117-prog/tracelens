# TraceLens

**Image origin, with context.** TraceLens is a privacy-first browser app that combines a local AI-image classifier with basic file metadata, then explains the limits of both signals.

> A detector score is not proof. TraceLens is an exploratory tool, not an authenticity verdict.

## What it does

- Runs an ONNX image classifier locally in the browser with ONNX Runtime Web.
- Displays an AI-like model score with cautious interpretation bands.
- Reads camera and software metadata from the selected file when available.
- Shows file dimensions and a local preview.
- Offers a downloadable JSON report.
- Verifies the downloaded model with its published SHA-256 before inference.
- Keeps image bytes in the browser; no image upload or analysis API is used.

The first scan downloads the roughly 23 MB detector from Hugging Face. After a successful download it is cached in the browser for later scans. ONNX Runtime Web's matching WASM runtime is loaded from jsDelivr. The image itself is never sent to either service.

## Run locally

Requires Node.js 20+ and npm.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. For a production bundle:

```bash
npm run build
npm run preview
```

## How to read the result

TraceLens displays the classifier's AI score, not a calibrated probability that the image was made by AI. A score at or above 65 is described as “model leans AI”; a score at or below 35 as “model leans photo”; the middle range is “no clear signal.” Even those labels describe the model's output only.

Metadata can offer context, but it can be removed or edited. Camera metadata does not prove that an image is unaltered. Missing camera metadata is common after editing and social sharing, and is not evidence that an image is AI-generated.

Do not use TraceLens as the sole basis for moderation, journalism, legal, employment, or other high-stakes decisions. Check the original source, publication context, and independent evidence.

## Model and attribution

TraceLens uses [`Loke-60000/slop-detect-vit-s-onnx`](https://huggingface.co/Loke-60000/slop-detect-vit-s-onnx), an int8 ONNX ViT-S detector released under the MIT License. The model card describes its preprocessing and reports 85.3% balanced accuracy on one held-out evaluation involving 21 unseen generators with web-like image degradation. That figure is the model author's reported benchmark result, not a guarantee for new images or a benchmark independently reproduced by TraceLens.

Model file SHA-256 pinned by TraceLens:

```text
d431efd677cc124ab68c3f1b20d628b7e9a8362a99803b1878475060b05cff5a
```

The app uses the model's published preprocessing: resize the shortest side to 440 px, center crop to 384 × 384, apply CLIP channel normalization, and run the model's output through a sigmoid. The decision guide follows the model card's 0.65 operating point and adds a conservative uncertain band below it; these labels are UI guidance, not independently calibrated thresholds.

## Why caution matters

NeurIPS 2025's AIGIBench evaluation found that detector performance can drop significantly on real-world data and across generators, image degradation, and preprocessing conditions. A score should be one clue among several, not a final answer. [Read the benchmark abstract](https://papers.nips.cc/paper_files/paper/2025/hash/fb693c67f61e5321746ffce8b6fdd2d0-Abstract-Datasets_and_Benchmarks_Track.html).

## Privacy

- Selected image files are read by the browser and are never uploaded.
- The detector model is downloaded from Hugging Face on first use and cached by your browser.
- The ONNX Runtime Web WASM runtime is loaded from jsDelivr when a scan starts.
- No image, score, or file metadata is sent to an analytics or inference service.
- The app does not retain scan history; downloading a report is an explicit local action.

## Current scope

This is a research/demo application. It does not verify C2PA Content Credentials, inspect the original camera sensor data, identify the specific generator, or determine whether a person edited an image. HEIC support depends on the browser's image decoder.

## License

The TraceLens application code is MIT-licensed. The separately downloaded detector has its own license and attribution; see its [model card](https://huggingface.co/Loke-60000/slop-detect-vit-s-onnx).
