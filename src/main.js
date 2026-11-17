import * as ort from 'onnxruntime-web';
import * as exifr from 'exifr';
import './style.css';

const MODEL_URL = 'https://huggingface.co/Loke-60000/slop-detect-vit-s-onnx/resolve/main/model.onnx';
const MODEL_SHA256 = 'd431efd677cc124ab68c3f1b20d628b7e9a8362a99803b1878475060b05cff5a';
const CACHE_NAME = 'tracelens-model-v1';
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/avif', 'image/heic', 'image/heif']);
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

const $ = (selector) => document.querySelector(selector);
const fileInput = $('#file-input');
const dropzone = $('#dropzone');
const previewWrap = $('#preview-wrap');
const progressArea = $('#progress-area');
const analyzeButton = $('#analyze-button');
const resultsSection = $('#results');
let selectedFile = null;
let previewUrl = null;
let sessionPromise = null;
let lastReport = null;
let toastTimer = null;

function formatBytes(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}

function setProgress(label, detail, percent = null) {
  progressArea.classList.remove('hidden');
  $('#progress-label').textContent = label;
  $('#progress-detail').textContent = detail;
  $('#progress-percent').textContent = percent === null ? '•••' : `${percent}%`;
  $('#progress-bar').style.width = percent === null ? '35%' : `${percent}%`;
  $('#progress-bar').classList.toggle('indeterminate', percent === null);
}

function resetProgress() {
  progressArea.classList.add('hidden');
  $('#progress-bar').style.width = '0%';
  $('#progress-bar').classList.remove('indeterminate');
}

function setFile(file) {
  if (!file) return;
  if (!ALLOWED_TYPES.has(file.type) && !/\.(jpe?g|png|webp|avif|heic|heif)$/i.test(file.name)) {
    showToast('Choose a JPG, PNG, WEBP, AVIF, or HEIC image.');
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showToast('This image is over the 20 MB limit.');
    return;
  }
  if (!file.size) {
    showToast('That file appears to be empty.');
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('#preview-image').src = previewUrl;
  $('#file-name').textContent = file.name;
  $('#file-details').textContent = `${formatBytes(file.size)} · ${file.type || 'image file'}`;
  dropzone.classList.add('hidden');
  previewWrap.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  resetProgress();
}

function clearFile() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('#preview-image').removeAttribute('src');
  fileInput.value = '';
  previewWrap.classList.add('hidden');
  dropzone.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  resetProgress();
}

function bytesToHex(buffer) {
  return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function sha256(buffer) {
  return bytesToHex(await crypto.subtle.digest('SHA-256', buffer));
}

async function fetchModel(progressCallback) {
  const cache = 'caches' in window ? await caches.open(CACHE_NAME) : null;
  const cached = cache ? await cache.match(MODEL_URL) : null;
  if (cached) {
    const cachedBuffer = await cached.arrayBuffer();
    const cachedHash = await sha256(cachedBuffer);
    if (cachedHash === MODEL_SHA256) {
      progressCallback('Model ready', 'A verified copy is saved in this browser.', 100);
      return cachedBuffer;
    }
    await cache.delete(MODEL_URL);
  }

  progressCallback('Downloading the detector', 'About 23 MB · first scan only. Your image is not sent.', 4);
  const response = await fetch(MODEL_URL, { mode: 'cors', credentials: 'omit', cache: 'no-store' });
  if (!response.ok) throw new Error(`The model download failed (HTTP ${response.status}). Check your connection and try again.`);
  const total = Number(response.headers.get('content-length')) || 0;
  let buffer;
  if (response.body && total) {
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      progressCallback('Downloading the detector', 'About 23 MB · first scan only. Your image is not sent.', Math.min(94, Math.round((loaded / total) * 94)));
    }
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    buffer = bytes.buffer;
  } else {
    buffer = await response.arrayBuffer();
    progressCallback('Download complete', 'Checking the model file before use.', 94);
  }
  const actualHash = await sha256(buffer);
  if (actualHash !== MODEL_SHA256) throw new Error('The downloaded model did not match its published SHA-256 checksum. It was not used. Try again later.');
  if (cache) {
    try { await cache.put(MODEL_URL, new Response(buffer.slice(0), { headers: { 'Content-Type': 'application/octet-stream' } })); } catch { /* Storage may be unavailable or full; inference can still proceed. */ }
  }
  progressCallback('Model verified', 'Checksum matches the published model card.', 100);
  return buffer;
}

function loadSession(progressCallback) {
  if (!sessionPromise) {
    sessionPromise = fetchModel(progressCallback).then((modelBuffer) => {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
      return ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }).catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

async function imageToTensor(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const shortest = Math.min(bitmap.width, bitmap.height);
  const scale = 440 / shortest;
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const scaled = document.createElement('canvas');
  scaled.width = width;
  scaled.height = height;
  const scaledContext = scaled.getContext('2d', { willReadFrequently: true });
  scaledContext.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const left = Math.floor((width - 384) / 2);
  const top = Math.floor((height - 384) / 2);
  const crop = scaledContext.getImageData(left, top, 384, 384).data;
  const planeSize = 384 * 384;
  const tensorData = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const source = pixel * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      const normalized = crop[source + channel] / 255;
      tensorData[channel * planeSize + pixel] = (normalized - CLIP_MEAN[channel]) / CLIP_STD[channel];
    }
  }
  return new ort.Tensor('float32', tensorData, [1, 3, 384, 384]);
}

async function getMetadata(file) {
  try {
    const parsed = await exifr.parse(file, { pick: ['Make', 'Model', 'DateTimeOriginal', 'CreateDate', 'Software', 'LensModel', 'Orientation'] });
    return parsed || {};
  } catch {
    return {};
  }
}

function readableDate(date) {
  if (!date) return null;
  const parsed = date instanceof Date ? date : new Date(date);
  return Number.isNaN(parsed.getTime()) ? String(date) : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(parsed);
}

function buildFindings(file, metadata, dimensions) {
  const findings = [];
  const camera = [metadata.Make, metadata.Model].filter(Boolean).join(' ').trim();
  if (camera) {
    findings.push({ type: 'neutral', title: 'Camera metadata found', detail: `${camera}${metadata.DateTimeOriginal ? ` · ${readableDate(metadata.DateTimeOriginal)}` : ''}`, icon: 'camera' });
  } else {
    findings.push({ type: 'muted', title: 'No camera metadata found', detail: 'Common for edited, shared, and AI-made images.', icon: 'info' });
  }
  if (metadata.Software) {
    findings.push({ type: 'neutral', title: 'Software metadata found', detail: String(metadata.Software), icon: 'edit' });
  }
  findings.push({ type: 'neutral', title: 'Image dimensions', detail: `${dimensions.width.toLocaleString()} × ${dimensions.height.toLocaleString()} px · ${formatBytes(file.size)}`, icon: 'image' });
  findings.push({ type: 'muted', title: 'Metadata is not proof', detail: 'It can be removed, rewritten, or preserved through editing.', icon: 'info' });
  return findings;
}

function iconFor(name) {
  const paths = {
    camera: '<path d="M4 7.5h3l1.4-2h7.2l1.4 2h3v11H4z"/><circle cx="12" cy="13" r="3.2"/>',
    edit: '<path d="m14.5 6.5 3 3M4 20l4.4-.8L19 8.6a2.1 2.1 0 0 0-3-3L5.4 16.2 4 20Z"/>',
    image: '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><circle cx="9" cy="10" r="1.5"/><path d="m5 17 4-4 3 3 3.5-4 3.5 4"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5m0-8h.01"/>',
  };
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`;
}

function renderResults(score, file, metadata, dimensions) {
  const rounded = Math.round(score * 100);
  const signal = rounded >= 65 ? 'ai' : rounded <= 35 ? 'photo' : 'uncertain';
  const info = {
    ai: { badge: 'MODEL LEANS AI', heading: 'AI-like patterns detected', copy: 'This classifier found patterns associated with AI-generated images in its training data. That can be a useful lead, but it is not proof.' },
    photo: { badge: 'MODEL LEANS PHOTO', heading: 'Fewer AI-like patterns', copy: 'This classifier found fewer patterns associated with AI-generated images in its training data. That does not verify a camera origin.' },
    uncertain: { badge: 'NO CLEAR SIGNAL', heading: 'The model is unsure', copy: 'This score is in the model’s uncertain range. Look for the original source and independent context.' },
  }[signal];

  $('#score-value').textContent = `${rounded}%`;
  $('#signal-badge').textContent = info.badge;
  $('#signal-badge').className = `signal-badge ${signal}`;
  $('#signal-heading').textContent = info.heading;
  $('#signal-copy').textContent = info.copy;
  $('#score-ring').style.setProperty('--score', `${rounded * 3.6}deg`);
  $('#score-ring').dataset.signal = signal;
  $('#scale-marker').style.left = `${rounded}%`;
  $('#context-status').textContent = 'LOCAL FILE CHECK';
  $('#findings-list').innerHTML = buildFindings(file, metadata, dimensions).map((item) => `
    <div class="finding-item ${item.type}"><span class="finding-icon">${iconFor(item.icon)}</span><span class="finding-text"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></span><span class="finding-mark">${item.type === 'muted' ? '·' : '✓'}</span></div>
  `).join('');
  $('#result-time').textContent = `SCANNED ${new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(new Date())}`;
  lastReport = {
    tool: 'TraceLens',
    generatedAt: new Date().toISOString(),
    file: file.name,
    bytes: file.size,
    dimensions,
    modelScore: rounded,
    modelSignal: info.badge,
    metadata: {
      camera: [metadata.Make, metadata.Model].filter(Boolean).join(' ') || null,
      software: metadata.Software || null,
      captured: readableDate(metadata.DateTimeOriginal) || null,
    },
    limitations: 'Model score is not a calibrated probability or proof of image origin. Metadata can be edited or removed.',
  };
  resultsSection.classList.remove('hidden');
  requestAnimationFrame(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.querySelector('.analyze-label').textContent = 'Preparing scan…';
  try {
    const [bitmap, metadata] = await Promise.all([
      createImageBitmap(selectedFile, { imageOrientation: 'from-image' }),
      getMetadata(selectedFile),
    ]);
    const dimensions = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    setProgress('Loading the image model', 'The first scan downloads and verifies the detector. Your image stays in this tab.', 0);
    const session = await loadSession(setProgress);
    setProgress('Reading image patterns', 'Running the model locally in your browser.', null);
    const input = await imageToTensor(selectedFile);
    const feeds = { [session.inputNames[0]]: input };
    const output = await session.run(feeds);
    const values = output[session.outputNames[0]].data;
    const logit = Number(values[0]);
    if (!Number.isFinite(logit)) throw new Error('The model returned an unreadable result. Try a different image.');
    const score = 1 / (1 + Math.exp(-logit));
    setProgress('Scan complete', 'The image was analyzed locally. No upload was made.', 100);
    renderResults(score, selectedFile, metadata, dimensions);
    analyzeButton.querySelector('.analyze-label').textContent = 'Analyze again';
  } catch (error) {
    console.error('TraceLens scan error:', error);
    showToast(error?.message || 'Could not analyze this image. Try a JPG or PNG.');
    resetProgress();
    analyzeButton.querySelector('.analyze-label').textContent = 'Try again';
  } finally {
    analyzeButton.disabled = false;
  }
}

$('#browse-button').addEventListener('click', (event) => { event.stopPropagation(); fileInput.click(); });
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', (event) => setFile(event.target.files?.[0]));
$('#remove-image').addEventListener('click', clearFile);
$('#analyze-button').addEventListener('click', analyze);
$('#new-scan').addEventListener('click', () => { clearFile(); window.scrollTo({ top: $('#scan-heading').offsetTop - 36, behavior: 'smooth' }); });
$('#download-report').addEventListener('click', () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${lastReport.file.replace(/\.[^.]+$/, '')}-tracelens-report.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); });
}
dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer?.files?.[0]));
