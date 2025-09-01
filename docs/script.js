// In-browser erg analysis using MediaPipe Pose Landmarker
// All processing happens locally; only the model file is fetched.

import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.7";
import { angleAt, angleToVertical, StrokeState } from "./lib/kinematics.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("overlay"),
  startBtn: document.getElementById("startWebcam"),
  stopBtn: document.getElementById("stopWebcam"),
  fileInput: document.getElementById("fileInput"),
  ytUrl: document.getElementById("ytUrl"),
  openYt: document.getElementById("openYt"),
  captureTab: document.getElementById("captureTab"),
  directUrl: document.getElementById("directUrl"),
  loadUrl: document.getElementById("loadUrl"),
  mStrokes: document.getElementById("mStrokes"),
  mSpm: document.getElementById("mSpm"),
  mDrr: document.getElementById("mDrr"),
  mBack: document.getElementById("mBack"),
  mShin: document.getElementById("mShin"),
  gShoulder: document.getElementById("gShoulder"),
  gHip: document.getElementById("gHip"),
  gKnee: document.getElementById("gKnee"),
  gAnkle: document.getElementById("gAnkle"),
  gWrist: document.getElementById("gWrist"),
  labelShoulder: document.getElementById("labelShoulder"),
  labelHip: document.getElementById("labelHip"),
  labelKnee: document.getElementById("labelKnee"),
  labelAnkle: document.getElementById("labelAnkle"),
  labelWrist: document.getElementById("labelWrist"),
  calibrateBtn: document.getElementById("calibrateBtn"),
  resetCalib: document.getElementById("resetCalib"),
  calibDist: document.getElementById("calibDist"),
  calibUnits: document.getElementById("calibUnits"),
  mScale: document.getElementById("mScale"),
  alert: document.getElementById("alert"),
  remoteVideo: document.getElementById("remoteVideo"),
  smoothAlpha: document.getElementById("smoothAlpha"),
  smoothAlphaVal: document.getElementById("smoothAlphaVal"),
  modelVariant: document.getElementById("modelVariant"),
  detectConf: document.getElementById("detectConf"),
  detectConfVal: document.getElementById("detectConfVal"),
  trackConf: document.getElementById("trackConf"),
  trackConfVal: document.getElementById("trackConfVal"),
  lmSmooth: document.getElementById("lmSmooth"),
  lmSmoothVal: document.getElementById("lmSmoothVal"),
  autoCatch: document.getElementById("autoCatch"),
  catchThresh: document.getElementById("catchThresh"),
};

let landmarker = null;
let running = false;
let videoSource = ""; // "webcam", "file", or "tab"
let ytWindow = null;
let scaleMetersPerPixel = null; // null means pixels
let calibrateMode = false;
let calibClicks = []; // [{x,y}] canvas pixel coords
let smoothingAlpha = parseFloat(els.smoothAlpha?.value || '0.30');
let landmarkAlpha = parseFloat(els.lmSmooth?.value || '0.20');
let detectConf = parseFloat(els.detectConf?.value || '0.50');
let trackConf = parseFloat(els.trackConf?.value || '0.50');
let modelVariant = els.modelVariant?.value || 'full';
let autoCatch = !!els.autoCatch?.checked;
let manualCatchThresh = parseFloat(els.catchThresh?.value || '110');

// Dynamic catch threshold window
const kneeWindow = [];
const kneeWindowSec = 8; // seconds of knee angle history

// Stroke detection state
let stroke = new StrokeState({ catchAngleMax: 110, smooth: 5 });

function resetState() {
  stroke = new StrokeState({ catchAngleMax: 110, smooth: 5 });
}

// angleAt and angleToVertical now imported from lib

function chooseSide(wrists) {
  // wrists: array of [leftVisibility, rightVisibility]
  const [lv, rv] = wrists;
  return lv >= rv ? "left" : "right";
}

// Stroke summary via class

async function setupLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.7/wasm"
  );
  const modelUrl = modelVariant === 'full'
    ? "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    : "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
  landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: { modelAssetPath: modelUrl },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: detectConf,
    minPosePresenceConfidence: detectConf,
    minTrackingConfidence: trackConf,
  });
}

function getContentRect() {
  const cw = els.canvas.width;
  const ch = els.canvas.height;
  const vw = els.video.videoWidth || cw;
  const vh = els.video.videoHeight || ch;
  const s = Math.min(cw / vw, ch / vh);
  const dw = vw * s;
  const dh = vh * s;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  return { ox, oy, s, vw, vh, dw, dh };
}

function adjustNormalizedLandmarksForCanvas(lms) {
  const { ox, oy, s, vw, vh } = getContentRect();
  // Map normalized video coords (0..1) to canvas normalized coords that include letterboxing offsets
  return lms.map((p) => ({
    x: (ox + (p.x * vw) * s) / els.canvas.width,
    y: (oy + (p.y * vh) * s) / els.canvas.height,
    z: p.z,
    visibility: p.visibility,
  }));
}

function drawResults(landmarks) {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!landmarks || landmarks.length === 0) return;
  const utils = new DrawingUtils(ctx);
  const adjusted = adjustNormalizedLandmarksForCanvas(landmarks[0]);
  utils.drawLandmarks(adjusted, { color: "#52d1b8", lineWidth: 2, radius: 2 });
}

function onResults(ts, results) {
  if (!results || !results.landmarks || results.landmarks.length === 0) return;
  const lm = results.landmarks[0];
  drawResults(results.landmarks);

  // Choose side by wrist visibility
  const leftWrist = lm[15];
  const rightWrist = lm[16];
  const side = chooseSide([leftWrist.visibility ?? 0, rightWrist.visibility ?? 0]);

  const idx = {
    left: { hip: 23, knee: 25, ankle: 27, shoulder: 11 },
    right: { hip: 24, knee: 26, ankle: 28, shoulder: 12 },
  }[side];

  // Landmark smoothing (EMA on normalized coordinates)
  const hip = smoothLm('hip', lm[idx.hip]);
  const knee = smoothLm('knee', lm[idx.knee]);
  const ankle = smoothLm('ankle', lm[idx.ankle]);
  const shoulder = smoothLm('shoulder', lm[idx.shoulder]);

  const kneeAngle = angleAt(hip, knee, ankle);
  const backTilt = angleToVertical(hip, shoulder);
  const shinTilt = angleToVertical(ankle, knee);

  const tSec = ts / 1000;
  updateKneeWindow(tSec, kneeAngle);
  if (autoCatch) stroke.catchAngleMax = estimateCatchThreshold();
  else stroke.catchAngleMax = manualCatchThresh;
  stroke.update(tSec, kneeAngle);
  const summary = stroke.summary();

  // Update UI
  els.mStrokes.textContent = String(summary.strokes);
  els.mSpm.textContent = summary.spm.toFixed(1);
  els.mDrr.textContent = summary.drr != null ? summary.drr.toFixed(2) : "—";
  els.mBack.textContent = Number.isFinite(backTilt) ? backTilt.toFixed(1) : "—";
  els.mShin.textContent = Number.isFinite(shinTilt) ? shinTilt.toFixed(1) : "—";

  // Update motion graphs for selected side keypoints
  updateMotionGraphs(ts / 1000, {
    shoulder,
    hip,
    knee,
    ankle,
    wrist: side === "left" ? lm[15] : lm[16],
  });
}

async function startWebcam() {
  if (!landmarker) await setupLandmarker();
  resetState();
  videoSource = "webcam";
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  els.video.srcObject = stream;
  await els.video.play();
  resizeCanvas();
  running = true;
  requestAnimationFrame(loop);
}

function stopWebcam() {
  running = false;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  const stream = els.video.srcObject;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  els.video.srcObject = null;
}

function resizeCanvas() {
  const rect = els.video.getBoundingClientRect();
  els.canvas.width = rect.width * devicePixelRatio;
  els.canvas.height = rect.height * devicePixelRatio;
  els.canvas.style.width = rect.width + "px";
  els.canvas.style.height = rect.height + "px";
  // Redraw calibration clicks overlay markers if any
  if (calibrateMode && calibClicks.length) drawCalibrationOverlay();
}

async function handleFile(file) {
  if (!landmarker) await setupLandmarker();
  resetState();
  videoSource = "file";
  const url = URL.createObjectURL(file);
  els.video.srcObject = null;
  els.video.src = url;
  await els.video.play();
  resizeCanvas();
  running = true;
  requestAnimationFrame(loop);
}

async function handleDirectUrl(url) {
  if (!landmarker) await setupLandmarker();
  resetState();
  videoSource = "url";
  clearAlert();
  try {
    els.remoteVideo.crossOrigin = "anonymous";
    els.remoteVideo.srcObject = null;
    els.remoteVideo.src = url;
    await els.remoteVideo.play();
    // Try to test CORS by drawing a frame to an offscreen canvas and reading back
    const testOk = await testFrameReadable(els.remoteVideo);
    if (!testOk) {
      showAlert("Could not analyze: cross-origin video without CORS. Please use Upload Video or Tab Capture.");
      els.remoteVideo.pause();
      els.remoteVideo.removeAttribute('src');
      els.remoteVideo.load();
      return;
    }
    // If readable, pipe the capture stream to our visible <video>
    const stream = els.remoteVideo.captureStream?.() || els.remoteVideo.mozCaptureStream?.();
    if (!stream) {
      showAlert("Browser does not support captureStream for media elements.");
      return;
    }
    els.video.srcObject = stream;
    await els.video.play();
    resizeCanvas();
    running = true;
    requestAnimationFrame(loop);
  } catch (e) {
    console.warn("Direct URL load failed", e);
    showAlert("Failed to load or play the URL. Ensure it is a direct video link (mp4/webm) with CORS enabled.");
  }
}

async function testFrameReadable(videoEl) {
  return new Promise((resolve) => {
    try {
      const c = document.createElement('canvas');
      c.width = Math.max(2, Math.floor(videoEl.videoWidth / 10));
      c.height = Math.max(2, Math.floor(videoEl.videoHeight / 10));
      const ctx = c.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, c.width, c.height);
      ctx.getImageData(0, 0, 1, 1);
      resolve(true);
    } catch (e) {
      resolve(false);
    }
  });
}

async function loop() {
  if (!running || !landmarker) return;
  if (els.video.readyState < 2) {
    requestAnimationFrame(loop);
    return;
  }
  const ts = performance.now();
  const results = await landmarker.detectForVideo(els.video, ts);
  onResults(ts, results);
  requestAnimationFrame(loop);
}

window.addEventListener("resize", resizeCanvas);
els.startBtn.addEventListener("click", startWebcam);
els.stopBtn.addEventListener("click", stopWebcam);
els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) handleFile(f);
});
els.loadUrl.addEventListener("click", () => {
  const url = els.directUrl.value.trim();
  if (url) handleDirectUrl(url);
});

// --- YouTube helpers (tab capture) ---
function openYouTube() {
  const url = els.ytUrl.value.trim();
  if (!url) return;
  try {
    ytWindow = window.open(url, "_blank");
  } catch (e) {
    console.warn("Failed to open URL", e);
  }
}

async function startTabCapture() {
  if (!landmarker) await setupLandmarker();
  resetState();
  videoSource = "tab";
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        displaySurface: "browser",
        frameRate: 30,
      },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    resizeCanvas();
    running = true;
    requestAnimationFrame(loop);
  } catch (e) {
    console.warn("Tab capture canceled or failed", e);
  }
}

els.openYt.addEventListener("click", openYouTube);
els.captureTab.addEventListener("click", startTabCapture);

// --- Real-time motion graphs ---
class MotionPlot {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.windowSec = opts.windowSec ?? 10; // show last N seconds
    this.samples = []; // raw samples: {t, v, a}
    this.alpha = opts.alpha ?? smoothingAlpha; // EMA smoothing factor
  }
  push(t, v, a) {
    this.samples.push({ t, v, a });
    const tMin = t - this.windowSec;
    while (this.samples.length && this.samples[0].t < tMin) this.samples.shift();
    this.draw();
  }
  setAlpha(alpha) { this.alpha = alpha; this.draw(); }
  draw() {
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b0f12";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#22313a";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      const y = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    if (this.samples.length < 2) return;
    const t0 = this.samples[0].t;
    const t1 = this.samples[this.samples.length - 1].t;
    const dt = Math.max(1e-3, t1 - t0);
    // Compute EMA-smoothed series for v and a
    const alpha = Math.max(0, Math.min(0.95, this.alpha ?? 0));
    const vSmooth = [];
    const aSmooth = [];
    let vE = this.samples[0].v;
    let aE = this.samples[0].a;
    for (let i = 0; i < this.samples.length; i++) {
      const s = this.samples[i];
      vE = alpha * s.v + (1 - alpha) * vE;
      aE = alpha * s.a + (1 - alpha) * aE;
      vSmooth.push({ t: s.t, y: vE });
      aSmooth.push({ t: s.t, y: aE });
    }
    let maxVal = 1e-3;
    for (const s of vSmooth) maxVal = Math.max(maxVal, Math.abs(s.y));
    for (const s of aSmooth) maxVal = Math.max(maxVal, Math.abs(s.y));
    const pad = 4;
    const sx = (x) => pad + ((x - t0) / dt) * (w - 2 * pad);
    const sy = (y) => h - pad - (y / maxVal) * (h - 2 * pad);
    // Velocity in teal
    ctx.strokeStyle = "#52d1b8";
    ctx.beginPath();
    for (let i = 0; i < vSmooth.length; i++) {
      const s = vSmooth[i];
      const x = sx(s.t);
      const y = sy(s.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Acceleration in purple
    ctx.strokeStyle = "#b87cff";
    ctx.beginPath();
    for (let i = 0; i < aSmooth.length; i++) {
      const s = aSmooth[i];
      const x = sx(s.t);
      const y = sy(s.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

const plots = {
  shoulder: new MotionPlot(els.gShoulder, { alpha: smoothingAlpha }),
  hip: new MotionPlot(els.gHip, { alpha: smoothingAlpha }),
  knee: new MotionPlot(els.gKnee, { alpha: smoothingAlpha }),
  ankle: new MotionPlot(els.gAnkle, { alpha: smoothingAlpha }),
  wrist: new MotionPlot(els.gWrist, { alpha: smoothingAlpha }),
};

const motionState = {
  prev: {
    shoulder: null, hip: null, knee: null, ankle: null, wrist: null,
  },
  prevVel: {
    shoulder: null, hip: null, knee: null, ankle: null, wrist: null,
  },
};

function updateMotionGraphs(t, points) {
  // points: {shoulder, hip, knee, ankle, wrist} normalized in video space (0..1)
  // Convert to canvas pixels using letterboxing-aware transform
  const { ox, oy, s, vw, vh } = getContentRect();
  const names = ["shoulder", "hip", "knee", "ankle", "wrist"];
  for (const name of names) {
    const p = points[name];
    if (!p) continue;
    const pos = { x: ox + (p.x * vw) * s, y: oy + (p.y * vh) * s };
    const prev = motionState.prev[name];
    let vMag = 0, aMag = 0;
    if (prev) {
      const dt = Math.max(1e-3, t - prev.t);
      const vx = (pos.x - prev.x) / dt;
      const vy = (pos.y - prev.y) / dt;
      vMag = Math.hypot(vx, vy);
      const prevV = motionState.prevVel[name];
      if (prevV) {
        const ax = (vx - prevV.vx) / dt;
        const ay = (vy - prevV.vy) / dt;
        aMag = Math.hypot(ax, ay);
      }
      motionState.prevVel[name] = { vx, vy };
    }
    motionState.prev[name] = { ...pos, t };
    if (scaleMetersPerPixel) {
      plots[name].push(t, vMag * scaleMetersPerPixel, aMag * scaleMetersPerPixel);
    } else {
      plots[name].push(t, vMag, aMag);
    }
  }
}

// --- Calibration ---
function startCalibration() {
  calibrateMode = true;
  calibClicks = [];
  updateScaleLabel();
}

function resetCalibration() {
  scaleMetersPerPixel = null;
  calibrateMode = false;
  calibClicks = [];
  drawCalibrationOverlay(true);
  updateScaleLabel();
}

function drawCalibrationOverlay(clearOnly = false) {
  const ctx = els.canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  if (clearOnly) {
    ctx.restore();
    return;
  }
  // Draw markers on top of existing overlay; use semi-transparent red
  ctx.fillStyle = "rgba(255,80,80,0.9)";
  ctx.strokeStyle = "rgba(255,80,80,0.9)";
  ctx.lineWidth = 2 * devicePixelRatio;
  if (calibClicks[0]) {
    const p = calibClicks[0];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4 * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
  }
  if (calibClicks.length === 2) {
    const a = calibClicks[0];
    const b = calibClicks[1];
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function canvasClickForCalibration(ev) {
  if (!calibrateMode) return;
  const rect = els.canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * devicePixelRatio;
  const y = (ev.clientY - rect.top) * devicePixelRatio;
  calibClicks.push({ x, y });
  drawCalibrationOverlay();
  if (calibClicks.length === 2) {
    calibrateMode = false;
    const dx = calibClicks[1].x - calibClicks[0].x;
    const dy = calibClicks[1].y - calibClicks[0].y;
    const pxDist = Math.hypot(dx, dy);
    const d = parseFloat(els.calibDist.value || "0");
    const units = els.calibUnits.value;
    const meters = units === "cm" ? d / 100 : d;
    if (pxDist > 0 && meters > 0) {
      scaleMetersPerPixel = meters / pxDist;
    }
    updateScaleLabel();
  }
}

function updateScaleLabel() {
  const units = scaleMetersPerPixel ? `m/px = ${scaleMetersPerPixel.toExponential(2)}` : "not set";
  els.mScale.textContent = units;
  const u = scaleMetersPerPixel ? "(m/s, m/s²)" : "(px/s, px/s²)";
  els.labelShoulder.textContent = `Shoulder v/a ${u}`;
  els.labelHip.textContent = `Hip v/a ${u}`;
  els.labelKnee.textContent = `Knee v/a ${u}`;
  els.labelAnkle.textContent = `Ankle v/a ${u}`;
  els.labelWrist.textContent = `Wrist v/a ${u}`;
}

els.calibrateBtn.addEventListener("click", startCalibration);
els.resetCalib.addEventListener("click", resetCalibration);
els.canvas.addEventListener("click", canvasClickForCalibration);
updateScaleLabel();

function showAlert(msg) {
  els.alert.textContent = msg;
  els.alert.style.display = 'block';
}
function clearAlert() {
  els.alert.textContent = '';
  els.alert.style.display = 'none';
}
// Smoothing control
if (els.smoothAlpha) {
  const updateSmoothLabel = () => { els.smoothAlphaVal.textContent = Number(els.smoothAlpha.value).toFixed(2); };
  updateSmoothLabel();
  els.smoothAlpha.addEventListener('input', () => {
    smoothingAlpha = parseFloat(els.smoothAlpha.value || '0');
    Object.values(plots).forEach(p => p.setAlpha(smoothingAlpha));
    updateSmoothLabel();
  });
}

// CV settings controls
function bindRange(el, label, setter) {
  if (!el) return;
  const upd = () => { label.textContent = Number(el.value).toFixed(2); setter(parseFloat(el.value)); };
  upd();
  el.addEventListener('input', upd);
}

bindRange(els.detectConf, els.detectConfVal, (v) => { detectConf = v; recreateLandmarker(); });
bindRange(els.trackConf, els.trackConfVal, (v) => { trackConf = v; recreateLandmarker(); });
bindRange(els.lmSmooth, els.lmSmoothVal, (v) => { landmarkAlpha = v; });

if (els.modelVariant) {
  els.modelVariant.addEventListener('change', async () => {
    modelVariant = els.modelVariant.value;
    await recreateLandmarker();
  });
}

if (els.autoCatch) {
  const toggle = () => {
    autoCatch = !!els.autoCatch.checked;
    els.catchThresh.disabled = autoCatch;
  };
  toggle();
  els.autoCatch.addEventListener('change', toggle);
}
if (els.catchThresh) {
  const upd = () => { manualCatchThresh = parseFloat(els.catchThresh.value || '110'); };
  upd();
  els.catchThresh.addEventListener('input', upd);
}

async function recreateLandmarker() {
  // Recreate landmarker with current options; keep running loop
  landmarker = null;
  await setupLandmarker();
}

// Landmark EMA smoother
const lmPrev = {
  shoulder: null, hip: null, knee: null, ankle: null, wrist: null,
};
function smoothLm(name, p) {
  if (!p) return p;
  const prev = lmPrev[name];
  if (!prev) {
    lmPrev[name] = { x: p.x, y: p.y };
    return p;
  }
  const a = Math.max(0, Math.min(0.9, landmarkAlpha || 0));
  const x = a * p.x + (1 - a) * prev.x;
  const y = a * p.y + (1 - a) * prev.y;
  lmPrev[name] = { x, y };
  return { x, y };
}

function updateKneeWindow(t, angle) {
  // Maintain windowed knee angles for dynamic thresholding
  if (!Number.isFinite(angle)) return;
  kneeWindow.push({ t, angle });
  const tMin = t - kneeWindowSec;
  while (kneeWindow.length && kneeWindow[0].t < tMin) kneeWindow.shift();
}

function estimateCatchThreshold() {
  if (kneeWindow.length < 10) return manualCatchThresh;
  let minA = Infinity, maxA = -Infinity;
  for (const k of kneeWindow) { if (k.angle < minA) minA = k.angle; if (k.angle > maxA) maxA = k.angle; }
  const range = Math.max(1, maxA - minA);
  const thr = minA + 0.35 * range; // heuristic
  return Math.max(70, Math.min(130, thr));
}
