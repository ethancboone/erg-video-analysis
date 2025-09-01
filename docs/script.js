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
};

let landmarker = null;
let running = false;
let videoSource = ""; // "webcam", "file", or "tab"
let ytWindow = null;
let scaleMetersPerPixel = null; // null means pixels
let calibrateMode = false;
let calibClicks = []; // [{x,y}] canvas pixel coords

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
    // CDN hosting of WASM assets
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.7/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

function drawResults(landmarks) {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!landmarks || landmarks.length === 0) return;
  const utils = new DrawingUtils(ctx);
  utils.drawLandmarks(landmarks[0], { color: "#52d1b8", lineWidth: 2, radius: 2 });
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

  const hip = lm[idx.hip];
  const knee = lm[idx.knee];
  const ankle = lm[idx.ankle];
  const shoulder = lm[idx.shoulder];

  const kneeAngle = angleAt(hip, knee, ankle);
  const backTilt = angleToVertical(hip, shoulder);
  const shinTilt = angleToVertical(ankle, knee);

  stroke.update(ts / 1000, kneeAngle);
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
    this.samples = []; // {t, v, a}
  }
  push(t, v, a) {
    this.samples.push({ t, v, a });
    const tMin = t - this.windowSec;
    while (this.samples.length && this.samples[0].t < tMin) this.samples.shift();
    this.draw();
  }
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
    let maxVal = 1e-3;
    for (const s of this.samples) maxVal = Math.max(maxVal, s.v, s.a);
    const pad = 4;
    const sx = (x) => pad + ((x - t0) / dt) * (w - 2 * pad);
    const sy = (y) => h - pad - (y / maxVal) * (h - 2 * pad);
    // Velocity in teal
    ctx.strokeStyle = "#52d1b8";
    ctx.beginPath();
    for (let i = 0; i < this.samples.length; i++) {
      const s = this.samples[i];
      const x = sx(s.t);
      const y = sy(s.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Acceleration in purple
    ctx.strokeStyle = "#b87cff";
    ctx.beginPath();
    for (let i = 0; i < this.samples.length; i++) {
      const s = this.samples[i];
      const x = sx(s.t);
      const y = sy(s.a);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

const plots = {
  shoulder: new MotionPlot(els.gShoulder),
  hip: new MotionPlot(els.gHip),
  knee: new MotionPlot(els.gKnee),
  ankle: new MotionPlot(els.gAnkle),
  wrist: new MotionPlot(els.gWrist),
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
  // points: {shoulder, hip, knee, ankle, wrist} where each has normalized x,y
  // Convert to pixels based on canvas size for consistent scale
  const w = els.canvas.width;
  const h = els.canvas.height;
  const names = ["shoulder", "hip", "knee", "ankle", "wrist"];
  for (const name of names) {
    const p = points[name];
    if (!p) continue;
    const pos = { x: p.x * w, y: p.y * h };
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
