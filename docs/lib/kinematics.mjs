// Pure math utilities for erg analysis (browser + Node tests)

export function angleAt(a, b, c) {
  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const dot = bax * bcx + bay * bcy;
  const magBA = Math.hypot(bax, bay);
  const magBC = Math.hypot(bcx, bcy);
  if (magBA === 0 || magBC === 0) return NaN;
  const cosang = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return (Math.acos(cosang) * 180) / Math.PI;
}

export function angleToVertical(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const mag = Math.hypot(dx, dy);
  if (mag === 0) return NaN;
  const theta = (Math.atan2(dy, dx) * 180) / Math.PI; // vs +x
  let diff = Math.abs(-90 - theta);
  diff = ((diff + 180) % 360) - 180;
  return Math.abs(diff);
}

export class StrokeState {
  constructor({
    catchAngleMax = 110,
    smooth = 5,
    minCatchIntervalSec = 0.9,
    confirmFrames = 2,
    windowSize = 9,
  } = {}) {
    this.catchAngleMax = catchAngleMax;
    this.smooth = Math.max(1, smooth);
    this.minCatchIntervalSec = minCatchIntervalSec;
    this.confirmFrames = Math.max(1, confirmFrames);
    this.windowSize = Math.max(3, windowSize);
    this.angles = []; // {t, a}
    this.deriv = []; // {t, d}
    this.catches = [];
    this.finishes = [];
    this._posStreak = 0;
  }

  smoothedAngle() {
    if (this.angles.length < this.smooth) return null;
    let s = 0;
    for (let i = this.angles.length - this.smooth; i < this.angles.length; i++) s += this.angles[i].a;
    return s / this.smooth;
  }

  update(t, angle) {
    if (!Number.isFinite(angle)) return;
    const L = this.angles.length;
    this.angles.push({ t, a: angle });
    if (L >= 1) {
      const { t: t0, a: a0 } = this.angles[L - 1];
      const dt = Math.max(1e-6, t - t0);
      this.deriv.push({ t, d: (angle - a0) / dt });
    }

    if (this.deriv.length >= 1) {
      const { d: dNow } = this.deriv[this.deriv.length - 1];
      this._posStreak = dNow >= 0 ? this._posStreak + 1 : 0;
    }

    if (this.deriv.length >= 2) {
      const { d: dPrev } = this.deriv[this.deriv.length - 2];
      const { d: dNow } = this.deriv[this.deriv.length - 1];
      const sm = this.smoothedAngle();
      const lastCatchOk =
        this.catches.length === 0 ||
        t - this.catches[this.catches.length - 1] > this.minCatchIntervalSec;

      if (
        sm != null &&
        dPrev < 0 && dNow >= 0 &&
        sm <= this.catchAngleMax &&
        lastCatchOk &&
        this._posStreak >= this.confirmFrames &&
        this._isLocalMin()
      ) {
        this.catches.push(t);
      }
    }

    // Fallback: if derivative is too noisy, accept strict local minima under threshold
    // using only windowed check and min interval.
    const sm2 = this.smoothedAngle();
    if (
      sm2 != null &&
      sm2 <= this.catchAngleMax &&
      this._isLocalMin() &&
      (this.catches.length === 0 || t - this.catches[this.catches.length - 1] > this.minCatchIntervalSec)
    ) {
      // Prevent double add within the same frame cycle
      const last = this.catches[this.catches.length - 1];
      if (!(last && Math.abs(last - t) < 1e-3)) {
        this.catches.push(t);
      }
    }

    if (this.catches.length >= 1) {
      const lastCatch = this.catches[this.catches.length - 1];
      const window = this.angles.filter((p) => p.t >= lastCatch);
      if (window.length >= 3) {
        const maxP = window.reduce((m, p) => (p.a > m.a ? p : m));
        if (this.finishes.length === 0 || maxP.t > this.finishes[this.finishes.length - 1]) {
          this.finishes.push(maxP.t);
        }
      }
    }
  }

  _isLocalMin() {
    if (this.angles.length < this.windowSize) return false;
    const end = this.angles.length - 1;
    const start = Math.max(0, end - this.windowSize + 1);
    let minA = Infinity;
    let minIdx = -1;
    for (let i = start; i <= end; i++) {
      const a = this.angles[i].a;
      if (a < minA) { minA = a; minIdx = i; }
    }
    return minIdx === end;
  }

  summary() {
    const strokes = this.catches.length;
    let spm = 0;
    let drr = null;
    if (strokes >= 2) {
      const periods = [];
      for (let i = 0; i < this.catches.length - 1; i++) {
        const dt = this.catches[i + 1] - this.catches[i];
        if (dt > 0.1) periods.push(dt);
      }
      if (periods.length) {
        const avg = periods.reduce((a, b) => a + b, 0) / periods.length;
        spm = 60 / avg;
      }
      for (let i = Math.min(this.finishes.length, this.catches.length - 1) - 1; i >= 0; i--) {
        const tC = this.catches[i];
        const tF = this.finishes[i];
        const tN = this.catches[i + 1];
        if (tC < tF && tF < tN) {
          const drive = tF - tC;
          const rec = tN - tF;
          if (drive > 0 && rec > 0) {
            drr = drive / rec;
            break;
          }
        }
      }
    }
    return { strokes, spm, drr };
  }
}
