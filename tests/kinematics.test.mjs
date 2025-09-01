import assert from 'node:assert/strict';
import test from 'node:test';
import { angleAt, angleToVertical, StrokeState } from '../docs/lib/kinematics.mjs';

test('angleAt right angle ~90 deg', () => {
  const a = { x: 0, y: 1 };
  const b = { x: 0, y: 0 };
  const c = { x: 1, y: 0 };
  const ang = angleAt(a, b, c);
  assert.ok(Math.abs(ang - 90) < 1e-6);
});

test('angleToVertical vertical vector = 0 deg', () => {
  const p1 = { x: 0, y: 1 };
  const p2 = { x: 0, y: 0 };
  const ang = angleToVertical(p1, p2);
  assert.ok(Math.abs(ang - 0) < 1e-6);
});

test('StrokeState detects approximate SPM', () => {
  const st = new StrokeState({ catchAngleMax: 110, smooth: 3 });
  // Simulate a sinusoidal knee angle (in degrees) around 100..160
  // Period ~ 1.5s -> ~40 SPM
  let t = 0;
  const dt = 1/60; // 60 Hz
  const T = 1.5; // seconds per stroke
  for (let i = 0; i < 60 * 20; i++) { // 20 seconds
    const ang = 130 + 30 * Math.sin(2 * Math.PI * t / T);
    st.update(t, ang);
    t += dt;
  }
  const sum = st.summary();
  assert.ok(sum.strokes > 5, 'should detect multiple strokes');
  assert.ok(sum.spm > 30 && sum.spm < 50, `SPM in plausible range, got ${sum.spm}`);
});
