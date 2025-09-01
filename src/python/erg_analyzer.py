import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

try:
    # When run as a module: python -m src.python.erg_analyzer
    from .pose_metrics import angle_at, angle_to_vertical, choose_side, StrokeDetector
except Exception:  # When run as a script: python src/python/erg_analyzer.py
    from pose_metrics import angle_at, angle_to_vertical, choose_side, StrokeDetector


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class FrameMetrics:
    t: float
    knee_angle: float
    hip_to_vertical: float
    shin_to_vertical: float


def pick_side_landmarks(lm) -> Tuple[dict, str]:
    # Return dict for chosen side with keys: hip, knee, ankle, shoulder, elbow, wrist, visibility
    left_vis = lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility
    right_vis = lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
    side = choose_side(left_vis, right_vis)
    if side == "left":
        idx = mp_pose.PoseLandmark
        return (
            {
                "hip": lm[idx.LEFT_HIP],
                "knee": lm[idx.LEFT_KNEE],
                "ankle": lm[idx.LEFT_ANKLE],
                "shoulder": lm[idx.LEFT_SHOULDER],
                "elbow": lm[idx.LEFT_ELBOW],
                "wrist": lm[idx.LEFT_WRIST],
            },
            side,
        )
    else:
        idx = mp_pose.PoseLandmark
        return (
            {
                "hip": lm[idx.RIGHT_HIP],
                "knee": lm[idx.RIGHT_KNEE],
                "ankle": lm[idx.RIGHT_ANKLE],
                "shoulder": lm[idx.RIGHT_SHOULDER],
                "elbow": lm[idx.RIGHT_ELBOW],
                "wrist": lm[idx.RIGHT_WRIST],
            },
            side,
        )


def to_xy(p, w: int, h: int) -> Tuple[float, float]:
    return p.x * w, p.y * h


def compute_metrics(lm, w: int, h: int) -> Optional[FrameMetrics]:
    try:
        pts, side = pick_side_landmarks(lm)
    except Exception:
        return None

    hip = to_xy(pts["hip"], w, h)
    knee = to_xy(pts["knee"], w, h)
    ankle = to_xy(pts["ankle"], w, h)
    shoulder = to_xy(pts["shoulder"], w, h)

    # Knee angle (hip–knee–ankle)
    knee_angle = angle_at(hip, knee, ankle)
    # Back angle to vertical: hip->shoulder vs vertical
    hip_to_vertical = angle_to_vertical(hip, shoulder)
    # Shin angle to vertical: ankle->knee vs vertical
    shin_to_vertical = angle_to_vertical(ankle, knee)

    return FrameMetrics(
        t=0.0,  # to be filled by caller
        knee_angle=knee_angle,
        hip_to_vertical=hip_to_vertical,
        shin_to_vertical=shin_to_vertical,
    )


def draw_overlays(
    frame,
    results,
    fm: Optional[FrameMetrics],
    stroke_summary: Optional[dict],
    color=(0, 255, 0),
):
    h, w = frame.shape[:2]
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    lines = []
    if fm:
        lines.append(f"Knee angle: {fm.knee_angle:5.1f}°")
        lines.append(f"Back tilt:  {fm.hip_to_vertical:5.1f}° from vertical")
        lines.append(f"Shin tilt:  {fm.shin_to_vertical:5.1f}° from vertical")
    if stroke_summary:
        lines.append(f"Strokes:    {stroke_summary['strokes']}")
        lines.append(f"SPM:        {stroke_summary['spm']:.1f}")
        drr = stroke_summary.get("drive_recovery_ratio")
        if drr:
            lines.append(f"Drive/Rec:  {drr:.2f}")

    y = 24
    for text in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 24


def run(source: str, annotate: Optional[str] = None, display: bool = True):
    # Video source
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if annotate:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotate, fourcc, fps, (width, height))

    stroke = StrokeDetector(catch_angle_max=110.0, smooth=5)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        t0 = time.time()
        idx = 0
        last_summary = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = time.time() - t0
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            fm = None
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                fm = compute_metrics(lm, width, height)
                if fm:
                    fm.t = t
                    stroke.update(fm.t, fm.knee_angle)

            if idx % int(max(1, fps // 4)) == 0:  # update summary a few times/sec
                last_summary = stroke.summary()

            out_frame = frame.copy()
            draw_overlays(out_frame, results, fm, last_summary)

            if display:
                cv2.imshow("Erg Analysis", out_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            if writer:
                writer.write(out_frame)

            idx += 1

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Indoor rowing side-view analyzer (local)")
    ap.add_argument(
        "--source",
        required=True,
        help='"webcam" or path to a video file',
    )
    ap.add_argument(
        "--annotate",
        default=None,
        help="Optional output video path (e.g., out.mp4)",
    )
    ap.add_argument(
        "--no-display",
        action="store_true",
        help="Disable on-screen window (useful for headless runs)",
    )
    args = ap.parse_args()

    run(args.source, annotate=args.annotate, display=not args.no_display)


if __name__ == "__main__":
    main()
