import math
from typing import Optional, Tuple


def angle_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Angle in degrees of vector p1->p2 relative to +x axis (right)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def angle_at(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Return the inner angle ABC in degrees (0..180)."""
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba == 0 or mag_bc == 0:
        return float("nan")
    cosang = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosang))


def angle_to_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Angle of vector p1->p2 relative to vertical (up). Returns degrees in [0..180].
    0 means perfectly vertical, 90 means horizontal.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    mag = math.hypot(dx, dy)
    if mag == 0:
        return float("nan")
    # Vertical up is angle -90 deg in screen coordinates (y increases downward).
    # Compute absolute difference from vertical.
    theta = math.degrees(math.atan2(dy, dx))  # relative to +x axis
    diff = abs((-90.0) - theta)
    # Normalize to [0..180]
    diff = ((diff + 180.0) % 360.0) - 180.0
    return abs(diff)


def choose_side(
    left_vis: Optional[float], right_vis: Optional[float]
) -> str:
    """Choose 'left' or 'right' based on greater visibility (fallback to 'left')."""
    lv = left_vis if left_vis is not None else -1.0
    rv = right_vis if right_vis is not None else -1.0
    return "left" if lv >= rv else "right"


class StrokeDetector:
    """
    Detects stroke cycle events from knee angle series using simple zero-crossings.

    - A catch is detected when angle derivative goes from negative->positive and
      the knee angle is below a threshold.
    - Finish is the max knee angle between catches.
    """

    def __init__(self, catch_angle_max: float = 110.0, smooth: int = 5):
        self.catch_angle_max = catch_angle_max
        self.smooth = max(1, smooth)
        self._angles = []  # list of (t, angle)
        self._deriv = []  # list of (t, d_angle/dt)
        self.catches = []  # times of catch
        self.finishes = []  # times of finish

    def _smoothed_angle(self) -> Optional[float]:
        if len(self._angles) < self.smooth:
            return None
        return sum(a for _, a in self._angles[-self.smooth :]) / self.smooth

    def update(self, t: float, angle: float) -> None:
        if math.isnan(angle):
            return
        self._angles.append((t, angle))
        if len(self._angles) >= 2:
            (t0, a0), (t1, a1) = self._angles[-2], self._angles[-1]
            dt = max(1e-6, t1 - t0)
            self._deriv.append((t1, (a1 - a0) / dt))

        # Check for catch event (derivative crossing from negative to positive)
        if len(self._deriv) >= 2:
            (t_prev, d_prev), (t_now, d_now) = self._deriv[-2], self._deriv[-1]
            sm = self._smoothed_angle()
            if (
                sm is not None
                and d_prev < 0 <= d_now
                and sm <= self.catch_angle_max
            ):
                # New catch
                if not self.catches or (t_now - self.catches[-1]) > 0.5:
                    self.catches.append(t_now)

        # Track finish as local max between catches
        if len(self.catches) >= 1:
            # search in window since last catch for max angle
            last_catch_t = self.catches[-1]
            window = [p for p in self._angles if p[0] >= last_catch_t]
            if len(window) >= 3:
                # max in the window is provisional finish
                t_fin, a_fin = max(window, key=lambda x: x[1])
                # Avoid multiple finishes for same stroke: record only if newer
                if not self.finishes or t_fin > self.finishes[-1]:
                    self.finishes.append(t_fin)

    def summary(self) -> dict:
        # Stroke count via catches
        strokes = len(self.catches)
        spm = 0.0
        drive_recovery_ratio = None
        if strokes >= 2:
            # SPM from average period between catches
            periods = [
                self.catches[i + 1] - self.catches[i]
                for i in range(len(self.catches) - 1)
                if (self.catches[i + 1] - self.catches[i]) > 0.1
            ]
            if periods:
                avg_period = sum(periods) / len(periods)
                spm = 60.0 / avg_period

            # Drive/Recovery ratio: use last complete stroke
            # Catch_i -> Finish_i = drive, Finish_i -> Catch_{i+1} = recovery
            for i in reversed(range(min(len(self.finishes), len(self.catches) - 1))):
                t_catch = self.catches[i]
                t_fin = self.finishes[i]
                t_next = self.catches[i + 1]
                if t_catch < t_fin < t_next:
                    drive = t_fin - t_catch
                    recovery = t_next - t_fin
                    if drive > 0 and recovery > 0:
                        drive_recovery_ratio = drive / recovery
                        break

        return {
            "strokes": strokes,
            "spm": spm,
            "drive_recovery_ratio": drive_recovery_ratio,
        }

