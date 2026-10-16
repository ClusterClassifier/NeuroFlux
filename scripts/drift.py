"""Notebook definitions for drift."""

import math
import numpy as np


class ADWIN:
    """Pure Python ADWIN implementation (No external dependencies)."""
    _MAX_BUCKETS = 5

    class _Bucket:
        __slots__ = ("total", "variance", "size")
        def __init__(self):
            self.total = self.variance = 0.0
            self.size = 0

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._total = self._variance = 0.0
        self._width = 0
        self._bucket_rows = [[]]
        self.drift_detected = False

    @property
    def width(self):
        return self._width

    def update(self, value: float):
        self.drift_detected = False
        self._insert(float(value))
        self._check_drift()

    def _insert(self, value):
        b = self._Bucket()
        b.total = value
        b.size = 1
        self._bucket_rows[0].insert(0, b)
        self._width += 1
        old_mean = self._total / self._width if self._width > 1 else value
        self._variance += (value - old_mean) * (value - (self._total + value) / self._width)
        self._total += value
        self._compress()

    def _compress(self):
        level = 0
        while level < len(self._bucket_rows):
            row = self._bucket_rows[level]
            if len(row) <= self._MAX_BUCKETS:
                break
            if level + 1 >= len(self._bucket_rows):
                self._bucket_rows.append([])
            b1, b2 = row[-2], row[-1]
            m = self._Bucket()
            m.size = b1.size + b2.size
            m.total = b1.total + b2.total
            d = (b2.total / b2.size - b1.total / b1.size) if b1.size > 0 and b2.size > 0 else 0.0
            m.variance = b1.variance + b2.variance + d**2 * b1.size * b2.size / max(m.size, 1)
            self._bucket_rows[level] = row[:-2]
            self._bucket_rows[level + 1].insert(0, m)
            level += 1

    def _check_drift(self):
        n0 = total0 = 0.0
        for row in reversed(self._bucket_rows):
            for bucket in reversed(row):
                n0 += bucket.size
                total0 += bucket.total
                n1 = self._width - n0
                if n1 <= 0 or n0 <= 0:
                    continue
                m0 = total0 / n0
                m1 = (self._total - total0) / n1
                dd = math.log(2.0 * math.log(self._width + 1) / self.delta) if self._width > 1 else 0.0
                eps = math.sqrt(dd / (2 * n0)) + math.sqrt(dd / (2 * n1))
                if abs(m0 - m1) >= eps:
                    self._shrink(int(n0))
                    self.drift_detected = True
                    return

    def _shrink(self, drop):
        removed = 0.0
        rem = drop
        for level in range(len(self._bucket_rows) - 1, -1, -1):
            row = self._bucket_rows[level]
            while row and rem > 0:
                b = row[-1]
                if b.size <= rem:
                    rem -= b.size
                    removed += b.total
                    row.pop()
                else:
                    f = rem / b.size
                    removed += b.total * f
                    b.total -= b.total * f
                    b.size -= rem
                    rem = 0
        self._width -= drop
        self._total -= removed
        if self._variance < 0:
            self._variance = 0.0


class ChangePointDetector:
    """
    ADWIN wrapper with publication-grade post-hoc event matching.

    The detector itself is unchanged. The important correction is that false
    positives and detection delay are computed by matching detections to the
    known task-transition steps after training, instead of treating ADWIN's
    internal window width as the detection delay.
    """

    def __init__(self, delta: float = 0.05):
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.step_counter = 0
        self.detection_log = []

    def update(self, td_error: float, global_step: int = None):
        self.step_counter += 1
        self.adwin.update(float(td_error))

        if self.adwin.drift_detected:
            detected_at = int(global_step if global_step is not None else self.step_counter)
            estimated_boundary = max(0, detected_at - self.adwin.width)
            self.detection_log.append({
                'detected_at': detected_at,
                'estimated_boundary': estimated_boundary,
                'adwin_width': int(self.adwin.width),
            })
            return True, estimated_boundary

        return False, None

    def reset(self, reset_step_counter: bool = False) -> None:
        self.adwin = ADWIN(delta=self.delta)
        if reset_step_counter:
            self.step_counter = 0

    def get_detection_stats(self, true_boundaries=None, tolerance_steps: int = 2000) -> dict:
        """
        Matches detections one-to-one to known task boundaries.

        A detection is a true positive if detected_at lies in
        [boundary, boundary + tolerance_steps]. Unmatched detections are false
        positives and unmatched boundaries are false negatives.
        """
        detections = sorted(int(e['detected_at']) for e in self.detection_log)
        boundaries = sorted(int(b) for b in (true_boundaries or []))

        if not boundaries:
            return {
                'num_detections': len(detections),
                'true_positive_count': np.nan,
                'false_positive_count': np.nan,
                'false_negative_count': np.nan,
                'detection_rate': np.nan,
                'false_positive_rate': np.nan,
                'false_negative_rate': np.nan,
                'mean_detection_delay': np.nan,
                'std_detection_delay': np.nan,
            }

        used = set()
        delays = []
        tp = 0
        for boundary in boundaries:
            candidates = [
                (idx, d) for idx, d in enumerate(detections)
                if idx not in used and boundary <= d <= boundary + tolerance_steps
            ]
            if candidates:
                idx, det = min(candidates, key=lambda z: z[1])
                used.add(idx)
                tp += 1
                delays.append(det - boundary)

        fp = len(detections) - tp
        fn = len(boundaries) - tp

        return {
            'num_detections': len(detections),
            'num_true_boundaries': len(boundaries),
            'true_positive_count': tp,
            'false_positive_count': fp,
            'false_negative_count': fn,
            'detection_rate': tp / len(boundaries) if boundaries else np.nan,
            'false_positive_rate': fp / len(detections) if detections else 0.0,
            'false_negative_rate': fn / len(boundaries) if boundaries else np.nan,
            'mean_detection_delay': float(np.mean(delays)) if delays else np.nan,
            'std_detection_delay': float(np.std(delays, ddof=1)) if len(delays) > 1 else (0.0 if len(delays) == 1 else np.nan),
            'median_detection_delay': float(np.median(delays)) if delays else np.nan,
            'tolerance_steps': int(tolerance_steps),
        }


print("Change-point detector defined successfully.")
