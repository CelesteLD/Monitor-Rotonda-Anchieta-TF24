"""
roi_counter.py — Python wrapper for the C++ OpenMP ROI counter binary.

Replaces the joblib-based parallel count in detector.py.
The binary must be compiled and available at CPP_BINARY_PATH.
"""

import json
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

# Path to the compiled binary (set via env var or default)
CPP_BINARY_PATH = os.environ.get("ROI_COUNTER_BIN", "/app/cpp/roi_counter")

# Number of OpenMP threads (matches OMP_NUM_THREADS env var if set,
# otherwise defaults to the value below)
OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))


def count_boxes_per_roi(boxes: list[dict], rois: dict) -> dict:
    """
    Count how many bounding boxes fall (by center point) inside each ROI.

    Parameters
    ----------
    boxes : list of dicts with keys x1, y1, x2, y2  (pixel coords)
    rois  : dict  { roi_name: (x1, y1, x2, y2), ... }

    Returns
    -------
    dict  { roi_name: count, ..., "_elapsed_ms": float, "_threads": int }
    """
    # Build the JSON payload expected by the C++ binary
    payload = {
        "threads": OMP_THREADS,
        "rois": [
            {
                "name": name,
                "x1": float(coords[0]),
                "y1": float(coords[1]),
                "x2": float(coords[2]),
                "y2": float(coords[3]),
            }
            for name, coords in rois.items()
        ],
        "boxes": [
            {
                "x1": float(b["x1"]),
                "y1": float(b["y1"]),
                "x2": float(b["x2"]),
                "y2": float(b["y2"]),
            }
            for b in boxes
        ],
    }

    payload_str = json.dumps(payload)

    try:
        result = subprocess.run(
            [CPP_BINARY_PATH],
            input=payload_str,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.error("C++ binary not found at %s — falling back to Python", CPP_BINARY_PATH)
        return _python_fallback(boxes, rois)
    except subprocess.TimeoutExpired:
        logger.error("C++ binary timed out — falling back to Python")
        return _python_fallback(boxes, rois)

    if result.returncode != 0:
        logger.error("C++ binary error: %s", result.stderr)
        return _python_fallback(boxes, rois)

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.error("C++ binary produced invalid JSON: %s", e)
        return _python_fallback(boxes, rois)

    # Flatten into the simple {name: count} dict that detector.py expects,
    # plus metadata keys prefixed with underscore
    counts = output.get("counts", {})
    counts["_elapsed_ms"] = output.get("elapsed_ms", 0.0)
    counts["_threads"]    = output.get("threads_used", 1)
    return counts


def _python_fallback(boxes: list[dict], rois: dict) -> dict:
    """Pure-Python fallback used when the binary is unavailable."""
    counts = {}
    for name, (x1, y1, x2, y2) in rois.items():
        c = 0
        for b in boxes:
            cx = (b["x1"] + b["x2"]) / 2
            cy = (b["y1"] + b["y2"]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                c += 1
        counts[name] = c
    counts["_elapsed_ms"] = 0.0
    counts["_threads"]    = 1
    return counts