"""
Voltage Sweep vs Frequency Amplitude — Colab Edition
=====================================================
Auto-detects the red 7-segment LED display in each frame via HSV masking,
OCRs the voltage, and measures audio amplitude at ~N_SAMPLES timestamps.
"""

# ══════════════════════════════════════════════════════════════════
# CONFIG  ← edit these
# ══════════════════════════════════════════════════════════════════
VIDEO_PATH   = "/content/voltage_sweep.mp4"
TARGET_FREQ  = 1000.0    # Hz to track
BANDWIDTH    = 20.0      # ±Hz around target
N_SAMPLES    = 28        # evenly-spaced frames to analyse (25–30)
OUT_PLOT     = "voltage_sweep_result.png"
OUT_CSV      = "voltage_sweep_data.csv"
# ══════════════════════════════════════════════════════════════════

import re, warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import signal

warnings.filterwarnings("ignore")

# ── deps ───────────────────────────────────────────────────────────
try:
    import pytesseract
    OCR_OK = True
except ImportError:
    raise ImportError("Run:  !pip install pytesseract && !apt-get install -q tesseract-ocr")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except ImportError:
    MOVIEPY_OK = False

# ══════════════════════════════════════════════════════════════════
# 1. AUDIO  →  amplitude at arbitrary timestamps
# ══════════════════════════════════════════════════════════════════

def load_audio(video_path):
    """Return (samples float32, sample_rate). Uses ffmpeg directly — avoids moviepy/numpy compat bugs."""
    import subprocess, tempfile, wave
    tmp = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", "44100", "-vn", tmp, "-loglevel", "error"],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode()}")
    with wave.open(tmp, "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    Path(tmp).unlink(missing_ok=True)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"[Audio] ffmpeg  sr={sr} Hz  {len(audio)/sr:.1f}s")
    return audio, sr


def amplitude_at_time(audio, sr, t_sec, target_freq, bandwidth, window_sec=0.1):
    """
    RMS amplitude in [target_freq ± bandwidth/2] over a short window centred at t_sec.
    Returns dBFS value.
    """
    half   = int(window_sec * sr / 2)
    centre = int(t_sec * sr)
    lo     = max(0, centre - half)
    hi     = min(len(audio), centre + half)
    chunk  = audio[lo:hi]
    if len(chunk) < 64:
        return np.nan

    n      = max(256, 1 << (len(chunk) - 1).bit_length())
    fft    = np.fft.rfft(chunk * np.hanning(len(chunk)), n=n)
    freqs  = np.fft.rfftfreq(n, 1/sr)
    mask   = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
    if not mask.any():
        return np.nan
    power  = (np.abs(fft[mask]) ** 2).mean()
    rms    = np.sqrt(power) / (n / 2)
    return float(20 * np.log10(max(rms, 1e-12)))


# ══════════════════════════════════════════════════════════════════
# 2. RED LED DISPLAY DETECTION
# ══════════════════════════════════════════════════════════════════

def find_led_roi(frame_bgr):
    """
    Detect the bounding box of the red 7-segment display using HSV red masking.
    Returns (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around 0°/180° in HSV
    mask1 = cv2.inRange(hsv, np.array([0,  120, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165,120, 80]), np.array([180,255, 255]))
    red   = cv2.bitwise_or(mask1, mask2)

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red    = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red    = cv2.morphologyEx(red, cv2.MORPH_OPEN,  kernel)

    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Pick the largest contour whose aspect ratio looks like a digit display
    best = None
    best_area = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar   = w / max(h, 1)
        if area > best_area and 1.0 < ar < 8.0 and w > 30 and h > 10:
            best_area = area
            best      = (x, y, w, h)
    return best


def ocr_led(frame_bgr, roi):
    """
    Crop ROI, isolate red channel (brightest for red LEDs),
    threshold, OCR with Tesseract digit mode.
    """
    x, y, w, h = roi
    pad  = 6
    crop = frame_bgr[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]

    # Red channel is strongest for red LEDs on dark background
    r_chan  = crop[:, :, 2]                         # BGR → R channel
    scaled  = cv2.resize(r_chan, None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
    _, thr  = cv2.threshold(scaled, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tesseract: single line, digits only
    cfg     = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    text    = pytesseract.image_to_string(thr, config=cfg).strip()
    nums    = re.findall(r"\d+", text)
    if not nums:
        return None
    v = float(nums[0])
    return v if 0 <= v <= 400 else None


# ══════════════════════════════════════════════════════════════════
# 3. SAMPLE N_SAMPLES FRAMES
# ══════════════════════════════════════════════════════════════════

def sample_frames(video_path, n_samples):
    """
    Evenly space n_samples frame indices across the video.
    Returns list of (time_sec, voltage, frame_bgr).
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"[Video] {total} frames @ {fps:.1f} fps  ({duration:.1f}s)")

    indices  = np.linspace(0, total - 1, n_samples, dtype=int)
    results  = []
    cached_roi = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        t = idx / fps

        # Detect ROI on first frame, reuse thereafter (display is static)
        if cached_roi is None:
            cached_roi = find_led_roi(frame)
            if cached_roi:
                print(f"[ROI]  Detected LED display at {cached_roi}")
            else:
                print("[ROI]  WARNING: LED display not auto-detected in this frame. "
                      "Will retry each frame.")

        roi = cached_roi or find_led_roi(frame)
        v   = ocr_led(frame, roi) if roi else None

        results.append((t, v, frame))
        status = f"{v:.0f}V" if v is not None else "?"
        print(f"  frame {idx:5d}  t={t:6.2f}s  voltage={status}")

    cap.release()
    return results, fps


# ══════════════════════════════════════════════════════════════════
# 4. PLOT
# ══════════════════════════════════════════════════════════════════

def plot_results(times, voltages, amplitudes, target_freq, out):
    # Remove NaN pairs
    mask  = ~(np.isnan(voltages) | np.isnan(amplitudes))
    t     = np.array(times)[mask]
    v     = np.array(voltages)[mask]
    a     = np.array(amplitudes)[mask]

    if len(v) < 3:
        print("[Plot] Not enough valid points to plot.")
        return

    corr = np.corrcoef(v, a)[0, 1]
    print(f"\n[Stats] n={len(v)}  Pearson r = {corr:.4f}")

    # Trend line
    p    = np.polyfit(v, a, 1)
    vfit = np.linspace(v.min(), v.max(), 200)
    afit = np.polyval(p, vfit)

    fig  = plt.figure(figsize=(13, 9), facecolor="#0f0f1a")
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def sax(ax, title, xl, yl):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel(xl, color="#aaaacc", fontsize=9)
        ax.set_ylabel(yl, color="#aaaacc", fontsize=9)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#444466")
        ax.grid(True, color="#2a2a4a", lw=0.6)

    # ── (A) Voltage over time ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, v, "o-", color="#00d4ff", lw=1.5, ms=6)
    sax(ax1, "Voltage Over Time", "Time (s)", "Voltage (V)")

    # ── (B) Amplitude over time ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, a, "o-", color="#ff6b9d", lw=1.5, ms=6)
    sax(ax2, f"Amplitude @ {target_freq} Hz Over Time",
        "Time (s)", "Amplitude (dBFS)")

    # ── (C) Voltage vs Amplitude — main result ───────────────────
    ax3 = fig.add_subplot(gs[1, :])
    sc  = ax3.scatter(v, a, c=t, cmap="plasma", s=80, zorder=3,
                      edgecolors="#ffffff44", linewidths=0.5)
    ax3.plot(vfit, afit, "--", color="#a8ff78", lw=1.5, alpha=0.8,
             label=f"Linear fit  r={corr:.3f}")
    cb  = plt.colorbar(sc, ax=ax3, pad=0.01)
    cb.set_label("Time (s)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Annotate each point with its voltage value
    for xi, yi, vi_ in zip(v, a, v):
        ax3.annotate(f"{vi_:.0f}V", (xi, yi),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=6.5, color="#ccccee", alpha=0.85)

    sax(ax3, f"Voltage vs Audio Amplitude @ {target_freq} Hz  (n={len(v)})",
        "Voltage (V)", "Amplitude (dBFS)")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    fig.suptitle(
        f"Voltage Sweep  ×  Frequency Amplitude @ {target_freq} Hz",
        color="white", fontsize=14, y=0.98
    )
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"[Plot] Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# 5. EXPORT CSV
# ══════════════════════════════════════════════════════════════════

def export_csv(times, voltages, amplitudes, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "voltage_V", "amplitude_dBFS"])
        for t, v, a in zip(times, voltages, amplitudes):
            w.writerow([
                f"{t:.4f}",
                f"{v:.1f}" if v is not None else "",
                f"{a:.3f}" if not np.isnan(a) else ""
            ])
    print(f"[CSV] Saved → {path}")


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*52}")
print(f"  Video  : {VIDEO_PATH}")
print(f"  Target : {TARGET_FREQ} Hz  ±{BANDWIDTH/2} Hz")
print(f"  Samples: {N_SAMPLES}")
print(f"{'='*52}\n")

# Load full audio once
audio, sr = load_audio(VIDEO_PATH)

# Sample frames + OCR
samples, fps = sample_frames(VIDEO_PATH, N_SAMPLES)

# For each sample, compute amplitude
times_out, volts_out, amps_out = [], [], []
print("\n[Amplitude]")
for t, v, _ in samples:
    amp = amplitude_at_time(audio, sr, t, TARGET_FREQ, BANDWIDTH)
    print(f"  t={t:6.2f}s  V={v}  amp={amp:.2f} dBFS")
    times_out.append(t)
    volts_out.append(v if v is not None else np.nan)
    amps_out.append(amp)

# Save + plot
export_csv(times_out, volts_out, amps_out, OUT_CSV)
plot_results(times_out, volts_out, amps_out, TARGET_FREQ, OUT_PLOT)
print("\nDone ✓")
