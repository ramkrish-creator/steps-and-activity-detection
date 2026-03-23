"""
=============================================================================
  ACCELEROMETER SIGNAL PROCESSING — STEP COUNTER & ACTIVITY CLASSIFIER
  Signal Processing for Machine Learning | Micro Project
=============================================================================
  Techniques used:
    - Butterworth Low-Pass Filter (noise removal)
    - Peak Detection (step counting)
    - FFT / Fourier Transform (dominant frequency analysis)
    - Autocorrelation (periodicity detection)
    - Wavelet Transform (activity feature extraction)
    - Statistical feature extraction
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.fft import fft, fftfreq
import pywt
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE = "data.txt"          # <-- change to your actual file path
SAMPLING_RATE = 20              # Hz — WISDM dataset is ~20 Hz
WINDOW_SIZE_SEC = 2.56          # seconds per analysis window
WINDOW_SAMPLES = int(WINDOW_SIZE_SEC * SAMPLING_RATE)  # 51 samples/window

ACTIVITY_MAP = {
    'A': 'Walking',
    'B': 'Jogging',
    'C': 'Stairs',
    'D': 'Sitting',
    'E': 'Standing',
    'F': 'Typing',
    'G': 'Teeth Brushing',
    'H': 'Soup',
    'I': 'Chips',
    'J': 'Pasta',
    'K': 'Drinking',
    'L': 'Sandwich',
    'M': 'Kicking',
    'O': 'Catch',
    'P': 'Dribbling',
    'Q': 'Writing',
    'R': 'Clapping',
    'S': 'Folding'
}

# Activities considered "locomotive" (involve steps)
LOCOMOTIVE_ACTIVITIES = {'A', 'B', 'C', 'M', 'P'}

# Frequency band definitions for activity detection
FREQ_BANDS = {
    'Standing/Sitting': (0.0, 0.5),
    'Walking':          (0.5, 3.0),
    'Running/Jogging':  (3.0, 8.0),
    'High Motion':      (8.0, 10.0),
}


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(filepath):
    """
    Load WISDM-style accelerometer data.
    Format: user_id, activity, timestamp, x, y, z;
    """
    print(f"\n{'='*60}")
    print("  LOADING DATA")
    print(f"{'='*60}")

    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                user_id  = parts[0].strip()
                activity = parts[1].strip()
                timestamp = int(parts[2].strip())
                x = float(parts[3].strip())
                y = float(parts[4].strip())
                z = float(parts[5].strip())
                rows.append([user_id, activity, timestamp, x, y, z])
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows, columns=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'])

    # Compute magnitude of acceleration vector
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    print(f"  Records loaded  : {len(df):,}")
    print(f"  Users           : {df['user_id'].nunique()}")
    print(f"  Activities      : {df['activity'].nunique()}")
    print(f"  Activity labels : {sorted(df['activity'].unique())}")
    print(f"  X range         : [{df['x'].min():.3f}, {df['x'].max():.3f}]")
    print(f"  Y range         : [{df['y'].min():.3f}, {df['y'].max():.3f}]")
    print(f"  Z range         : [{df['z'].min():.3f}, {df['z'].max():.3f}]")

    return df


# =============================================================================
# 2. SIGNAL FILTERING — Butterworth Low-Pass Filter
# =============================================================================

def butter_lowpass_filter(data, cutoff=5.0, fs=SAMPLING_RATE, order=4):
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.
    Cutoff at 5 Hz retains walking/running frequencies.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    normal_cutoff = min(normal_cutoff, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def preprocess_segment(segment):
    """Filter all axes and recompute magnitude."""
    seg = segment.copy()
    for axis in ['x', 'y', 'z']:
        if len(seg) >= 13:  # filtfilt requires padlen
            seg[axis] = butter_lowpass_filter(seg[axis].values)
    seg['magnitude'] = np.sqrt(seg['x']**2 + seg['y']**2 + seg['z']**2)
    return seg


# =============================================================================
# 3. STEP COUNTING — Peak Detection on Filtered Magnitude
# =============================================================================

def count_steps(magnitude, fs=SAMPLING_RATE):
    """
    Count steps by detecting peaks in the acceleration magnitude.

    A step produces a characteristic impact peak. We:
    1. Remove DC offset (mean subtraction)
    2. Find peaks with minimum height and minimum separation
       (~0.3 s apart = max ~3.3 steps/sec which is brisk running)
    """
    signal_centered = magnitude - np.mean(magnitude)
    std = np.std(signal_centered)

    # Peaks must be > 0.3 * std above mean and at least 0.3 s apart
    min_distance = int(0.3 * fs)
    height_thresh = 0.3 * std

    peaks, props = find_peaks(
        signal_centered,
        height=height_thresh,
        distance=min_distance,
        prominence=0.1 * std
    )
    return len(peaks), peaks


# =============================================================================
# 4. FOURIER TRANSFORM — Dominant Frequency Analysis
# =============================================================================

def compute_fft(magnitude, fs=SAMPLING_RATE):
    """
    Compute FFT of the magnitude signal.
    Returns frequencies, power spectrum, and dominant frequency.
    """
    n = len(magnitude)
    signal_centered = magnitude - np.mean(magnitude)

    fft_vals = fft(signal_centered)
    freqs = fftfreq(n, d=1.0/fs)

    # Keep only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask])**2

    dominant_freq = freqs_pos[np.argmax(power)] if len(power) > 0 else 0.0
    return freqs_pos, power, dominant_freq


# =============================================================================
# 5. POWER SPECTRAL DENSITY — Welch Method
# =============================================================================

def compute_psd(magnitude, fs=SAMPLING_RATE):
    """
    Welch PSD gives a smoother frequency power estimate than raw FFT.
    Good for longer signals.
    """
    nperseg = min(len(magnitude), 64)
    freqs, psd = welch(magnitude - np.mean(magnitude), fs=fs, nperseg=nperseg)
    return freqs, psd


# =============================================================================
# 6. AUTOCORRELATION — Periodicity / Rhythm Detection
# =============================================================================

def compute_autocorrelation(magnitude):
    """
    Autocorrelation reveals periodic patterns (regular steps).
    A high peak at a lag corresponding to step period → regular walking.
    Returns normalized autocorrelation and lag array.
    """
    sig = magnitude - np.mean(magnitude)
    n = len(sig)
    acf = np.correlate(sig, sig, mode='full')
    acf = acf[n-1:]          # keep lags 0 → n-1
    acf /= acf[0] + 1e-10   # normalize to 1

    lags = np.arange(len(acf))
    return lags, acf


def autocorrelation_step_estimate(magnitude, fs=SAMPLING_RATE):
    """
    Estimate step frequency from autocorrelation.
    Find first prominent peak after lag=0 — this is the step period.
    """
    lags, acf = compute_autocorrelation(magnitude)

    # Search lags between 0.2 s and 2 s (0.5–5 Hz step range)
    min_lag = int(0.2 * fs)
    max_lag = int(2.0 * fs)
    search = acf[min_lag:max_lag]

    if len(search) == 0:
        return 0.0

    peaks, _ = find_peaks(search, height=0.1)
    if len(peaks) == 0:
        return 0.0

    first_peak_lag = peaks[0] + min_lag
    step_period_sec = first_peak_lag / fs
    step_freq_hz = 1.0 / step_period_sec if step_period_sec > 0 else 0.0
    return step_freq_hz


# =============================================================================
# 7. WAVELET TRANSFORM — Multi-Resolution Energy Features
# =============================================================================

def compute_wavelet_features(magnitude, wavelet='db4', level=4):
    """
    Discrete Wavelet Transform decomposes signal into frequency sub-bands.
    Energy in each sub-band is a discriminative feature for activity type.

    Level decomposition at 20 Hz:
      D1 ~  5–10 Hz  (very fast motion / noise)
      D2 ~ 2.5–5 Hz  (running / fast walk)
      D3 ~ 1.25–2.5 Hz (normal walk)
      D4 ~ 0.6–1.25 Hz (slow walk)
      A4 ~  0–0.6 Hz  (quasi-static / standing)
    """
    max_level = pywt.dwt_max_level(len(magnitude), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(magnitude, wavelet, level=level)
    energies = [np.sum(c**2) / (len(c) + 1e-10) for c in coeffs]
    total_energy = sum(energies) + 1e-10
    relative_energies = [e / total_energy for e in energies]

    labels = [f'A{level}'] + [f'D{level - i}' for i in range(level)]
    return dict(zip(labels, relative_energies)), coeffs


# =============================================================================
# 8. ACTIVITY CLASSIFICATION — Pure Signal Processing (No ML)
# =============================================================================

def classify_activity(dominant_freq, acf_step_freq, wavelet_features,
                       magnitude_std, magnitude_mean):
    """
    Rule-based classifier using signal processing features.

    Decision hierarchy:
    1. Very low variance → Sitting / Standing
    2. Dominant frequency bands from FFT → Walking / Running
    3. Autocorrelation step frequency confirms rhythmic motion
    4. Wavelet energy distribution refines classification
    """

    # Feature: overall motion level
    motion_level = magnitude_std

    # --- RULE 1: Static / Near-static ---
    if motion_level < 0.5:
        if magnitude_mean < 11.0:
            return "Sitting", "Very low variance, near-gravity magnitude"
        else:
            return "Standing", "Very low variance, upright magnitude (~9.8 m/s²)"

    # --- RULE 2: Frequency-based classification ---
    # Use the higher of FFT dominant freq and autocorrelation step freq
    step_freq = max(dominant_freq, acf_step_freq)

    # Wavelet energy in high-frequency detail bands
    high_freq_energy = wavelet_features.get('D1', 0) + wavelet_features.get('D2', 0)
    low_freq_energy  = wavelet_features.get('D3', 0) + wavelet_features.get('D4', 0)

    if step_freq < 0.3 and motion_level < 1.5:
        return "Standing", "Minimal periodic signal, low variance"

    elif 0.3 <= step_freq < 1.4:
        if low_freq_energy > 0.3:
            return "Walking (slow)", f"Step freq ~{step_freq:.2f} Hz, low-band wavelet energy dominant"
        else:
            return "Walking", f"Step freq ~{step_freq:.2f} Hz"

    elif 1.4 <= step_freq < 3.5:
        if high_freq_energy > 0.25:
            return "Jogging/Running", f"Step freq ~{step_freq:.2f} Hz, high-band wavelet energy"
        else:
            return "Brisk Walking", f"Step freq ~{step_freq:.2f} Hz"

    elif step_freq >= 3.5:
        return "Running (fast)", f"Step freq ~{step_freq:.2f} Hz, high cadence"

    else:
        if motion_level > 3.0:
            return "Dynamic Activity", f"High variance ({motion_level:.2f}), unclear periodicity"
        return "Standing / Other", "Low periodic signal"


# =============================================================================
# 9. WINDOW-BASED ANALYSIS PIPELINE
# =============================================================================

def analyze_windows(df, user_id=None, activity_filter=None):
    """
    Slide a fixed-size window over the signal and extract all features per window.
    Returns a DataFrame of results.
    """
    # Filter
    data = df.copy()
    if user_id:
        data = data[data['user_id'] == str(user_id)]
    if activity_filter:
        data = data[data['activity'] == activity_filter]

    data = data.sort_values('timestamp').reset_index(drop=True)

    if len(data) < WINDOW_SAMPLES:
        print(f"  [WARNING] Not enough samples ({len(data)}) for even one window.")
        return pd.DataFrame()

    results = []
    n_windows = len(data) // WINDOW_SAMPLES

    print(f"\n  Analyzing {n_windows} windows of {WINDOW_SAMPLES} samples each...")

    for i in range(n_windows):
        start = i * WINDOW_SAMPLES
        end   = start + WINDOW_SAMPLES
        window = data.iloc[start:end].copy()

        # Preprocess (filter)
        window = preprocess_segment(window)
        mag = window['magnitude'].values

        # True label (majority vote in window)
        true_label = window['activity'].mode()[0]
        true_activity = ACTIVITY_MAP.get(true_label, true_label)

        # --- Signal Features ---
        # FFT
        freqs, power, dom_freq = compute_fft(mag)

        # Autocorrelation step frequency
        acf_step_freq = autocorrelation_step_estimate(mag)

        # Wavelet features
        wavelet_feats, _ = compute_wavelet_features(mag)

        # Statistical features
        mag_std  = np.std(mag)
        mag_mean = np.mean(mag)
        mag_rms  = np.sqrt(np.mean(mag**2))

        # Step count (peak detection)
        n_steps, peak_idxs = count_steps(mag)
        duration_sec = WINDOW_SAMPLES / SAMPLING_RATE
        step_rate_per_min = (n_steps / duration_sec) * 60

        # Activity classification
        pred_activity, reason = classify_activity(
            dom_freq, acf_step_freq, wavelet_feats, mag_std, mag_mean
        )

        results.append({
            'window':           i + 1,
            'true_label':       true_label,
            'true_activity':    true_activity,
            'predicted':        pred_activity,
            'dominant_freq_hz': round(dom_freq, 3),
            'acf_step_freq_hz': round(acf_step_freq, 3),
            'steps_in_window':  n_steps,
            'step_rate_per_min':round(step_rate_per_min, 1),
            'mag_mean':         round(mag_mean, 4),
            'mag_std':          round(mag_std, 4),
            'mag_rms':          round(mag_rms, 4),
            'wavelet_A4':       round(wavelet_feats.get('A4', 0), 4),
            'wavelet_D4':       round(wavelet_feats.get('D4', 0), 4),
            'wavelet_D3':       round(wavelet_feats.get('D3', 0), 4),
            'wavelet_D2':       round(wavelet_feats.get('D2', 0), 4),
            'wavelet_D1':       round(wavelet_feats.get('D1', 0), 4),
            'reason':           reason,
        })

    return pd.DataFrame(results)


# =============================================================================
# 10. AGGREGATE RESULTS — Total Steps & Overall Activity
# =============================================================================

def aggregate_results(results_df):
    """
    Summarize across all windows:
    - Total steps
    - Dominant predicted activity
    - Per-activity step counts
    """
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*60}")

    total_steps = results_df['steps_in_window'].sum()
    dominant_activity = results_df['predicted'].mode()[0]
    avg_step_rate = results_df[results_df['steps_in_window'] > 0]['step_rate_per_min'].mean()

    print(f"  Total windows analyzed  : {len(results_df)}")
    print(f"  Total steps counted     : {total_steps}")
    print(f"  Avg step rate           : {avg_step_rate:.1f} steps/min")
    print(f"  Dominant activity (pred): {dominant_activity}")
    print()

    # Per true-activity breakdown
    print("  Per-Activity Summary (based on true labels):")
    print(f"  {'Activity':<20} {'Windows':>8} {'Total Steps':>12} {'Avg Dom Freq':>14}")
    print(f"  {'-'*56}")
    for act_code, grp in results_df.groupby('true_label'):
        act_name = ACTIVITY_MAP.get(act_code, act_code)
        print(f"  {act_name:<20} {len(grp):>8} {grp['steps_in_window'].sum():>12} "
              f"{grp['dominant_freq_hz'].mean():>13.3f} Hz")

    return total_steps, dominant_activity, avg_step_rate


# =============================================================================
# 11. VISUALIZATION
# =============================================================================

def plot_raw_vs_filtered(df, user_id=None, activity='A', n_samples=200):
    """Plot raw vs filtered signal for one activity."""
    data = df[df['activity'] == activity]
    if user_id:
        data = data[data['user_id'] == str(user_id)]
    data = data.head(n_samples)
    if len(data) < 10:
        return

    mag_raw      = data['magnitude'].values
    mag_filtered = butter_lowpass_filter(mag_raw)
    t = np.arange(len(mag_raw)) / SAMPLING_RATE

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f'Raw vs Filtered Acceleration Magnitude — {ACTIVITY_MAP.get(activity, activity)}',
                 fontsize=14, fontweight='bold')

    axes[0].plot(t, mag_raw, color='#e74c3c', alpha=0.8, lw=1.2, label='Raw')
    axes[0].set_ylabel('Magnitude (m/s²)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t, mag_filtered, color='#2980b9', lw=1.5, label='Filtered (Butterworth LP)')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Magnitude (m/s²)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_raw_vs_filtered.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 01_raw_vs_filtered.png")


def plot_step_detection(df, user_id=None, activity='A', n_samples=200):
    """Show step peaks on filtered signal."""
    data = df[df['activity'] == activity]
    if user_id:
        data = data[data['user_id'] == str(user_id)]
    data = data.head(n_samples)
    if len(data) < 10:
        return

    mag = butter_lowpass_filter(data['magnitude'].values)
    n_steps, peaks = count_steps(mag)
    t = np.arange(len(mag)) / SAMPLING_RATE

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, mag, color='#2c3e50', lw=1.3, label='Filtered Magnitude')
    ax.plot(t[peaks], mag[peaks], 'rv', ms=9, label=f'Steps detected: {n_steps}')
    ax.axhline(np.mean(mag), color='gray', ls='--', alpha=0.5, label='Mean')
    ax.set_title(f'Step Detection via Peak Finding — {ACTIVITY_MAP.get(activity, activity)}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Magnitude (m/s²)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_step_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 02_step_detection.png")


def plot_fft_comparison(df, activities=['A', 'B', 'E']):
    """Compare FFT spectra of different activities."""
    fig, axes = plt.subplots(1, len(activities), figsize=(14, 4), sharey=False)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, act in enumerate(activities):
        data = df[df['activity'] == act].head(WINDOW_SAMPLES * 3)
        if len(data) < WINDOW_SAMPLES:
            continue

        mag = butter_lowpass_filter(data['magnitude'].values[:WINDOW_SAMPLES])
        freqs, power, dom_freq = compute_fft(mag)

        ax = axes[idx] if len(activities) > 1 else axes
        ax.plot(freqs, power / (power.max() + 1e-10),
                color=colors[idx % len(colors)], lw=1.8)
        ax.axvline(dom_freq, color='red', ls='--', alpha=0.8,
                   label=f'Dom: {dom_freq:.2f} Hz')
        ax.set_title(ACTIVITY_MAP.get(act, act), fontweight='bold')
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Normalized Power')
        ax.set_xlim(0, 10); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('FFT Power Spectrum — Activity Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_fft_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 03_fft_comparison.png")


def plot_autocorrelation(df, activities=['A', 'B', 'E']):
    """Plot autocorrelation for different activities."""
    fig, axes = plt.subplots(1, len(activities), figsize=(14, 4), sharey=False)
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, act in enumerate(activities):
        data = df[df['activity'] == act].head(WINDOW_SAMPLES * 2)
        if len(data) < WINDOW_SAMPLES:
            continue

        mag = butter_lowpass_filter(data['magnitude'].values[:WINDOW_SAMPLES])
        lags, acf = compute_autocorrelation(mag)
        t_lags = lags / SAMPLING_RATE

        ax = axes[idx] if len(activities) > 1 else axes
        ax.plot(t_lags[:int(3 * SAMPLING_RATE)],
                acf[:int(3 * SAMPLING_RATE)],
                color=colors[idx % len(colors)], lw=1.6)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_title(ACTIVITY_MAP.get(act, act), fontweight='bold')
        ax.set_xlabel('Lag (s)'); ax.set_ylabel('Autocorrelation')
        ax.grid(alpha=0.3)

    fig.suptitle('Autocorrelation — Periodicity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 04_autocorrelation.png")


def plot_wavelet(df, activity='A'):
    """Plot wavelet decomposition of a signal segment."""
    data = df[df['activity'] == activity].head(WINDOW_SAMPLES)
    if len(data) < WINDOW_SAMPLES:
        return

    mag = butter_lowpass_filter(data['magnitude'].values)
    _, coeffs = compute_wavelet_features(mag, level=4)
    wavelet_feats, _ = compute_wavelet_features(mag, level=4)

    levels = ['A4 (0–0.6 Hz)', 'D4 (0.6–1.25 Hz)',
              'D3 (1.25–2.5 Hz)', 'D2 (2.5–5 Hz)', 'D1 (5–10 Hz)']
    n = len(coeffs)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 10))

    axes[0].plot(mag, color='#2c3e50', lw=1.3)
    axes[0].set_title(f'Wavelet Decomposition (db4) — {ACTIVITY_MAP.get(activity, activity)}',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Original')
    axes[0].grid(alpha=0.3)

    colors = ['#8e44ad', '#2980b9', '#27ae60', '#e67e22', '#e74c3c']
    for i, (coeff, label) in enumerate(zip(coeffs, levels)):
        t = np.linspace(0, len(mag) / SAMPLING_RATE, len(coeff))
        axes[i + 1].plot(t, coeff, color=colors[i % len(colors)], lw=1.3)
        axes[i + 1].set_ylabel(label, fontsize=8)
        axes[i + 1].grid(alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('05_wavelet.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 05_wavelet.png")


def plot_wavelet_energy_bars(df, activities=['A', 'B', 'E']):
    """Bar chart of wavelet sub-band energies per activity."""
    sub_bands = ['A4', 'D4', 'D3', 'D2', 'D1']
    freq_labels = ['A4\n0–0.6Hz', 'D4\n0.6–1.25Hz', 'D3\n1.25–2.5Hz', 'D2\n2.5–5Hz', 'D1\n5–10Hz']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    energies_all = {}
    for act in activities:
        data = df[df['activity'] == act].head(WINDOW_SAMPLES)
        if len(data) < 20:
            continue
        mag = butter_lowpass_filter(data['magnitude'].values)
        feats, _ = compute_wavelet_features(mag, level=4)
        energies_all[ACTIVITY_MAP.get(act, act)] = [feats.get(sb, 0) for sb in sub_bands]

    if not energies_all:
        return

    x = np.arange(len(sub_bands))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (act_name, energies) in enumerate(energies_all.items()):
        ax.bar(x + i * width, energies, width, label=act_name,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(freq_labels)
    ax.set_ylabel('Relative Energy')
    ax.set_title('Wavelet Sub-Band Energy Distribution by Activity', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('06_wavelet_energy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 06_wavelet_energy.png")


def plot_results_summary(results_df):
    """Summary dashboard of window-by-window results."""
    if results_df.empty:
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Signal Processing Analysis — Results Dashboard',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (A) Steps per window
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(results_df['window'], results_df['steps_in_window'],
            color='#3498db', alpha=0.8)
    ax1.set_xlabel('Window #'); ax1.set_ylabel('Steps')
    ax1.set_title('Steps Detected Per Window')
    ax1.grid(axis='y', alpha=0.3)

    # (B) Dominant frequency per window
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results_df['window'], results_df['dominant_freq_hz'],
                c='#e74c3c', s=30, alpha=0.7)
    ax2.axhline(0.5, ls='--', color='gray', alpha=0.5, label='0.5 Hz')
    ax2.axhline(3.0, ls='--', color='orange', alpha=0.5, label='3.0 Hz')
    ax2.set_xlabel('Window #'); ax2.set_ylabel('Freq (Hz)')
    ax2.set_title('Dominant Frequency')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # (C) Predicted activity distribution
    ax3 = fig.add_subplot(gs[1, 0])
    pred_counts = results_df['predicted'].value_counts()
    ax3.barh(pred_counts.index, pred_counts.values, color='#9b59b6', alpha=0.85)
    ax3.set_xlabel('Windows'); ax3.set_title('Predicted Activity Distribution')
    ax3.grid(axis='x', alpha=0.3)

    # (D) Magnitude std over time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results_df['window'], results_df['mag_std'], color='#27ae60', lw=1.5)
    ax4.fill_between(results_df['window'], results_df['mag_std'],
                     alpha=0.3, color='#27ae60')
    ax4.set_xlabel('Window #'); ax4.set_ylabel('Std Dev (m/s²)')
    ax4.set_title('Magnitude Variability Over Time')
    ax4.grid(alpha=0.3)

    # (E) Wavelet energy heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    wv_cols = ['wavelet_A4', 'wavelet_D4', 'wavelet_D3', 'wavelet_D2', 'wavelet_D1']
    wv_data = results_df[wv_cols].values.T
    im = ax5.imshow(wv_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax5.set_yticks(range(5))
    ax5.set_yticklabels(['A4', 'D4', 'D3', 'D2', 'D1'])
    ax5.set_xlabel('Window #'); ax5.set_title('Wavelet Energy Heatmap')
    plt.colorbar(im, ax=ax5, fraction=0.04)

    plt.savefig('07_results_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: 07_results_dashboard.png")


# =============================================================================
# 12. MAIN ENTRY POINT
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  ACCELEROMETER SIGNAL PROCESSING PROJECT")
    print("  Step Counter + Activity Classifier")
    print("="*60)

    # --- Load ---
    df = load_data(DATA_FILE)

    # --- Choose user and activities to analyze ---
    # For demo: pick first user and first two locomotive activities available
    first_user = df['user_id'].iloc[0]
    available_acts = df['activity'].unique().tolist()

    print(f"\n  Demo user: {first_user}")
    print(f"  Available activities: {available_acts}")

    # Pick activities for comparison plots (prefer A/B/E if available)
    compare_acts = [a for a in ['A', 'B', 'E'] if a in available_acts][:3]
    if len(compare_acts) < 2:
        compare_acts = available_acts[:3]

    primary_act = compare_acts[0]

    # --- Plots ---
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    plot_raw_vs_filtered(df, user_id=first_user, activity=primary_act)
    plot_step_detection(df, user_id=first_user, activity=primary_act)
    plot_fft_comparison(df, activities=compare_acts)
    plot_autocorrelation(df, activities=compare_acts)
    plot_wavelet(df, activity=primary_act)
    plot_wavelet_energy_bars(df, activities=compare_acts)

    # --- Window Analysis ---
    print(f"\n{'='*60}")
    print("  WINDOW-BY-WINDOW ANALYSIS")
    print(f"{'='*60}")

    results = analyze_windows(df, user_id=first_user)

    if not results.empty:
        # Print per-window table
        print(f"\n  {'Win':>4} {'True':>10} {'Predicted':>22} {'DomFreq':>9} "
              f"{'ACF Freq':>9} {'Steps':>6} {'StepRate/min':>13}")
        print(f"  {'-'*80}")
        for _, row in results.head(30).iterrows():
            print(f"  {int(row['window']):>4} {row['true_activity']:>10} {row['predicted']:>22} "
                  f"{row['dominant_freq_hz']:>9.3f} {row['acf_step_freq_hz']:>9.3f} "
                  f"{int(row['steps_in_window']):>6} {row['step_rate_per_min']:>13.1f}")

        if len(results) > 30:
            print(f"  ... (showing first 30 of {len(results)} windows)")

        # --- Aggregate ---
        total_steps, dominant_activity, avg_step_rate = aggregate_results(results)

        # --- Dashboard ---
        plot_results_summary(results)

        # --- Save CSV ---
        results.to_csv('results.csv', index=False)
        print(f"\n  Full results saved to: results.csv")

        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  User analyzed        : {first_user}")
        print(f"  Total steps counted  : {total_steps}")
        print(f"  Avg step rate        : {avg_step_rate:.1f} steps/min")
        print(f"  Dominant activity    : {dominant_activity}")
        print(f"{'='*60}\n")

    else:
        print("  No results to display — check data file and parameters.")


if __name__ == "__main__":
    main()
