# tdoa_utils.py
import numpy as np
from scipy.signal import correlate

SPEED_OF_SOUND = 343.0  # m/s

def compute_tdoa(ch1, ch2, sample_rate):
    """Compute time difference of arrival (TDOA) using cross-correlation."""
    ch1 -= np.mean(ch1)
    ch2 -= np.mean(ch2)
    corr = correlate(ch2, ch1, mode="full")
    lags = np.arange(-len(ch1) + 1, len(ch2))
    lag = lags[np.argmax(corr)]
    return lag / sample_rate  # seconds

def tdoa_to_angle(tdoa, mic_spacing):
    """Convert TDOA to 2D angle (degrees)."""
    ratio = (SPEED_OF_SOUND * tdoa) / mic_spacing
    ratio = np.clip(ratio, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))

def estimate_angle(audio_data, mic_spacing, sample_rate):
    """Estimate angle from stereo audio (returns None if mono)."""
    if audio_data.ndim == 1 or audio_data.shape[1] < 2:
        return None
    ch1, ch2 = audio_data[:, 0], audio_data[:, 1]
    tdoa = compute_tdoa(ch1, ch2, sample_rate)
    return tdoa_to_angle(tdoa, mic_spacing)
