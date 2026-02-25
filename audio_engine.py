"""
MiXL Audio Engine
DSP-based audio analysis: BPM, Key Detection, Camelot Wheel,
Energy Analysis, Time-Stretching, Crossfade Mixing
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import wave
import struct
import os
import json
from pathlib import Path


# ─── Krumhansl-Schmuckler Key Profiles ───────────────────────────────────────

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot Wheel mapping: (note, mode) -> camelot code
CAMELOT = {
    ('B',  'major'): '1B', ('F#', 'major'): '2B', ('C#', 'major'): '3B',
    ('G#', 'major'): '4B', ('D#', 'major'): '5B', ('A#', 'major'): '6B',
    ('F',  'major'): '7B', ('C',  'major'): '8B', ('G',  'major'): '9B',
    ('D',  'major'): '10B',('A',  'major'): '11B',('E',  'major'): '12B',
    ('G#', 'minor'): '1A', ('D#', 'minor'): '2A', ('A#', 'minor'): '3A',
    ('F',  'minor'): '4A', ('C',  'minor'): '5A', ('G',  'minor'): '6A',
    ('D',  'minor'): '7A', ('A',  'minor'): '8A', ('E',  'minor'): '9A',
    ('B',  'minor'): '10A',('F#', 'minor'): '11A',('C#', 'minor'): '12A',
}

# Enharmonic equivalents
ENHARMONIC = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb',
               'G#': 'Ab', 'A#': 'Bb', 'Db': 'C#',
               'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}


def load_audio(filepath):
    """Load audio file, return (samples_mono_float, sample_rate)"""
    ext = Path(filepath).suffix.lower()
    if ext == '.wav':
        rate, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, rate
    else:
        raise ValueError(f"Unsupported format: {ext}. Please use WAV files.")


def save_audio(filepath, samples, sample_rate):
    """Save float32 audio to WAV"""
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    wavfile.write(filepath, sample_rate, int_samples)


# ─── BPM Detection ────────────────────────────────────────────────────────────

def detect_bpm(samples, sr, min_bpm=60, max_bpm=200):
    """
    Beat tracking via onset strength envelope + autocorrelation.
    Returns estimated BPM (float).
    """
    # Compute onset strength: energy difference between frames
    hop = 512
    frame_len = 2048
    n_frames = (len(samples) - frame_len) // hop + 1

    # Spectral flux onset strength
    prev_spec = None
    onset_env = []
    for i in range(n_frames):
        frame = samples[i * hop: i * hop + frame_len] * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame))
        if prev_spec is not None:
            diff = np.maximum(spec - prev_spec, 0)
            onset_env.append(diff.sum())
        else:
            onset_env.append(0.0)
        prev_spec = spec

    onset_env = np.array(onset_env, dtype=np.float32)

    # Normalize
    if onset_env.max() > 0:
        onset_env /= onset_env.max()

    # Autocorrelation to find beat period
    fps = sr / hop
    min_lag = int(fps * 60 / max_bpm)
    max_lag = int(fps * 60 / min_bpm)

    # Truncate for speed
    env_trim = onset_env[:min(len(onset_env), int(fps * 60))]
    ac = np.correlate(env_trim, env_trim, mode='full')
    ac = ac[len(ac) // 2:]

    # Search in valid BPM range
    search = ac[min_lag:max_lag + 1]
    best_lag = np.argmax(search) + min_lag
    bpm = 60.0 * fps / best_lag

    # Harmonic check: sometimes 2x or 0.5x is more natural
    candidates = [bpm, bpm * 2, bpm / 2]
    candidates = [b for b in candidates if min_bpm <= b <= max_bpm]
    if not candidates:
        candidates = [bpm]

    # Pick the one closest to 120 (most common dance music BPM)
    bpm = min(candidates, key=lambda b: abs(b - 120))
    return round(bpm, 2)


# ─── Key Detection ────────────────────────────────────────────────────────────

def compute_chromagram(samples, sr, n_chroma=12):
    """
    Compute chromagram using STFT + pitch-class binning.
    Returns chroma vector of shape (12,).
    """
    hop = 512
    frame_len = 4096
    n_frames = (len(samples) - frame_len) // hop + 1

    chroma = np.zeros(n_chroma)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)

    # Pitch-class bin for each FFT bin (A4 = 440 Hz)
    A4 = 440.0
    # Map frequency to MIDI note
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = np.where(freqs > 0,
                        12 * np.log2(freqs / (A4 / 2)) + 69,
                        -1)
    pitch_class = (np.round(midi) % 12).astype(int)

    for i in range(n_frames):
        frame = samples[i * hop: i * hop + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame)) ** 2
        for pc in range(n_chroma):
            mask = pitch_class == pc
            chroma[pc] += spec[mask].sum()

    if chroma.sum() > 0:
        chroma /= chroma.sum()
    return chroma


def detect_key(samples, sr):
    """
    Krumhansl-Schmuckler algorithm.
    Returns (key_name, mode, confidence, camelot_code).
    """
    chroma = compute_chromagram(samples, sr)

    best_score = -np.inf
    best_key = 0
    best_mode = 'major'

    for i in range(12):
        # Rotate chroma to align with each key
        rotated = np.roll(chroma, -i)

        # Pearson correlation with major/minor profiles
        maj_corr = np.corrcoef(rotated, KS_MAJOR)[0, 1]
        min_corr = np.corrcoef(rotated, KS_MINOR)[0, 1]

        if maj_corr > best_score:
            best_score = maj_corr
            best_key = i
            best_mode = 'major'
        if min_corr > best_score:
            best_score = min_corr
            best_key = i
            best_mode = 'minor'

    note = NOTE_NAMES[best_key]
    camelot = CAMELOT.get((note, best_mode), '?')

    # Try enharmonic if not found
    if camelot == '?':
        enh = ENHARMONIC.get(note)
        if enh:
            camelot = CAMELOT.get((enh, best_mode), '?')

    confidence = round(float(best_score), 3)
    return note, best_mode, confidence, camelot


# ─── Compatibility Scoring ────────────────────────────────────────────────────

def camelot_distance(c1, c2):
    """
    Compute harmonic distance between two Camelot codes.
    0 = perfect match, 1 = adjacent, higher = less compatible.
    """
    if c1 == '?' or c2 == '?':
        return 5

    # Parse number and letter
    def parse(c):
        letter = c[-1]
        num = int(c[:-1])
        return num, letter

    n1, l1 = parse(c1)
    n2, l2 = parse(c2)

    same_mode = l1 == l2
    num_dist = min(abs(n1 - n2), 12 - abs(n1 - n2))

    if c1 == c2:
        return 0  # Perfect
    if same_mode and num_dist == 1:
        return 1  # Adjacent same mode (very compatible)
    if not same_mode and num_dist == 0:
        return 1  # Relative major/minor
    if same_mode and num_dist == 2:
        return 2
    if not same_mode and num_dist == 1:
        return 2
    return num_dist + (0 if same_mode else 1)


def compute_compatibility(bpm1, key1, mode1, camelot1,
                          bpm2, key2, mode2, camelot2):
    """
    Returns compatibility dict with score 0-100 and breakdown.
    """
    # BPM score
    bpm_diff = abs(bpm1 - bpm2)
    if bpm_diff == 0:
        bpm_score = 100
    elif bpm_diff <= 2:
        bpm_score = 95
    elif bpm_diff <= 5:
        bpm_score = 80
    elif bpm_diff <= 10:
        bpm_score = 60
    elif bpm_diff <= 20:
        bpm_score = 35
    else:
        bpm_score = max(0, 35 - (bpm_diff - 20))

    # Key score
    dist = camelot_distance(camelot1, camelot2)
    key_score_map = {0: 100, 1: 90, 2: 70, 3: 50, 4: 30, 5: 15}
    key_score = key_score_map.get(dist, max(0, 10 - dist * 2))

    # Overall (weighted)
    overall = int(bpm_score * 0.4 + key_score * 0.6)

    # Verdict
    if overall >= 85:
        verdict = "🎯 Perfect Mix"
        color = "#00ff88"
    elif overall >= 70:
        verdict = "✅ Great Mix"
        color = "#88ff00"
    elif overall >= 50:
        verdict = "⚡ Decent Mix"
        color = "#ffcc00"
    elif overall >= 30:
        verdict = "⚠️ Risky Mix"
        color = "#ff8800"
    else:
        verdict = "❌ Clash"
        color = "#ff3344"

    return {
        'overall': overall,
        'bpm_score': bpm_score,
        'key_score': key_score,
        'bpm_diff': round(bpm_diff, 1),
        'camelot_distance': dist,
        'verdict': verdict,
        'verdict_color': color,
    }


# ─── Energy / Transition Zone Analysis ───────────────────────────────────────

def analyze_energy(samples, sr, window_sec=1.0):
    """
    Compute RMS energy curve over time.
    Returns (times_sec, rms_values, transition_zones).
    """
    hop = int(sr * window_sec)
    n_frames = len(samples) // hop
    rms = []
    for i in range(n_frames):
        frame = samples[i * hop: (i + 1) * hop]
        rms.append(np.sqrt(np.mean(frame ** 2)))

    rms = np.array(rms)
    times = np.arange(n_frames) * window_sec

    # Normalize
    if rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        rms_norm = rms

    # Find low-energy zones (breakdowns) — good for transitions
    threshold = np.percentile(rms_norm, 30)
    transition_zones = []
    in_zone = False
    zone_start = 0

    for i, r in enumerate(rms_norm):
        if r <= threshold and not in_zone:
            in_zone = True
            zone_start = i
        elif r > threshold and in_zone:
            in_zone = False
            duration = (i - zone_start) * window_sec
            if duration >= 2.0:  # At least 2 seconds
                transition_zones.append({
                    'start_sec': float(zone_start * window_sec),
                    'end_sec': float(i * window_sec),
                    'duration_sec': round(duration, 1),
                    'avg_energy': round(float(rms_norm[zone_start:i].mean()), 3)
                })

    # Best transition zone = longest low-energy section
    best_zone = None
    if transition_zones:
        best_zone = max(transition_zones, key=lambda z: z['duration_sec'])

    return times.tolist(), rms_norm.tolist(), transition_zones, best_zone


# ─── Time Stretching ─────────────────────────────────────────────────────────

def time_stretch(samples, sr, ratio):
    """
    Phase-vocoder inspired time stretching.
    ratio > 1 = slower, ratio < 1 = faster.
    """
    if abs(ratio - 1.0) < 0.001:
        return samples

    hop_a = 512       # Analysis hop
    hop_s = int(hop_a * ratio)  # Synthesis hop
    win_len = 2048
    window = np.hanning(win_len)

    n_frames = (len(samples) - win_len) // hop_a
    output_len = n_frames * hop_s + win_len
    output = np.zeros(output_len)
    norm = np.zeros(output_len)

    prev_phase = None

    for i in range(n_frames):
        # Analysis frame
        start = i * hop_a
        frame = samples[start: start + win_len]
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))
        frame = frame * window

        # FFT
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        if prev_phase is None:
            prev_phase = phase
            out_phase = phase
        else:
            # Phase advancement
            delta_phase = phase - prev_phase
            prev_phase = phase.copy()

            # Expected phase advance
            freq_idx = np.arange(len(phase))
            expected = 2 * np.pi * freq_idx * hop_a / win_len
            delta_phase = delta_phase - expected
            delta_phase = delta_phase - 2 * np.pi * np.round(delta_phase / (2 * np.pi))
            out_phase = out_phase + expected * (hop_s / hop_a) + delta_phase * (hop_s / hop_a)

        # Reconstruct
        out_spec = mag * np.exp(1j * out_phase)
        out_frame = np.fft.irfft(out_spec) * window

        # Overlap-add
        s_out = i * hop_s
        output[s_out: s_out + win_len] += out_frame
        norm[s_out: s_out + win_len] += window ** 2

    # Normalize OLA
    norm = np.where(norm > 1e-8, norm, 1.0)
    output = output / norm

    return output.astype(np.float32)


# ─── Crossfade Mixing ────────────────────────────────────────────────────────

def crossfade_mix(track1, track2, sr,
                  crossfade_sec=8.0,
                  transition_start_ratio=0.75):
    """
    Mix two tracks with a crossfade.
    transition_start_ratio: where in track1 to begin crossfading (0-1).
    Returns mixed audio.
    """
    crossfade_samples = int(crossfade_sec * sr)
    t1_len = len(track1)
    t2_len = len(track2)

    # Where crossfade starts in track1
    xfade_start = int(t1_len * transition_start_ratio)
    xfade_start = min(xfade_start, t1_len - crossfade_samples)
    xfade_start = max(0, xfade_start)

    # Build output
    pre_fade = track1[:xfade_start]

    # Fade region — ensure same length
    t1_fade = track1[xfade_start: xfade_start + crossfade_samples]
    t2_fade = track2[:crossfade_samples]

    fade_len = min(len(t1_fade), len(t2_fade), crossfade_samples)
    t1_fade = t1_fade[:fade_len]
    t2_fade = t2_fade[:fade_len]

    fade_out = np.linspace(1.0, 0.0, fade_len) ** 1.5
    fade_in = np.linspace(0.0, 1.0, fade_len) ** 1.5
    fade_region = t1_fade * fade_out + t2_fade * fade_in

    # After fade: rest of track2
    post_fade = track2[crossfade_samples:]

    # Total mix duration cap: 90 seconds or full
    result = np.concatenate([pre_fade, fade_region, post_fade])
    return result.astype(np.float32)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def analyze_track(filepath):
    """Analyze a single track. Returns dict of features."""
    samples, sr = load_audio(filepath)

    # Downsample for speed if needed
    if sr > 22050:
        # Simple decimation
        factor = sr // 22050
        samples = samples[::factor]
        sr = sr // factor

    bpm = detect_bpm(samples, sr)
    key, mode, confidence, camelot = detect_key(samples, sr)
    times, rms, zones, best_zone = analyze_energy(samples, sr)

    duration = len(samples) / sr

    return {
        'filepath': filepath,
        'filename': Path(filepath).name,
        'duration_sec': round(duration, 2),
        'sample_rate': sr,
        'bpm': bpm,
        'key': key,
        'mode': mode,
        'key_confidence': confidence,
        'camelot': camelot,
        'energy_times': times,
        'energy_rms': rms,
        'transition_zones': zones,
        'best_transition': best_zone,
    }


def mix_tracks(track1_path, track2_path, output_path,
               crossfade_sec=8.0, normalize=True):
    """
    Full mixing pipeline:
    1. Analyze both tracks
    2. Compute compatibility
    3. Time-stretch track2 to match track1's BPM
    4. Find best transition zone in track1
    5. Crossfade mix
    6. Save output
    Returns full result dict.
    """
    print(f"[MiXL] Analyzing track 1: {track1_path}")
    t1 = analyze_track(track1_path)
    print(f"[MiXL] Analyzing track 2: {track2_path}")
    t2 = analyze_track(track2_path)

    compat = compute_compatibility(
        t1['bpm'], t1['key'], t1['mode'], t1['camelot'],
        t2['bpm'], t2['key'], t2['mode'], t2['camelot']
    )

    # Load raw audio
    s1, sr1 = load_audio(track1_path)
    s2, sr2 = load_audio(track2_path)

    # Resample s2 to sr1 if different (simple)
    if sr2 != sr1:
        # Simple ratio resampling
        ratio_resample = sr1 / sr2
        new_len = int(len(s2) * ratio_resample)
        s2 = np.interp(
            np.linspace(0, len(s2) - 1, new_len),
            np.arange(len(s2)), s2
        ).astype(np.float32)

    # Time-stretch track2 to match track1 BPM
    bpm_ratio = t2['bpm'] / t1['bpm']
    print(f"[MiXL] Time-stretching track 2 by ratio {bpm_ratio:.3f}")
    if abs(bpm_ratio - 1.0) > 0.01:
        s2_stretched = time_stretch(s2, sr1, bpm_ratio)
    else:
        s2_stretched = s2

    # Determine crossfade start from best transition zone of track1
    transition_ratio = 0.75
    if t1['best_transition']:
        bt = t1['best_transition']
        start_sec = bt['start_sec']
        dur1 = t1['duration_sec']
        transition_ratio = min(start_sec / dur1, 0.9)

    print(f"[MiXL] Crossfading at {transition_ratio:.1%} of track 1")
    mixed = crossfade_mix(s1, s2_stretched, sr1,
                          crossfade_sec=crossfade_sec,
                          transition_start_ratio=transition_ratio)

    # Normalize
    if normalize and mixed.max() > 0:
        mixed = mixed / np.abs(mixed).max() * 0.9

    save_audio(output_path, mixed, sr1)
    print(f"[MiXL] Mix saved to {output_path}")

    return {
        'track1': t1,
        'track2': t2,
        'compatibility': compat,
        'mix_duration_sec': round(len(mixed) / sr1, 2),
        'output_path': output_path,
        'crossfade_sec': crossfade_sec,
        'bpm_ratio_applied': round(bpm_ratio, 4),
        'transition_start_ratio': round(transition_ratio, 3),
    }