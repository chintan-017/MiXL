"""
MiXL — Intelligent DJ Brain v2
================================
Actually listens to what's happening in the audio moment-by-moment,
finds the ideal cut/entry points based on musical content,
then surgically executes the transition.

Real DJ decision chain:
  1. Deep listening  — beat grid, phrase map, energy map, onset density
  2. Content scanning — find the BEST CUT POINT in T1 and BEST ENTRY in T2
  3. Situation assessment — what do the tracks need?
  4. Strategy selection — 7 real DJ techniques
  5. Track surgery — modify T2's entry if needed
  6. Precision execution — sample-accurate, content-aware transition
"""

import numpy as np
import scipy.signal as signal
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEEP LISTENING
# ══════════════════════════════════════════════════════════════════════════════

def compute_beat_grid(samples, sr, bpm):
    """Phase-locked beat grid via onset autocorrelation."""
    beat_period = sr * 60.0 / bpm
    hop = 256
    frame_len = 1024
    n_frames = (len(samples) - frame_len) // hop + 1
    onset_env = np.zeros(n_frames)
    prev_spec = None

    for i in range(n_frames):
        frame = samples[i * hop: i * hop + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame))
        if prev_spec is not None:
            onset_env[i] = np.maximum(spec - prev_spec, 0).sum()
        prev_spec = spec

    if onset_env.max() > 0:
        onset_env /= onset_env.max()

    beat_period_frames = beat_period / hop
    n_test = max(1, int(beat_period_frames))
    scores = np.zeros(n_test)

    for phase in range(n_test):
        positions = np.arange(phase, len(onset_env), beat_period_frames).astype(int)
        positions = positions[positions < len(onset_env)]
        if len(positions) > 0:
            scores[phase] = onset_env[positions].sum()

    best_phase = int(np.argmax(scores) * hop)
    n_beats = int(len(samples) / beat_period) + 1
    beats = [int(best_phase + i * beat_period) for i in range(n_beats)]
    return [b for b in beats if b < len(samples)]


def compute_rms_curve(samples, sr, window_sec=0.5):
    """Fine-grained RMS energy curve. Returns (times, rms) arrays."""
    hop = int(sr * window_sec)
    n = len(samples) // hop
    rms = np.array([
        np.sqrt(np.mean(samples[i*hop:(i+1)*hop]**2))
        for i in range(n)
    ], dtype=np.float32)
    times = np.arange(n) * window_sec
    if rms.max() > 0:
        rms /= rms.max()
    return times, rms


def compute_onset_density(samples, sr, window_sec=1.0):
    """
    Count onsets per second over time — high density = busy section,
    low density = breakdown/sparse section ideal for transitions.
    """
    hop = 512
    frame_len = 1024
    n_frames = (len(samples) - frame_len) // hop + 1
    onset_env = np.zeros(n_frames)
    prev_spec = None

    for i in range(n_frames):
        frame = samples[i*hop: i*hop+frame_len]
        if len(frame) < frame_len:
            break
        frame *= np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame))
        if prev_spec is not None:
            diff = np.maximum(spec - prev_spec, 0).sum()
            onset_env[i] = diff
        prev_spec = spec

    if onset_env.max() > 0:
        onset_env /= onset_env.max()

    # Threshold peaks to count onsets
    threshold = onset_env.mean() + onset_env.std() * 0.5
    is_onset = onset_env > threshold

    fps = sr / hop
    window_frames = int(window_sec * fps)
    density = np.array([
        is_onset[i:i+window_frames].sum() / window_sec
        for i in range(0, len(is_onset), window_frames // 2)
    ], dtype=np.float32)

    return density


def spectral_flux_curve(samples, sr, hop=1024):
    """
    Frame-by-frame spectral flux (rate of spectral change).
    Peaks = transients. Troughs = held notes or silence.
    """
    frame_len = 2048
    n_frames = (len(samples) - frame_len) // hop + 1
    flux = np.zeros(n_frames)
    prev = None
    for i in range(n_frames):
        frame = samples[i*hop: i*hop+frame_len]
        if len(frame) < frame_len:
            break
        frame = frame * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame))
        if prev is not None:
            flux[i] = np.sum((spec - prev)**2)
        prev = spec.copy()
    if flux.max() > 0:
        flux /= flux.max()
    return flux


def spectral_profile(samples, sr):
    """5-band spectral character + centroid."""
    frame_len = 4096
    hop = 2048
    n_frames = max(1, (len(samples) - frame_len) // hop)
    freqs = np.fft.rfftfreq(frame_len, d=1.0/sr)
    bands = {'sub_bass':(20,80),'bass':(80,250),'low_mid':(250,1000),'high_mid':(1000,4000),'air':(4000,20000)}
    band_e = {k: 0.0 for k in bands}
    centroids = []

    for i in range(min(n_frames, 200)):
        frame = samples[i*hop: i*hop+frame_len]
        if len(frame) < frame_len: break
        frame = frame * np.hanning(frame_len)
        mag = np.abs(np.fft.rfft(frame))**2
        total = mag.sum()
        if total < 1e-10: continue
        for band, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_e[band] += mag[mask].sum()
        centroids.append(np.sum(freqs * mag) / (total + 1e-10))

    total_e = sum(band_e.values()) + 1e-10
    profile = {k: round(v/total_e, 4) for k, v in band_e.items()}
    profile['centroid'] = round(float(np.mean(centroids)) if centroids else 0, 1)
    return profile


def analyze_track_sections(samples, sr, beats):
    """Detect intro / body / outro by energy shape."""
    if len(beats) < 4:
        q = len(samples) // 4
        return {
            'intro_end': q, 'intro_end_sec': round(q/sr, 2),
            'outro_start': 3*q, 'outro_start_sec': round(3*q/sr, 2),
            'rms_curve': [], 'n_windows': 0,
        }

    window_beats = 8
    n_windows = len(beats) // window_beats
    rms_curve = []
    for w in range(n_windows):
        sb = w * window_beats
        eb = min((w+1)*window_beats, len(beats))
        chunk = samples[beats[sb]:beats[eb-1]]
        rms_curve.append(np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0)

    rms_curve = np.array(rms_curve)
    if rms_curve.max() == 0:
        rms_curve = np.ones(len(rms_curve))
    peak = rms_curve.max()

    intro_w = 0
    for i, r in enumerate(rms_curve):
        if r >= peak * 0.6:
            intro_w = i; break

    outro_w = n_windows - 1
    for i in range(n_windows-1, -1, -1):
        if rms_curve[i] >= peak * 0.55:
            outro_w = i + 1; break

    intro_beat = min(intro_w * window_beats, len(beats)-1)
    outro_beat  = min(outro_w * window_beats, len(beats)-1)

    return {
        'intro_end': beats[intro_beat],
        'intro_end_sec': round(beats[intro_beat]/sr, 2),
        'outro_start': beats[outro_beat],
        'outro_start_sec': round(beats[outro_beat]/sr, 2),
        'rms_curve': rms_curve.tolist(),
        'n_windows': n_windows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTENT-AWARE CUT POINT FINDER
#    This is the core of what makes MiXL actually listen to the music.
#    Instead of cutting at "75% of track duration", it scans the audio
#    and finds the musically optimal exit/entry points.
# ══════════════════════════════════════════════════════════════════════════════

def score_cut_points(samples, sr, beats, role='exit'):
    """
    Score every 4-beat (bar) boundary in the track for suitability as a cut point.

    For EXIT points (end of T1), we want:
      - Low energy (breakdown, not a drop)
      - Low onset density (sparse, not busy)
      - Low spectral flux (not a transient-heavy moment)
      - Not too early or too late in the track

    For ENTRY points (start of T2), we want:
      - Low energy (natural intro, not mid-chorus)
      - Low onset density
      - Ideally a natural 'buildup start' moment

    Returns list of (beat_index, score, sample_pos, time_sec) sorted best-first.
    """
    if len(beats) < 8:
        mid = len(samples) // 2
        return [(0, 1.0, mid, mid/sr)]

    # Build per-beat features
    beat_period = int(np.mean(np.diff(beats))) if len(beats) > 1 else int(sr * 60.0 / 120)
    window = beat_period * 4  # 4-beat window around each candidate

    # Precompute RMS and flux at beat-level resolution
    beat_rms = []
    beat_flux = []
    prev_spec = None

    for b in beats:
        chunk = samples[b: b + window]
        if len(chunk) < window // 2:
            beat_rms.append(0.0)
            beat_flux.append(0.0)
            prev_spec = None
            continue

        rms = np.sqrt(np.mean(chunk**2))
        beat_rms.append(rms)

        frame = chunk[:min(len(chunk), 2048)] * np.hanning(min(len(chunk), 2048))
        spec = np.abs(np.fft.rfft(frame, n=2048))
        if prev_spec is not None and len(spec) == len(prev_spec):
            beat_flux.append(np.sum((spec - prev_spec)**2))
        else:
            beat_flux.append(0.0)
        prev_spec = spec.copy()

    beat_rms  = np.array(beat_rms,  dtype=np.float32)
    beat_flux = np.array(beat_flux, dtype=np.float32)

    if beat_rms.max() > 0:  beat_rms  /= beat_rms.max()
    if beat_flux.max() > 0: beat_flux /= beat_flux.max()

    n_beats = len(beats)
    duration_sec = len(samples) / sr
    scored = []

    # Only consider every 4th beat (bar boundaries)
    bar_indices = range(0, n_beats - 4, 4)

    for bi in bar_indices:
        b_samp = beats[bi]
        t_sec  = b_samp / sr

        rms  = beat_rms[bi]
        flux = beat_flux[bi]

        # Position score: for exit, prefer 55–85% of track
        # for entry, prefer 0–35% of track
        t_norm = t_sec / duration_sec
        if role == 'exit':
            # Sweet spot 55–85%
            if 0.55 <= t_norm <= 0.85:
                pos_score = 1.0
            elif t_norm < 0.55:
                pos_score = t_norm / 0.55
            else:
                pos_score = max(0, 1.0 - (t_norm - 0.85) / 0.15)
        else:  # entry
            # Sweet spot 0–30%
            if t_norm <= 0.30:
                pos_score = 1.0 - t_norm * 2
            else:
                pos_score = max(0, 1.0 - t_norm)

        # Low energy = good cut (quieter moments are cleaner to cut from/into)
        energy_score = 1.0 - rms

        # Low flux = stable moment (not a transient burst)
        stability_score = 1.0 - flux

        # Phrase alignment bonus: multiples of 16 beats are stronger
        phrase_bonus = 0.2 if bi % 16 == 0 else (0.1 if bi % 8 == 0 else 0.0)

        total = (pos_score * 0.35 +
                 energy_score * 0.30 +
                 stability_score * 0.25 +
                 phrase_bonus)

        scored.append((bi, total, b_samp, round(t_sec, 2)))

    # Sort best first
    scored.sort(key=lambda x: -x[1])
    return scored


def find_best_cut_point(samples, sr, beats, role='exit', top_n=5):
    """
    Find the single best cut point and return details.
    """
    candidates = score_cut_points(samples, sr, beats, role=role)
    if not candidates:
        mid = len(samples) // 2
        return {'sample': mid, 'time_sec': mid/sr, 'beat_idx': 0, 'score': 0.5}

    best = candidates[0]
    return {
        'sample': best[2],
        'time_sec': best[3],
        'beat_idx': best[0],
        'score': round(best[1], 3),
        'top_candidates': [(c[3], round(c[1], 3)) for c in candidates[:top_n]]
    }


def find_best_entry_point(samples, sr, beats):
    """Find best entry point for Track B — where it sounds freshest."""
    return find_best_cut_point(samples, sr, beats, role='entry')


def find_best_loop_point(samples, sr, beats, loop_bars=4):
    """
    Find the best self-similar loop point — a section that sounds good
    repeated. Used for loop-extend transitions.
    Scores by low energy variance (consistent sections loop better).
    """
    if len(beats) < loop_bars * 4:
        q = len(samples) // 4
        return q, q + loop_bars * int(sr * 0.5)

    beat_period = int(np.mean(np.diff(beats))) if len(beats) > 1 else int(sr * 60/120)
    loop_len = beat_period * loop_bars * 4

    best_score = -1
    best_start = beats[0]
    best_end   = beats[min(loop_bars*4, len(beats)-1)]

    # Scan from 60–85% of track for a stable loop point
    n_beats = len(beats)
    start_beat = int(n_beats * 0.60)
    end_beat   = int(n_beats * 0.85)

    for bi in range(start_beat, min(end_beat, n_beats - loop_bars*4), 4):
        b_start = beats[bi]
        b_end   = b_start + loop_len
        if b_end >= len(samples):
            break

        chunk = samples[b_start:b_end]
        # Low variance = consistent energy = good loop
        variance_score = 1.0 - np.std(chunk) / (np.abs(chunk).mean() + 1e-8)
        # Phrase alignment bonus
        phrase_bonus = 0.2 if bi % 16 == 0 else 0.0
        score = variance_score + phrase_bonus

        if score > best_score:
            best_score = score
            best_start = b_start
            best_end   = min(b_end, len(samples))

    return best_start, best_end


# ══════════════════════════════════════════════════════════════════════════════
# 3. SITUATION ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def assess_situation(t1_analysis, t2_analysis):
    from audio_engine import camelot_distance
    bpm1, bpm2 = t1_analysis['bpm'], t2_analysis['bpm']
    bpm_diff = abs(bpm1 - bpm2)
    key_dist = camelot_distance(t1_analysis['camelot'], t2_analysis['camelot'])
    sp1, sp2 = t1_analysis['spectral'], t2_analysis['spectral']
    brightness_delta = sp2['centroid'] - sp1['centroid']
    bass_delta = sp2['bass'] - sp1['bass']
    sections2 = t2_analysis['sections']
    has_clean_intro = sections2['intro_end_sec'] > 4.0

    return {
        'bpm_diff': round(bpm_diff, 1),
        'bpm_ratio': round(bpm2/bpm1, 4),
        'bpm_class': (
            'identical' if bpm_diff < 1 else 'tight' if bpm_diff <= 4 else
            'workable'  if bpm_diff <= 12 else 'stretch' if bpm_diff <= 25 else 'extreme'
        ),
        'key_dist': key_dist,
        'key_class': (
            'perfect'    if key_dist == 0 else 'harmonic' if key_dist == 1 else
            'compatible' if key_dist == 2 else 'clash'
        ),
        'brightness_delta': round(brightness_delta, 1),
        'bass_delta': round(bass_delta, 4),
        'has_clean_intro_b': has_clean_intro,
        'outro_start_sec_a': t1_analysis['sections']['outro_start_sec'],
        'intro_end_sec_b': sections2['intro_end_sec'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRATEGY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    'surgical_blend': {
        'name': 'Surgical Blend',
        'description': (
            'AI scans both tracks bar-by-bar to find the optimal exit point '
            'in Track A (lowest energy, most stable moment) and the cleanest '
            'entry in Track B. Hard cut on the beat grid, with a micro '
            'crossfade (≤200ms) to mask the seam. No long fade — just precision.'
        ),
        'icon': '🎯', 'color': '#00ff88',
    },
    'filter_sweep': {
        'name': 'Filter Sweep',
        'description': (
            'Track B enters through a rising low-pass filter sweep from sub-bass '
            'upward over 16 beats. Cut point in T1 chosen at a spectral trough '
            '— the quietest, least busy bar.'
        ),
        'icon': '🌊', 'color': '#00f0ff',
    },
    'echo_throw': {
        'name': 'Echo Throw',
        'description': (
            'Track A\'s exit bar is smeared with a tempo-synced ping-pong echo '
            'that decays into Track B\'s first bar. BPM mismatch is hidden in the blur.'
        ),
        'icon': '🔁', 'color': '#b347ff',
    },
    'energy_drop': {
        'name': 'Energy Drop',
        'description': (
            'AI finds the deepest breakdown in T1, fades it to silence there, '
            'then drops T2 cold on its strongest downbeat. Silence is the bridge.'
        ),
        'icon': '⬇️', 'color': '#ff8800',
    },
    'pitch_bridge': {
        'name': 'Pitch Bridge',
        'description': (
            'Track B is pitch-shifted toward T1\'s key at the cut point, '
            'then gradually corrected back to native key over 8 bars. '
            'The audience hears a smooth harmonic shift, not a clash.'
        ),
        'icon': '🎵', 'color': '#ffcc00',
    },
    'bass_swap': {
        'name': 'Bass Swap',
        'description': (
            'Cut at the cleanest bar in T1. During the overlap, T1\'s bass '
            'is high-passed out while T2\'s bass is low-passed in. '
            'Highs blend freely. No low-end mud.'
        ),
        'icon': '🔊', 'color': '#ff2d78',
    },
    'loop_extend': {
        'name': 'Loop Extension',
        'description': (
            'AI finds the most self-similar 4-bar loop in T1\'s outro and '
            'repeats it 2× to extend the runway. T2 enters at its natural '
            'intro start after the loop.'
        ),
        'icon': '🔄', 'color': '#88ff00',
    },
}


def select_strategy(situation):
    reasoning, modifications = [], []
    bpm_class  = situation['bpm_class']
    key_class  = situation['key_class']
    has_intro  = situation['has_clean_intro_b']
    bdelta     = situation['brightness_delta']
    bass_delta = situation['bass_delta']
    key_dist   = situation['key_dist']

    reasoning.append(f"BPM: {bpm_class} (Δ{situation['bpm_diff']} BPM)")
    reasoning.append(f"Harmonic: {key_class} (Camelot distance {key_dist})")
    reasoning.append(f"Track B clean intro: {'yes' if has_intro else 'no'}")
    reasoning.append(f"Spectral brightness shift: {bdelta:+.0f} Hz")

    if bpm_class in ('identical','tight') and key_class in ('perfect','harmonic'):
        reasoning.append("Optimal conditions → Surgical Blend with AI-selected cut point")
        strategy = 'surgical_blend'
        if bass_delta > 0.05:
            modifications.append('eq_reduce_bass_b')
            reasoning.append("Track B is bassier — will EQ its bass entry down")

    elif bpm_class in ('identical','tight','workable') and key_class == 'clash':
        reasoning.append("Key clash but BPM is workable → Bass Swap prevents low-end collision")
        strategy = 'bass_swap'
        modifications.append('isolate_bass_b')
        if not has_intro:
            modifications.append('attenuate_intro_b')
            reasoning.append("Track B has no clean intro — will shape its entry volume")

    elif bpm_class == 'workable' and key_class in ('perfect','harmonic','compatible'):
        if abs(bdelta) > 500:
            reasoning.append(f"Large spectral shift ({bdelta:+.0f} Hz) → Filter Sweep builds anticipation")
            strategy = 'filter_sweep'
            modifications.append('lpf_sweep_b')
        else:
            reasoning.append("Workable BPM + compatible key → Surgical Blend with time-stretch")
            strategy = 'surgical_blend'

    elif bpm_class == 'stretch' and key_class in ('perfect','harmonic'):
        reasoning.append("Large BPM gap but harmonic keys → Echo Throw masks the tempo seam")
        strategy = 'echo_throw'
        modifications.append('echo_tail_a')

    elif bpm_class == 'extreme':
        reasoning.append("Extreme BPM gap — silence bridge is the honest move")
        strategy = 'energy_drop'
        modifications.append('natural_fadeout_a')

    elif key_class == 'clash' and bpm_class in ('stretch','extreme'):
        reasoning.append("Both BPM and key clash → Pitch Bridge + time-stretch surgery")
        strategy = 'pitch_bridge'
        modifications.append('pitch_shift_b')
        modifications.append('gradual_pitch_correct_b')
        reasoning.append(f"Track B pitch-shifted {key_dist} semitone(s) toward T1's key")

    elif not has_intro and key_class != 'clash':
        reasoning.append("Track B has no clean intro → Loop Extension buys extra runway")
        strategy = 'loop_extend'
        modifications.append('loop_outro_a')
        modifications.append('build_intro_b')

    else:
        reasoning.append("General case → Filter Sweep for clean separation")
        strategy = 'filter_sweep'

    return strategy, reasoning, modifications


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUDIO SURGERY — DSP tools
# ══════════════════════════════════════════════════════════════════════════════

def apply_lowpass(samples, sr, cutoff_hz, order=4):
    nyq = sr / 2.0
    cutoff = min(cutoff_hz / nyq, 0.99)
    if cutoff <= 0.001: return np.zeros_like(samples)
    b, a = signal.butter(order, cutoff, btype='low')
    return signal.filtfilt(b, a, samples).astype(np.float32)


def apply_highpass(samples, sr, cutoff_hz, order=4):
    nyq = sr / 2.0
    cutoff = min(cutoff_hz / nyq, 0.99)
    if cutoff <= 0.001: return samples.copy()
    b, a = signal.butter(order, cutoff, btype='high')
    return signal.filtfilt(b, a, samples).astype(np.float32)


def pitch_shift_semitones(samples, sr, semitones):
    if abs(semitones) < 0.05: return samples.copy()
    from audio_engine import time_stretch
    ratio = 2 ** (semitones / 12.0)
    stretched = time_stretch(samples, sr, 1.0 / ratio)
    return np.interp(
        np.linspace(0, len(stretched)-1, len(samples)),
        np.arange(len(stretched)), stretched
    ).astype(np.float32)


def filter_sweep_intro(samples, sr, sweep_sec=16.0, start_hz=60, end_hz=18000):
    sweep_samples = min(int(sweep_sec * sr), len(samples))
    out = samples.copy()
    n_steps = 48
    step_size = max(1, sweep_samples // n_steps)
    for i in range(n_steps):
        start = i * step_size
        end = min(start + step_size, sweep_samples, len(out))
        if start >= len(out): break
        t = i / n_steps
        cutoff = start_hz * (end_hz / start_hz) ** t
        chunk = out[start:end]
        if len(chunk) < 20: continue
        out[start:end] = apply_lowpass(chunk, sr, cutoff) * (0.25 + 0.75 * t**0.5)
    return out


def echo_tail(samples, sr, echo_sec=5.0, n_echoes=5, decay=0.52):
    echo_samp = int(echo_sec * sr)
    tail_start = max(0, len(samples) - echo_samp)
    tail = samples[tail_start:].copy()
    delay_samp = int(375 * sr / 1000)  # dotted-eighth
    out = tail.copy()
    for i in range(1, n_echoes+1):
        offset = i * delay_samp
        if offset < len(out):
            out[offset:] += tail[:len(out)-offset] * (decay**i)
    out = out * np.linspace(1.0, 0.0, len(out)) ** 0.6
    result = samples.copy()
    result[tail_start:] = np.clip(out[:len(result)-tail_start], -1, 1)
    return result


def gradual_pitch_correct(samples, sr, start_semitones, n_beats, bpm):
    beat_samp = int(sr * 60.0 / bpm)
    total = min(n_beats * beat_samp, len(samples))
    n_steps = min(n_beats, 16)
    step_samp = max(1, total // n_steps)
    out = samples.copy()
    for i in range(n_steps):
        s = i * step_samp
        e = min(s + step_samp, total)
        semitones = start_semitones * (1.0 - i/n_steps)
        if abs(semitones) < 0.08: continue
        shifted = pitch_shift_semitones(samples[s:e].copy(), sr, semitones)
        out[s:e] = shifted[:e-s]
    return out


def bass_swap_crossfade(s1_fade, s2_fade, sr):
    n = len(s1_fade)
    n_steps = 32
    step = max(1, n // n_steps)
    out = np.zeros(n, dtype=np.float32)
    BASS_HZ = 200
    for i in range(n_steps):
        s = i * step
        e = min(s + step, n)
        t = i / n_steps
        c1, c2 = s1_fade[s:e], s2_fade[s:e]
        if len(c1) < 8: continue
        b1 = apply_lowpass(c1, sr, BASS_HZ)
        b2 = apply_lowpass(c2, sr, BASS_HZ)
        bass = b1 * np.cos(t*np.pi/2) + b2 * np.sin(t*np.pi/2)
        hi   = (c1-b1)*(1-t) + (c2-b2)*t
        out[s:e] = (bass + hi)[:e-s]
    return out


def micro_crossfade(a, b, sr, ms=80):
    """Tiny crossfade (default 80ms) to remove click at a hard cut."""
    n = min(int(sr * ms / 1000), len(a), len(b))
    if n < 2:
        return np.concatenate([a, b]).astype(np.float32)
    fade_out = np.cos(np.linspace(0, np.pi/2, n))
    fade_in  = np.sin(np.linspace(0, np.pi/2, n))
    seam = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], seam, b[n:]]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRANSITION EXECUTION — Content-aware, beat-precise
# ══════════════════════════════════════════════════════════════════════════════

def execute_transition(s1, s2, sr, strategy, situation, t1_analysis, t2_analysis,
                       progress_cb=None):
    """
    Execute the chosen strategy using AI-selected cut/entry points.
    Returns (mixed_audio, execution_log).
    """
    log_lines = []
    def log(msg):
        log_lines.append(msg)
        if progress_cb: progress_cb(msg)
        print(f"  [DJ] {msg}")

    bpm        = t1_analysis['bpm']
    beat_samp  = int(sr * 60.0 / bpm)
    phrase_samp = beat_samp * 16
    beats1     = t1_analysis.get('beats', [])
    beats2     = t2_analysis.get('beats', [])

    # ── AI content scan: find best CUT in T1 and ENTRY in T2 ─────────────
    log("Scanning Track A bar-by-bar for optimal exit point...")
    cut_info = find_best_cut_point(s1, sr, beats1, role='exit')
    exit_samp = cut_info['sample']
    log(f"Best exit: {cut_info['time_sec']:.1f}s (beat {cut_info['beat_idx']}, "
        f"quality score {cut_info['score']:.2f})")
    if cut_info.get('top_candidates'):
        tops = ', '.join(f"{t:.1f}s" for t, _ in cut_info['top_candidates'][:3])
        log(f"Runner-up candidates: {tops}")

    log("Scanning Track B bar-by-bar for cleanest entry point...")
    entry_info = find_best_entry_point(s2, sr, beats2)
    entry_samp = entry_info['sample']
    log(f"Best entry: {entry_info['time_sec']:.1f}s (beat {entry_info['beat_idx']}, "
        f"quality score {entry_info['score']:.2f})")

    # Ensure we have enough material on both sides
    min_keep_t1 = int(sr * 10)  # keep at least 10s of T1
    if exit_samp < min_keep_t1:
        exit_samp = min_keep_t1
        log(f"Exit point moved forward to {exit_samp/sr:.1f}s (minimum track length)")

    remaining_t1 = len(s1) - exit_samp
    if remaining_t1 < beat_samp:
        exit_samp = max(0, len(s1) - beat_samp * 4)
        log(f"Exit point adjusted: not enough T1 tail, using {exit_samp/sr:.1f}s")

    # ── Execute chosen strategy ────────────────────────────────────────────

    if strategy == 'surgical_blend':
        log("Technique: Surgical Blend — precision cut + micro-crossfade")
        log(f"Cutting Track A at {exit_samp/sr:.1f}s → Track B entry at {entry_samp/sr:.1f}s")

        # Hard cut with a short micro-crossfade (80ms) to kill the click
        t1_cut = s1[:exit_samp].copy()
        t2_cut = s2[entry_samp:].copy()

        # Tail of T1 into head of T2: micro-crossfade
        result = micro_crossfade(t1_cut, t2_cut, sr, ms=80)
        log(f"Hard cut applied — micro-crossfade of 80ms masks the seam")
        log(f"Total mix: {len(result)/sr:.1f}s | T1 used: {exit_samp/sr:.1f}s | T2 from: {entry_samp/sr:.1f}s")

    elif strategy == 'bass_swap':
        log("Technique: Bass Swap — independent low/high frequency crossfade")
        log(f"Cut zone: T1 exit {exit_samp/sr:.1f}s → T2 entry {entry_samp/sr:.1f}s")

        # Overlap window = 2 phrases
        overlap = min(phrase_samp * 2, len(s1) - exit_samp, len(s2) - entry_samp, int(sr * 16))
        overlap = max(overlap, int(sr * 4))

        t1_f = s1[exit_samp: exit_samp + overlap].copy()
        t2_f = s2[entry_samp: entry_samp + overlap].copy()
        fade_len = min(len(t1_f), len(t2_f))

        blend = bass_swap_crossfade(t1_f[:fade_len], t2_f[:fade_len], sr)
        log(f"Bass swap over {fade_len/sr:.1f}s — low-end collision prevented")
        result = np.concatenate([s1[:exit_samp], blend, s2[entry_samp + fade_len:]])

    elif strategy == 'filter_sweep':
        log("Technique: Filter Sweep — Track B unveiled frequency-by-frequency")
        log(f"Cut zone: T1 exit {exit_samp/sr:.1f}s, T2 entry {entry_samp/sr:.1f}s")

        sweep_sec = min(phrase_samp * 2 / sr, 16.0)
        log(f"LPF sweep: 60 Hz → 18 kHz over {sweep_sec:.0f}s")

        t2_from_entry = s2[entry_samp:].copy()
        t2_swept = filter_sweep_intro(t2_from_entry, sr, sweep_sec=sweep_sec)

        overlap = min(int(sweep_sec * sr), len(s1) - exit_samp, len(t2_swept))
        t1_f = s1[exit_samp: exit_samp + overlap].copy()
        t2_f = t2_swept[:overlap].copy()
        fade_len = min(len(t1_f), len(t2_f))

        fade_out = np.linspace(1, 0, fade_len) ** 1.3
        fade_in  = np.linspace(0, 1, fade_len) ** 0.7
        blend = t1_f[:fade_len] * fade_out + t2_f[:fade_len] * fade_in
        result = np.concatenate([s1[:exit_samp], blend, t2_swept[fade_len:]])
        log("Filter sweep complete")

    elif strategy == 'echo_throw':
        log("Technique: Echo Throw — echo smears the tempo seam")
        log(f"Echo applied to T1 tail, T2 drops at {entry_samp/sr:.1f}s")

        echo_sec = min(6.0, (len(s1) - exit_samp) / sr)
        s1_echoed = echo_tail(s1, sr, echo_sec=echo_sec)

        overlap = min(phrase_samp * 2, len(s1_echoed) - exit_samp, len(s2) - entry_samp)
        overlap = max(overlap, beat_samp * 4)

        t1_f = s1_echoed[exit_samp: exit_samp + overlap].copy()
        t2_f = s2[entry_samp: entry_samp + overlap].copy()
        fade_len = min(len(t1_f), len(t2_f))

        hold = int(fade_len * 0.55)
        fade_out = np.concatenate([np.ones(hold), np.linspace(1,0,fade_len-hold)**2])
        fade_in  = np.concatenate([np.zeros(hold), np.linspace(0,1,fade_len-hold)**0.4])
        blend = t1_f[:fade_len] * fade_out + t2_f[:fade_len] * fade_in
        result = np.concatenate([s1_echoed[:exit_samp], blend, s2[entry_samp+fade_len:]])
        log("Echo throw complete — T2 dropped on the 1")

    elif strategy == 'energy_drop':
        log("Technique: Energy Drop — silence bridge")

        # Find the deepest energy trough in the second half of T1
        _, rms = compute_rms_curve(s1, sr, window_sec=0.5)
        half = len(rms) // 2
        trough_idx = half + int(np.argmin(rms[half:]))
        trough_samp = trough_idx * int(sr * 0.5)
        trough_samp = max(trough_samp, int(sr * 10))

        fade_sec = 3.0
        fade_samp = int(fade_sec * sr)
        fade_region = s1[trough_samp: trough_samp + fade_samp].copy()
        fade_region *= np.linspace(1.0, 0.0, len(fade_region)) ** 1.4

        silence = np.zeros(beat_samp * 2, dtype=np.float32)
        log(f"Deepest trough at {trough_samp/sr:.1f}s → {fade_sec:.0f}s fade → 2-beat silence")
        log(f"Track B enters cold at its {entry_samp/sr:.1f}s mark")
        result = np.concatenate([s1[:trough_samp], fade_region, silence, s2[entry_samp:]])

    elif strategy == 'pitch_bridge':
        log("Technique: Pitch Bridge — harmonic alignment")

        key_dist = situation['key_dist']
        semitone_shift = max(-6, min(6, key_dist * (-1 if situation['bass_delta'] < 0 else 1)))
        log(f"Pitch-shifting T2 by {semitone_shift:+d} semitones")
        t2_from_entry = s2[entry_samp:].copy()
        t2_shifted = pitch_shift_semitones(t2_from_entry, sr, semitone_shift)

        log("Gradual pitch correction over 32 beats")
        t2_corrected = gradual_pitch_correct(t2_shifted, sr, semitone_shift, 32, bpm)

        overlap = min(phrase_samp * 2, len(s1) - exit_samp, len(t2_corrected))
        t1_f = s1[exit_samp: exit_samp + overlap].copy()
        t2_f = t2_corrected[:overlap].copy()
        fade_len = min(len(t1_f), len(t2_f))

        theta = np.linspace(0, np.pi/2, fade_len)
        blend = t1_f[:fade_len]*np.cos(theta) + t2_f[:fade_len]*np.sin(theta)
        result = np.concatenate([s1[:exit_samp], blend, t2_corrected[fade_len:]])
        log("Pitch bridge complete")

    elif strategy == 'loop_extend':
        log("Technique: Loop Extension — finding best self-similar loop in T1")

        loop_start, loop_end = find_best_loop_point(s1, sr, beats1, loop_bars=4)
        loop_chunk = s1[loop_start:loop_end].copy()
        log(f"Loop found: {loop_start/sr:.1f}s–{loop_end/sr:.1f}s ({(loop_end-loop_start)/sr:.1f}s)")

        # Micro-crossfade the loop seamlessly
        xf = min(int(sr * 0.08), len(loop_chunk)//4)
        lc = loop_chunk.copy()
        lc[:xf] *= np.linspace(0, 1, xf)
        lc[-xf:] *= np.linspace(1, 0, xf)
        extension = np.tile(lc, 2)
        log(f"Looping ×2 — adds {len(extension)/sr:.1f}s before T2 entry")

        # T2 enters at its natural entry point
        t2_from_entry = s2[entry_samp:].copy()

        # Build: T1 up to loop_start + loop extension + microfade into T2
        pre = s1[:loop_start]
        result = micro_crossfade(
            np.concatenate([pre, extension]),
            t2_from_entry,
            sr, ms=120
        )
        log(f"Loop extension complete — T2 enters at {entry_samp/sr:.1f}s")

    else:
        log(f"Unknown strategy '{strategy}' — cosine fallback")
        t1_cut = s1[:exit_samp].copy()
        t2_cut = s2[entry_samp:].copy()
        result = micro_crossfade(t1_cut, t2_cut, sr, ms=80)

    return result.astype(np.float32), log_lines


# ══════════════════════════════════════════════════════════════════════════════
# 7. FULL INTELLIGENT MIX PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def intelligent_mix(track1_path, track2_path, output_path, progress_cb=None):
    from audio_engine import (load_audio, save_audio, detect_bpm, detect_key,
                               compute_compatibility, time_stretch)

    def log(msg):
        print(f"[MiXL-Brain] {msg}")
        if progress_cb: progress_cb(msg)

    log("Loading Track A...")
    s1_raw, sr1 = load_audio(track1_path)
    log("Loading Track B...")
    s2_raw, sr2 = load_audio(track2_path)

    TARGET_SR = 22050
    def ds(s, sr):
        if sr > TARGET_SR:
            f = sr // TARGET_SR
            return s[::f].copy(), sr // f
        return s.copy(), sr

    s1_ds, sr1_ds = ds(s1_raw, sr1)
    s2_ds, sr2_ds = ds(s2_raw, sr2)

    log("Detecting BPM (Track A)...")
    bpm1 = detect_bpm(s1_ds, sr1_ds)
    log(f"Track A: {bpm1} BPM")

    log("Detecting key (Track A)...")
    key1, mode1, conf1, cam1 = detect_key(s1_ds, sr1_ds)
    log(f"Track A: {key1} {mode1} — Camelot {cam1}")

    log("Detecting BPM (Track B)...")
    bpm2 = detect_bpm(s2_ds, sr2_ds)
    log(f"Track B: {bpm2} BPM")

    log("Detecting key (Track B)...")
    key2, mode2, conf2, cam2 = detect_key(s2_ds, sr2_ds)
    log(f"Track B: {key2} {mode2} — Camelot {cam2}")

    log("Building beat grids...")
    beats1_ds = compute_beat_grid(s1_ds, sr1_ds, bpm1)
    beats2_ds = compute_beat_grid(s2_ds, sr2_ds, bpm2)
    log(f"Track A: {len(beats1_ds)} beats | Track B: {len(beats2_ds)} beats")

    log("Mapping phrase structure...")
    sections1 = analyze_track_sections(s1_ds, sr1_ds, beats1_ds)
    sections2 = analyze_track_sections(s2_ds, sr2_ds, beats2_ds)
    log(f"Track A: intro {sections1['intro_end_sec']}s | outro from {sections1['outro_start_sec']}s")
    log(f"Track B: intro {sections2['intro_end_sec']}s | outro from {sections2['outro_start_sec']}s")

    log("Profiling spectral character...")
    spec1 = spectral_profile(s1_ds, sr1_ds)
    spec2 = spectral_profile(s2_ds, sr2_ds)

    t1_info = {
        'bpm': bpm1, 'key': key1, 'mode': mode1, 'camelot': cam1, 'confidence': conf1,
        'beats': beats1_ds, 'sections': sections1, 'spectral': spec1,
        'duration_sec': round(len(s1_ds)/sr1_ds, 2), 'filename': Path(track1_path).name,
    }
    t2_info = {
        'bpm': bpm2, 'key': key2, 'mode': mode2, 'camelot': cam2, 'confidence': conf2,
        'beats': beats2_ds, 'sections': sections2, 'spectral': spec2,
        'duration_sec': round(len(s2_ds)/sr2_ds, 2), 'filename': Path(track2_path).name,
    }

    log("Assessing mixing situation...")
    situation = assess_situation(t1_info, t2_info)
    log(f"BPM: {situation['bpm_class']} | Key: {situation['key_class']}")

    log("Selecting transition strategy...")
    strategy, reasoning, modifications = select_strategy(situation)
    log(f"Strategy: {STRATEGIES[strategy]['icon']} {STRATEGIES[strategy]['name']}")
    for r in reasoning:
        log(f"  → {r}")

    # Prepare full-rate audio
    log("Preparing full-resolution audio...")
    s1 = s1_raw.copy()
    s2 = s2_raw.copy()
    for arr in [s1, s2]:
        mx = np.abs(arr).max()
        if mx > 0:
            arr /= mx; arr *= 0.95

    if sr2 != sr1:
        log(f"Resampling Track B: {sr2} → {sr1} Hz")
        new_len = int(len(s2) * sr1 / sr2)
        s2 = np.interp(np.linspace(0, len(s2)-1, new_len), np.arange(len(s2)), s2).astype(np.float32)

    # BPM sync
    bpm_ratio = bpm2 / bpm1
    if abs(bpm_ratio - 1.0) > 0.005:
        log(f"Time-stretching Track B ×{bpm_ratio:.4f} ({bpm2:.1f} → {bpm1:.1f} BPM)...")
        s2 = time_stretch(s2, sr1, bpm_ratio)
        log(f"Track B after stretch: {len(s2)/sr1:.1f}s")

    # Scale beat positions to full sample rate
    scale = sr1 / sr1_ds
    beats1_full = [int(b * scale) for b in beats1_ds]
    beats2_full = [int(b * scale) for b in beats2_ds]

    sections1_full = dict(sections1)
    sections1_full['outro_start'] = int(sections1['outro_start_sec'] * sr1)
    sections1_full['intro_end']   = int(sections1['intro_end_sec']   * sr1)

    sections2_full = dict(sections2)
    sections2_full['outro_start'] = int(sections2['outro_start_sec'] * sr1)
    sections2_full['intro_end']   = int(sections2['intro_end_sec']   * sr1)

    t1_full = {**t1_info, 'beats': beats1_full, 'sections': sections1_full}
    t2_full = {**t2_info, 'beats': beats2_full, 'sections': sections2_full, 'bpm': bpm1}

    log(f"Executing: {STRATEGIES[strategy]['name']}...")
    mixed, exec_log = execute_transition(
        s1, s2, sr1, strategy, situation, t1_full, t2_full,
        progress_cb=progress_cb
    )

    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak * 0.92
    save_audio(output_path, mixed, sr1)
    log(f"Mix saved: {output_path} ({len(mixed)/sr1:.1f}s)")

    compat = compute_compatibility(bpm1, key1, mode1, cam1, bpm2, key2, mode2, cam2)

    return {
        'track1': {k: v for k, v in t1_info.items() if k != 'beats'},
        'track2': {k: v for k, v in t2_info.items() if k != 'beats'},
        'compatibility': compat,
        'situation': situation,
        'strategy': strategy,
        'strategy_info': STRATEGIES[strategy],
        'reasoning': reasoning,
        'modifications': modifications,
        'execution_log': exec_log,
        'mix_duration_sec': round(len(mixed)/sr1, 2),
        'output_path': output_path,
        'bpm_ratio_applied': round(bpm_ratio, 4),
    }