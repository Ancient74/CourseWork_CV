import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate, correlation_lags, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalize_for_corr(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    x /= (np.linalg.norm(x) + 1e-9)
    return x


def _corr_1d(a, b):
    return correlate(_normalize_for_corr(a), _normalize_for_corr(b),
                     mode="full", method="fft")


def _corr_multichannel(A, B):
    out = None
    for c in range(A.shape[0]):
        r = _corr_1d(A[c], B[c])
        out = r if out is None else out + r
    return out


def _pos_norm(x):
    x = np.maximum(x, 0.0)
    m = float(x.max())
    return x / (m + 1e-9) if m > 0 else x


def gcc_phat(a, b, max_shift_samples):
    n_a, n_b = len(a), len(b)
    n_fft = 1 << (n_a + n_b - 1).bit_length()

    A = np.fft.rfft(a.astype(np.float64), n=n_fft)
    B = np.fft.rfft(b.astype(np.float64), n=n_fft)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n_fft)

    full_cc = np.concatenate([cc[n_fft - (n_b - 1):], cc[:n_a]])
    lags = np.arange(-(n_b - 1), n_a)

    mask = np.abs(lags) <= max_shift_samples
    lags_w = lags[mask]
    cc_w = full_cc[mask]
    best = int(np.argmax(cc_w))
    return int(lags_w[best]), cc_w, lags_w


def multi_window_fine_tune(d, o, sr, max_shift_ms=50,
                           n_windows=6, window_seconds=8):
    n = min(len(d), len(o))
    win = int(window_seconds * sr)
    max_shift = int(max_shift_ms * sr / 1000)

    if n < win * 2:
        lag, _, _ = gcc_phat(d[:n], o[:n], max_shift)
        return lag, [lag], 0.0

    margin = int(n * 0.1)
    usable = n - 2 * margin
    if usable < n_windows * win:
        n_windows = max(2, usable // win)
    stride = (usable - win) // max(1, n_windows - 1) if n_windows > 1 else 0

    lags = []
    for i in range(n_windows):
        s = margin + i * stride
        e = s + win
        if e > n:
            break
        lag, _, _ = gcc_phat(d[s:e], o[s:e], max_shift)
        lags.append(lag)

    lags_arr = np.array(lags, dtype=np.float64)
    med = float(np.median(lags_arr))
    mad = float(np.median(np.abs(lags_arr - med)))

    threshold = max(2.0, 3.0 * mad)
    keep = np.abs(lags_arr - med) <= threshold
    if np.sum(keep) >= 2:
        robust_lag = int(round(float(np.median(lags_arr[keep]))))
    else:
        robust_lag = int(round(med))

    mad_ms = mad * 1000.0 / sr
    return robust_lag, lags, mad_ms


def verify_multi_segment(drumless_mono, original_mono, sr, shift_samples,
                         n_segments=4, tolerance_ms=50, search_ms=150):
    if shift_samples > 0:
        d = drumless_mono[shift_samples:]
        o = original_mono
    elif shift_samples < 0:
        d = drumless_mono
        o = original_mono[-shift_samples:]
    else:
        d, o = drumless_mono, original_mono

    n = min(len(d), len(o))
    seg_len = n // n_segments
    if seg_len < sr * 2:
        return 0.0, []

    tolerance = int(tolerance_ms * sr / 1000)
    search = int(search_ms * sr / 1000)

    residuals_ms = []
    aligned = 0
    for i in range(n_segments):
        d_seg = d[i * seg_len: (i + 1) * seg_len]
        o_seg = o[i * seg_len: (i + 1) * seg_len]
        corr = _corr_1d(d_seg, o_seg)
        lags = correlation_lags(len(d_seg), len(o_seg), mode="full")
        mask = np.abs(lags) <= search
        best = int(lags[mask][np.argmax(corr[mask])])
        residuals_ms.append(best * 1000.0 / sr)
        if abs(best) <= tolerance:
            aligned += 1

    return aligned / n_segments, residuals_ms


def find_alignment_shift(drumless_mono, original_mono, sr,
                         max_shift_seconds=60, hop_length=512,
                         n_candidates=5, verify=True,
                         fine_window_ms=50, fine_n_windows=6,
                         fine_window_seconds=8,
                         use_hpss=True):

    if use_hpss:
        print("  Applying HPSS to isolate harmonic elements...")
        d_align, _ = librosa.effects.hpss(drumless_mono)
        o_align, _ = librosa.effects.hpss(original_mono)
    else:
        d_align, o_align = drumless_mono, original_mono

    d_onset = librosa.onset.onset_strength(y=d_align, sr=sr, hop_length=hop_length)
    o_onset = librosa.onset.onset_strength(y=o_align, sr=sr, hop_length=hop_length)
    d_chroma = librosa.feature.chroma_cqt(y=d_align, sr=sr, hop_length=hop_length)
    o_chroma = librosa.feature.chroma_cqt(y=o_align, sr=sr, hop_length=hop_length)

    c_onset = _corr_1d(d_onset, o_onset)
    c_chroma = _corr_multichannel(d_chroma, o_chroma)
    lags = correlation_lags(len(d_onset), len(o_onset), mode="full")

    env_rate = sr / hop_length
    max_lag_frames = int(max_shift_seconds * env_rate)
    mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    c_onset = c_onset[mask]
    c_chroma = c_chroma[mask]
    lags = lags[mask]

    combined = np.sqrt(_pos_norm(c_onset) * _pos_norm(c_chroma))

    min_sep_frames = max(10, int(0.5 * env_rate))
    peak_idx, _ = find_peaks(combined, distance=min_sep_frames)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(combined))])
    top = peak_idx[np.argsort(combined[peak_idx])[::-1][:n_candidates]]

    candidates = []
    for idx in top:
        frames = int(lags[idx])
        shift_s = frames * hop_length
        peak_val = float(combined[idx])
        if verify:
            agreement, residuals_ms = verify_multi_segment(
                d_align, o_align, sr, shift_s
            )
        else:
            agreement, residuals_ms = 1.0, []
        score = peak_val * (0.3 + 0.7 * agreement)
        candidates.append({
            "shift_samples": shift_s,
            "shift_seconds": shift_s / sr,
            "peak_value": peak_val,
            "agreement": agreement,
            "residuals_ms": residuals_ms,
            "score": score,
            "frame_index": int(idx),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    coarse_shift = best["shift_samples"]

    if coarse_shift > 0:
        d_a, o_a = d_align[coarse_shift:], o_align
    elif coarse_shift < 0:
        d_a, o_a = d_align, o_align[-coarse_shift:]
    else:
        d_a, o_a = d_align, o_align

    fine_lag, per_window_lags, mad_ms = multi_window_fine_tune(
        d_a, o_a, sr,
        max_shift_ms=fine_window_ms,
        n_windows=fine_n_windows,
        window_seconds=fine_window_seconds,
    )

    total_shift = coarse_shift + fine_lag

    return total_shift, {
        "corr_onset": c_onset,
        "corr_chroma": c_chroma,
        "corr_combined": combined,
        "lags_frames": lags,
        "env_rate": env_rate,
        "hop_length": hop_length,
        "candidates": candidates,
        "best_candidate": best,
        "coarse_shift": coarse_shift,
        "fine_lag": fine_lag,
        "fine_per_window_lags": per_window_lags,
        "fine_mad_ms": mad_ms,
        "peak_indices": [int(p) for p in top],
    }


def rms_match(to_adjust, reference_mono, target_peak=0.98):
    """Scale `to_adjust` so its RMS matches `reference_mono`'s. Peak-limit to avoid clipping."""
    adj_mono = librosa.to_mono(to_adjust) if to_adjust.ndim > 1 else to_adjust
    adj_rms = float(np.sqrt(np.mean(adj_mono ** 2))) + 1e-9
    ref_rms = float(np.sqrt(np.mean(reference_mono ** 2))) + 1e-9
    gain = ref_rms / adj_rms
    out = to_adjust * gain
    peak = float(np.max(np.abs(out)))
    if peak > target_peak:
        out = out * (target_peak / peak)
        gain = gain * (target_peak / peak)
    return out, gain


def save_diagnostic(info, shift, sr, d_final_mono, o_final_mono, out_path):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.2, 1.5])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    lags_sec = info["lags_frames"] * info["hop_length"] / sr
    ax0.plot(lags_sec, _pos_norm(info["corr_onset"]),
             color="tab:blue", linewidth=0.6, alpha=0.5, label="onset (rhythm)")
    ax0.plot(lags_sec, _pos_norm(info["corr_chroma"]),
             color="tab:green", linewidth=0.6, alpha=0.5, label="chroma (harmony)")
    ax0.plot(lags_sec, info["corr_combined"],
             color="black", linewidth=1.0, label="combined (geom. mean)")

    chosen = info["best_candidate"]
    for cand in info["candidates"]:
        lag_s = cand["shift_seconds"]
        color = "red" if cand is chosen else "gray"
        size = 90 if cand is chosen else 30
        ax0.scatter([lag_s], [cand["peak_value"]], s=size,
                    color=color, zorder=5, edgecolors="black", linewidths=0.5)
        ax0.annotate(
            f"agr={cand['agreement']:.2f}\nscore={cand['score']:.2f}",
            xy=(lag_s, cand["peak_value"]),
            xytext=(3, 5), textcoords="offset points",
            fontsize=7, color=color
        )

    ax0.set_title(
        f"Coarse alignment — chosen shift: {chosen['shift_seconds']:+.3f}s  "
        f"fine-tune: {info['fine_lag']*1000/sr:+.2f} ms  "
        f"fine-tune MAD: {info['fine_mad_ms']:.3f} ms  "
        f"agreement: {chosen['agreement']:.2f}"
    )
    ax0.set_xlabel("Lag (s)   positive = trim drumless, negative = trim original")
    ax0.set_ylabel("Correlation")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    pw = np.array(info["fine_per_window_lags"], dtype=np.float64) * 1000.0 / sr
    window_idx = np.arange(len(pw))
    ax1.plot(window_idx, pw, "o-", color="steelblue", markersize=8)
    median_ms = info["fine_lag"] * 1000.0 / sr
    ax1.axhline(median_ms, color="red", linestyle="--",
                label=f"chosen median {median_ms:+.2f} ms")
    ax1.set_xlabel("Fine-tune window #")
    ax1.set_ylabel("GCC-PHAT lag (ms)")
    ax1.set_title(f"Per-window fine-tune lags — tight cluster = reliable alignment")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    preview = min(int(5 * sr), len(d_final_mono), len(o_final_mono))
    t = np.arange(preview) / sr
    ax2.plot(t, o_final_mono[:preview], color="darkorange",
             linewidth=0.5, alpha=0.8, label="Original")
    ax2.plot(t, d_final_mono[:preview], color="steelblue",
             linewidth=0.5, alpha=0.8, label="Drumless")
    ax2.set_title("First 5 s of aligned output — transients should line up")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=80)
    plt.close(fig)


def print_candidates(candidates, chosen):
    print(f"  Candidates (top {len(candidates)}):")
    for i, c in enumerate(candidates, 1):
        mark = "★" if c is chosen else " "
        resid = ""
        if c["residuals_ms"]:
            resid = "  residuals(ms)=[" + ",".join(
                f"{r:+.0f}" for r in c["residuals_ms"]
            ) + "]"
        print(f"    {mark} {i}. shift {c['shift_seconds']:+8.3f}s   "
              f"peak={c['peak_value']:.3f}   "
              f"agree={c['agreement']:.2f}   "
              f"score={c['score']:.3f}{resid}")



def _crop_range(x, start, end):
    return x[:, start:end] if x.ndim > 1 else x[start:end]


def _trim_start(x, n):
    if n <= 0:
        return x
    return x[:, n:] if x.ndim > 1 else x[n:]


def _take_head(x, n):
    return x[:, :n] if x.ndim > 1 else x[:n]


def _to_mono(x):
    return librosa.to_mono(x) if x.ndim > 1 else x


def _to_sf(x):
    """soundfile expects shape (samples, channels)."""
    return x.T if x.ndim > 1 else x


def process_pair(drumless_path, original_path, out_drumless, out_original,
                 crop_ratio=0.2, max_shift_seconds=60,
                 n_candidates=5, verify=True,
                 fine_n_windows=6, fine_window_seconds=8,
                 match_volume=True, use_hpss=True, diag_path=None):
    drumless, sr = librosa.load(str(drumless_path), sr=None, mono=False)
    original, sr_o = librosa.load(str(original_path), sr=None, mono=False)

    if sr != sr_o:
        print(f"  resampling original {sr_o} → {sr} Hz")
        if original.ndim > 1:
            original = np.stack([
                librosa.resample(ch, orig_sr=sr_o, target_sr=sr) for ch in original
            ])
        else:
            original = librosa.resample(original, orig_sr=sr_o, target_sr=sr)

    L_d, L_o = drumless.shape[-1], original.shape[-1]
    drumless_c = _crop_range(drumless, int(L_d * crop_ratio), int(L_d * (1 - crop_ratio)))
    original_c = _crop_range(original, int(L_o * crop_ratio), int(L_o * (1 - crop_ratio)))
    print(f"  cropped — drumless: {drumless_c.shape[-1]/sr:.1f}s, "
          f"original: {original_c.shape[-1]/sr:.1f}s")

    d_mono = _to_mono(drumless_c)
    o_mono = _to_mono(original_c)
    shift, info = find_alignment_shift(
        d_mono, o_mono, sr,
        max_shift_seconds=max_shift_seconds,
        n_candidates=n_candidates,
        verify=verify,
        fine_n_windows=fine_n_windows,
        fine_window_seconds=fine_window_seconds,
        use_hpss=use_hpss,
    )
    print_candidates(info["candidates"], info["best_candidate"])
    best = info["best_candidate"]
    pw_str = "[" + ", ".join(
        f"{l*1000/sr:+.2f}" for l in info["fine_per_window_lags"]
    ) + "] ms"
    print(f"  fine-tune per window: {pw_str}   "
          f"MAD={info['fine_mad_ms']:.3f} ms")
    print(f"  → chosen shift: {shift/sr*1000:+.1f} ms   "
          f"(coarse {info['coarse_shift']/sr*1000:+.1f} ms, "
          f"fine {info['fine_lag']*1000/sr:+.2f} ms)   "
          f"agreement {best['agreement']:.2f}")

    if best["agreement"] < 0.5 and verify:
        print("  ! LOW AGREEMENT — multi-segment verification disagrees; "
              "output may be misaligned.")
    if info["fine_mad_ms"] > 3.0:
        print(f"  ! HIGH FINE-TUNE SPREAD ({info['fine_mad_ms']:.2f} ms MAD) — "
              "fine-tune windows disagree; inspect diagnostic plot.")

    if shift > 0:
        drumless_c = _trim_start(drumless_c, shift)
    elif shift < 0:
        original_c = _trim_start(original_c, -shift)

    n = min(drumless_c.shape[-1], original_c.shape[-1])
    drumless_f = _take_head(drumless_c, n)
    original_f = _take_head(original_c, n)

    if match_volume:
        original_mono_for_rms = _to_mono(original_f)
        drumless_f, gain = rms_match(drumless_f, original_mono_for_rms)
        print(f"  RMS-matched drumless → original (gain ×{gain:.2f})")

    sf.write(str(out_drumless), _to_sf(drumless_f), sr)
    sf.write(str(out_original), _to_sf(original_f), sr)

    if diag_path is not None:
        save_diagnostic(info, shift, sr,
                        _to_mono(drumless_f), _to_mono(original_f), diag_path)

    return {
        "shift_samples": shift,
        "shift_seconds": shift / sr,
        "agreement": best["agreement"],
        "fine_mad_ms": info["fine_mad_ms"],
        "final_length_s": n / sr,
    }


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Preprocess drumless/original pairs")
    parser.add_argument("--drumless-dir", default="Data/TrainingData/Drumless")
    parser.add_argument("--original-dir", default="Data/TrainingData/Original")
    parser.add_argument("--output-dir", default="Data/Preprocessing")
    parser.add_argument("--crop-ratio", type=float, default=0.2,
                        help="Fraction to crop from EACH end (default 0.2)")
    parser.add_argument("--max-shift", type=float, default=60,
                        help="Max coarse alignment shift to search, seconds (default 60)")
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="Top-N combined-correlation peaks to consider (default 5)")
    parser.add_argument("--fine-windows", type=int, default=6,
                        help="Number of GCC-PHAT windows for fine-tune (default 6)")
    parser.add_argument("--fine-window-seconds", type=float, default=8,
                        help="Length of each fine-tune window in seconds (default 8)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip multi-segment verification")
    parser.add_argument("--match-volume", action="store_true",
                        help="RMS-match drumless loudness to original in output")
    parser.add_argument("--use-hpss", action="store_true",
                        help="Apply HPSS before feature extraction and fine-tune "
                             "(slower, recommended for drum-heavy tracks)")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Skip saving diagnostic PNG plots")
    args = parser.parse_args()

    if not 0 <= args.crop_ratio < 0.5:
        parser.error("--crop-ratio must be in [0, 0.5)")

    drumless_dir = Path(args.drumless_dir)
    original_dir = Path(args.original_dir)
    output_dir = Path(args.output_dir)
    out_d_dir = output_dir / "Drumless"
    out_o_dir = output_dir / "Original"
    out_diag_dir = output_dir / "Diagnostics"

    out_d_dir.mkdir(parents=True, exist_ok=True)
    out_o_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_diagnostics:
        out_diag_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(drumless_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {drumless_dir}")
        return

    print(f"Found {len(mp3_files)} drumless files. Processing...\n")

    ok = failed = low_agreement = wide_spread = 0
    for f in mp3_files:
        orig = original_dir / f.name
        if not orig.exists():
            print(f"✗ {f.name}: no matching original, skipping")
            failed += 1
            continue

        print(f"→ {f.name}")
        try:
            out_d = out_d_dir / (f.stem + ".wav")
            out_o = out_o_dir / (f.stem + ".wav")
            diag = None if args.no_diagnostics else out_diag_dir / (f.stem + ".png")
            info = process_pair(
                f, orig, out_d, out_o,
                crop_ratio=args.crop_ratio,
                max_shift_seconds=args.max_shift,
                n_candidates=args.n_candidates,
                verify=not args.no_verify,
                fine_n_windows=args.fine_windows,
                fine_window_seconds=args.fine_window_seconds,
                match_volume=args.match_volume,
                use_hpss=args.use_hpss,
                diag_path=diag,
            )
            print(f"  ✓ saved, final length {info['final_length_s']:.1f}s\n")
            ok += 1
            if info["agreement"] < 0.5:
                low_agreement += 1
            if info["fine_mad_ms"] > 3.0:
                wide_spread += 1
        except Exception as e:
            print(f"  ✗ error: {e}\n")
            failed += 1

    print(f"Done. {ok} OK, {failed} failed/skipped, "
          f"{low_agreement} low-agreement, {wide_spread} wide-fine-tune-spread.")
    print(f"Output: {output_dir.resolve()}")
    if not args.no_diagnostics and ok:
        print(f"→ Inspect {out_diag_dir}. Middle panel = per-window fine-tune lags; "
              f"tight cluster = solid alignment.")
        print(f"→ Use verify.py --mode diff to ear-check alignment quality.")


if __name__ == "__main__":
    main()