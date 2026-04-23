"""
Listen to / look at preprocessed pairs to confirm alignment.

Three modes:
  stereo   Play drumless in your LEFT ear and original in your RIGHT ear.
           Use HEADPHONES. If aligned, they sit in the same "time" and feel
           like one song split by ear; if misaligned you'll hear clear lag.
  diff     Play (original - drumless). If alignment is good, this sounds
           ≈ like drums alone. Misalignment = ghost notes and phasey smearing.
           This is the most sensitive ear-check.
  plot     Show waveforms of drumless, original, and (original - drumless).

Usage:
    python verify.py                         # lists available songs
    python verify.py --song 0                # play pair #0, stereo mode, 15s
    python verify.py --song my_song --mode diff --start 30 --duration 20
    python verify.py --song 2 --mode plot

Requires: numpy, soundfile, sounddevice (for playback), matplotlib (for plot)
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import soundfile as sf


def load_mono(path):
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def _window(d, o, sr, start, duration):
    s = max(0, int(start * sr))
    end = min(len(d), len(o))
    if duration is not None:
        end = min(end, s + int(duration * sr))
    d = d[s:end]
    o = o[s:end]
    n = min(len(d), len(o))
    return d[:n], o[:n], n


def play_stereo(d_path, o_path, start, duration):
    import sounddevice as sd
    d, sr = load_mono(d_path)
    o, sr_o = load_mono(o_path)
    assert sr == sr_o, "Sample rates differ"

    d, o, n = _window(d, o, sr, start, duration)
    stereo = np.column_stack([d, o])
    peak = float(np.max(np.abs(stereo)))
    if peak > 0:
        stereo = stereo * (0.9 / peak)

    print(f"▶ Stereo split (L = drumless, R = original)")
    print(f"  {start:.2f}s → {start + n/sr:.2f}s   ({n/sr:.2f}s total)")
    print(f"  Use HEADPHONES. Ctrl+C to stop.")
    sd.play(stereo, sr)
    sd.wait()


def play_diff(d_path, o_path, start, duration):
    import sounddevice as sd
    d, sr = load_mono(d_path)
    o, sr_o = load_mono(o_path)
    assert sr == sr_o, "Sample rates differ"

    d, o, n = _window(d, o, sr, start, duration)
    diff = o - d
    peak = float(np.max(np.abs(diff)))
    if peak > 0:
        diff = diff * (0.9 / peak)

    print(f"▶ Difference (original - drumless) — should sound like drums-only")
    print(f"  {start:.2f}s → {start + n/sr:.2f}s   ({n/sr:.2f}s total)")
    print(f"  Bad alignment = smeared / flangey / ghost notes. Ctrl+C to stop.")
    sd.play(diff, sr)
    sd.wait()


def show_plot(d_path, o_path, start, duration):
    import matplotlib.pyplot as plt
    d, sr = load_mono(d_path)
    o, _ = load_mono(o_path)
    d, o, n = _window(d, o, sr, start, duration)
    t = start + np.arange(n) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, d, color="steelblue", linewidth=0.5)
    axes[0].set_ylabel("Drumless")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, o, color="darkorange", linewidth=0.5)
    axes[1].set_ylabel("Original")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, o - d, color="crimson", linewidth=0.5)
    axes[2].set_ylabel("Original − Drumless\n(should be mostly drums)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{Path(d_path).stem} — alignment check")
    plt.tight_layout()
    plt.show()


def list_pairs(root):
    root = Path(root)
    d_dir = root / "Drumless"
    o_dir = root / "Original"
    pairs = []
    for f in sorted(d_dir.glob("*.wav")):
        o = o_dir / f.name
        if o.exists():
            pairs.append((f, o))
    return pairs


def resolve_song(pairs, arg):
    if arg is None:
        return None
    if arg.isdigit():
        i = int(arg)
        return pairs[i] if 0 <= i < len(pairs) else None
    for d, o in pairs:
        if d.stem == arg:
            return (d, o)
    return None


def main():
    p = argparse.ArgumentParser(description="Verify drumless/original alignment")
    p.add_argument("--dir", default="Data/Preprocessing",
                   help="Preprocessing output directory")
    p.add_argument("--song", default=None,
                   help="Song stem (without extension) or index. Omit to list.")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds (default 0)")
    p.add_argument("--duration", type=float, default=15.0,
                   help="Duration in seconds (default 15)")
    p.add_argument("--mode", choices=["stereo", "diff", "plot"], default="stereo",
                   help="stereo: L=drumless R=original.  "
                        "diff: play (O-D), should sound like drums.  "
                        "plot: show waveforms.")
    args = p.parse_args()

    pairs = list_pairs(args.dir)
    if not pairs:
        print(f"No processed pairs in {args.dir}. Run preprocess.py first.")
        sys.exit(1)

    if args.song is None:
        print("Available songs:")
        for i, (d, _) in enumerate(pairs):
            print(f"  [{i:2d}] {d.stem}")
        print("\nRun again with --song INDEX or --song NAME")
        return

    chosen = resolve_song(pairs, args.song)
    if chosen is None:
        print(f"Song '{args.song}' not found. Run without --song to see list.")
        sys.exit(1)

    d_path, o_path = chosen
    print(f"Song: {d_path.stem}")

    if args.mode == "stereo":
        play_stereo(d_path, o_path, args.start, args.duration)
    elif args.mode == "diff":
        play_diff(d_path, o_path, args.start, args.duration)
    else:
        show_plot(d_path, o_path, args.start, args.duration)


if __name__ == "__main__":
    main()