import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy import signal
from precompile import nperseg, noverlap, segment_to_np, to_chunks

def stft_stereo(chunk_samples, fs):
    """STFT both channels; return complex array (F, T, 2)."""
    Zxx_channels = []
    for ch in range(chunk_samples.shape[1]):
        _, _, Zxx = signal.stft(
            chunk_samples[:, ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            boundary=None,
        )
        Zxx_channels.append(Zxx)
    return np.stack(Zxx_channels, axis=-1)


def prepare_input(chunk_samples, fs):
    Zxx = stft_stereo(chunk_samples, fs)
    mag = np.log1p(np.abs(Zxx))
    mag_stacked = np.concatenate([mag[:, :, 0], mag[:, :, 1]], axis=0)
    scale = mag_stacked.max() + 1e-8
    mag_norm = mag_stacked / scale
    return mag_norm.astype(np.float32), float(scale), Zxx


def mask_to_audio(pred_mag_norm, scale, Zxx_in, fs):
    H = pred_mag_norm.shape[0]
    F = H // 2

    pred_log_mag = pred_mag_norm * scale
    pred_abs = np.expm1(pred_log_mag)

    abs_L = pred_abs[:F, :]
    abs_R = pred_abs[F:, :]

    phase_L = np.angle(Zxx_in[:, :, 0])
    phase_R = np.angle(Zxx_in[:, :, 1])

    Zxx_out_L = abs_L * np.exp(1j * phase_L)
    Zxx_out_R = abs_R * np.exp(1j * phase_R)

    _, audio_L = signal.istft(Zxx_out_L, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, audio_R = signal.istft(Zxx_out_R, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return np.stack([audio_L, audio_R], axis=1)


def read_song(path, original_file_name):
    seg = AudioSegment.from_file(os.path.join(path, "Original", original_file_name))
    seg = seg.set_frame_rate(48000)
    fs, samples = segment_to_np(seg)
    chunks = to_chunks(samples, fs)
    return fs, chunks


def process_song(model, fs, chunks):
    audio_parts = []

    prepared = [prepare_input(c, fs) for c in chunks]
    mag_batch = np.stack([p[0] for p in prepared], axis=0)[..., None]

    preds = model.predict(mag_batch, batch_size=1, verbose=1)

    for pred, (_, scale, Zxx) in zip(preds, prepared):
        audio = mask_to_audio(pred[..., 0], scale, Zxx, fs)
        audio_parts.append(audio)

    return np.concatenate(audio_parts, axis=0)


def save_audio(audio, fs, out_path):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)

    AudioSegment(
        audio_int16.tobytes(),
        frame_rate=fs,
        sample_width=2,
        channels=2,
    ).export(out_path, format="mp3", bitrate="192k")


def visualize_chunks(model, fs, chunks, chunk_start, chunk_end,
                     channel='L', save_path=None, show=True):
    chunk_start = max(0, chunk_start)
    chunk_end   = min(len(chunks), chunk_end)
    if chunk_end <= chunk_start:
        raise ValueError(f"Empty range: [{chunk_start}, {chunk_end})")

    selected = chunks[chunk_start:chunk_end]
    n = len(selected)

    prepared = [prepare_input(c, fs) for c in selected]
    mag_batch = np.stack([p[0] for p in prepared], axis=0)[..., None]
    preds = model.predict(mag_batch, batch_size=1, verbose=0)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(15, 3.5 * n),
        squeeze=False,
    )

    def pick_channel(stacked):
        H = stacked.shape[0]
        F = H // 2
        if channel == 'L':
            return stacked[:F]
        elif channel == 'R':
            return stacked[F:]
        elif channel == 'mean':
            return 0.5 * (stacked[:F] + stacked[F:])
        else:
            raise ValueError(f"channel must be 'L', 'R', or 'mean', got {channel!r}")

    F = prepared[0][0].shape[0] // 2
    T = prepared[0][0].shape[1]
    t_axis_sec = np.arange(T) * (nperseg - noverlap) / fs
    f_axis_khz = np.arange(F) * fs / nperseg / 1000.0

    for row, (pred, (mag_norm, scale, _)) in enumerate(zip(preds, prepared)):
        chunk_idx = chunk_start + row

        input_mag = pick_channel(mag_norm)
        mask      = pick_channel(pred[..., 0])
        output    = input_mag * mask

        extent = [t_axis_sec[0], t_axis_sec[-1], f_axis_khz[0], f_axis_khz[-1]]

        im0 = axes[row, 0].imshow(
            input_mag, aspect='auto', origin='lower',
            extent=extent, cmap='magma', vmin=0, vmax=1,
        )
        axes[row, 0].set_title(f"Chunk {chunk_idx} — Input (log-mag, normalized)")
        axes[row, 0].set_ylabel("Frequency (kHz)")
        axes[row, 0].set_xlabel("Time (s)")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(
            mask, aspect='auto', origin='lower',
            extent=extent, cmap='viridis', vmin=0, vmax=1,
        )
        axes[row, 1].set_title(
            f"Chunk {chunk_idx} — Mask  "
            f"(mean={mask.mean():.3f}, <0.5: {(mask < 0.5).mean()*100:.1f}%)"
        )
        axes[row, 1].set_ylabel("Frequency (kHz)")
        axes[row, 1].set_xlabel("Time (s)")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(
            output, aspect='auto', origin='lower',
            extent=extent, cmap='magma', vmin=0, vmax=1,
        )
        axes[row, 2].set_title(f"Chunk {chunk_idx} — Output (input × mask)")
        axes[row, 2].set_ylabel("Frequency (kHz)")
        axes[row, 2].set_xlabel("Time (s)")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Chunks {chunk_start}–{chunk_end-1}  (channel: {channel})",
        fontsize=14, y=1.0,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def process_all(model, validation_data_path):
    """Default behavior: run every file in Original/ and write to Drumless/."""
    original_dir = os.path.join(validation_data_path, "Original")
    drumless_dir = os.path.join(validation_data_path, "Drumless")
    os.makedirs(drumless_dir, exist_ok=True)

    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_names = sorted(
        f for f in os.listdir(original_dir)
        if os.path.splitext(f)[1].lower() in audio_exts
    )

    print(f"Found {len(file_names)} file(s) to process.")

    for i, file_name in enumerate(file_names, start=1):
        stem, _ = os.path.splitext(file_name)
        out_path = os.path.join(drumless_dir, f"{stem}.mp3")

        if os.path.exists(out_path):
            print(f"[{i}/{len(file_names)}] skipping (already exists): {file_name}")
            continue

        print(f"[{i}/{len(file_names)}] processing: {file_name}")
        try:
            fs, chunks = read_song(validation_data_path, file_name)
            audio = process_song(model, fs, chunks)
            save_audio(audio, fs, out_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on validation songs, or visualize a chunk range."
    )
    parser.add_argument("--model", default="best_model.keras",
                        help="Path to the saved Keras model (default: best_model.keras)")
    parser.add_argument("--data-dir", default="Data/ValidationData",
                        help="Root directory with Original/ (and Drumless/ output) "
                             "(default: Data/ValidationData)")
    parser.add_argument("--visualize", metavar="FILENAME", default=None,
                        help="Visualize chunks of a specific file instead of "
                             "running full inference. Pass the filename inside "
                             "Original/ (e.g. 'song.wav').")
    parser.add_argument("--chunk-start", type=int, default=0,
                        help="First chunk index to visualize (inclusive, default 0)")
    parser.add_argument("--chunk-end", type=int, default=None,
                        help="Last chunk index to visualize (exclusive). "
                             "Defaults to chunk-start + 5.")
    parser.add_argument("--channel", choices=["L", "R", "mean"], default="L",
                        help="Which stereo half to display (default L)")
    parser.add_argument("--save-fig", default=None,
                        help="If given, save the visualization to this path "
                             "instead of (or in addition to) showing it")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display the figure interactively "
                             "(useful with --save-fig on headless machines)")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, safe_mode=False)

    if args.visualize is not None:
        chunk_end = (args.chunk_end if args.chunk_end is not None
                     else args.chunk_start + 5)
        print(f"Loading: {args.visualize}")
        fs, chunks = read_song(args.data_dir, args.visualize)
        print(f"Total chunks: {len(chunks)}. "
              f"Visualizing [{args.chunk_start}, {chunk_end}).")
        visualize_chunks(
            model, fs, chunks,
            chunk_start=args.chunk_start,
            chunk_end=chunk_end,
            channel=args.channel,
            save_path=args.save_fig,
            show=not args.no_show,
        )
    else:
        process_all(model, args.data_dir)


if __name__ == "__main__":
    main()
    