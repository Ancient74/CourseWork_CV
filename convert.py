import tensorflow as tf
import numpy as np
import os
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


validation_data_path = "Data/ValidationData"
original_dir = os.path.join(validation_data_path, "Original")
drumless_dir = os.path.join(validation_data_path, "Drumless")
os.makedirs(drumless_dir, exist_ok=True)

model = tf.keras.models.load_model('best_model.keras', safe_mode=False)

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
