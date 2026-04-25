import os
import numpy as np
from scipy import signal
from pydub import AudioSegment
from tqdm import tqdm

training_data_path = "Data/Preprocessing"

nperseg = 1024
noverlap = 512

def segment_to_np(segment):
    samples = np.array(segment.get_array_of_samples())

    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    return segment.frame_rate, samples

def to_chunks(signal, fs, chunk_sec=5):
    chunk_size = fs * chunk_sec
    res = []

    for i in range(chunk_size, len(signal) - chunk_size + 1, chunk_size):
        chunk = signal[i:i + chunk_size]
        res.append(chunk)

    return res

def read_data(path):
    file_names = os.listdir(os.path.join(training_data_path, "Drumless"))

    input_segments = [
        AudioSegment.from_wav(os.path.join(path, "Original", f))
        for f in file_names
    ]

    output_segments = [
        AudioSegment.from_mp3(os.path.join(path, "Drumless", f))
        for f in file_names
    ]

    inp = []
    out = []

    for in_seg, out_seg in zip(input_segments, output_segments):
        in_seg = in_seg.set_frame_rate(48000)
        out_seg = out_seg.set_frame_rate(48000)
        
        fs_in, x = segment_to_np(in_seg)
        fs_out, y = segment_to_np(out_seg)

        assert fs_in == fs_out
        fs = fs_in

        inp.extend([(fs, c) for c in to_chunks(x, fs)])
        out.extend([(fs, c) for c in to_chunks(y, fs)])

    return inp, out


def stft(data):
    fs, samples = data

    Zxx_channels = []
    for ch in range(samples.shape[1]):
        f, t, Zxx = signal.stft(
            samples[:, ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            boundary=None
        )
        Zxx_channels.append(Zxx)

    Zxx = np.stack(Zxx_channels, axis=-1)
    return f, t, Zxx

def create_image_pair(input_data, output_data):
    fs_in, _ = input_data
    _, _, Zxx_in  = stft(input_data)
    _, _, Zxx_out = stft(output_data)

    def stack_channels(Zxx):
        return np.concatenate([Zxx[:, :, 0], Zxx[:, :, 1]], axis=0)

    Z_in  = stack_channels(Zxx_in)
    Z_out = stack_channels(Zxx_out)

    mag_in  = np.log1p(np.abs(Z_in))
    mag_out = np.log1p(np.abs(Z_out))

    scale = mag_in.max() + 1e-8
    mag_in_n  = mag_in  / scale
    mag_out_n = np.clip(mag_out / scale, 0.0, 1.0)

    H, W = mag_in.shape

    in_img  = np.zeros((H, W, 2), dtype=np.float32)
    out_img = np.zeros((H, W, 1), dtype=np.float32) 

    in_img[..., 0] = mag_in_n
    in_img[..., 1] = np.angle(Z_in) / np.pi
    out_img[..., 0] = mag_out_n

    return in_img, out_img, float(scale), int(fs_in)


def precalculate_dataset(input_chunks, output_chunks, save_dir="precomputed_data"):
    in_path = os.path.join(save_dir, "inputs")
    out_path = os.path.join(save_dir, "outputs")
    
    os.makedirs(in_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    print(f"Precalculating {len(input_chunks)} samples...")
    
    for i in tqdm(range(len(input_chunks))):
        rgb_in, rgb_out, scale, fs = create_image_pair(input_chunks[i], output_chunks[i])
        np.savez(os.path.join(in_path, f"sample_{i}.npz"), image=rgb_in, scale=scale, fs=fs)
        np.savez(os.path.join(out_path, f"sample_{i}.npz"), image=rgb_out)

    print(f"Done! Data saved to {save_dir}")

if __name__ == "__main__":
    input_data, output_data = read_data(training_data_path)
    precalculate_dataset(input_data, output_data)
