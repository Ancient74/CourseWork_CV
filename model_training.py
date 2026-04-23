import argparse

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Lambda, Multiply, Concatenate, GroupNormalization, Dropout, LeakyReLU, GaussianNoise, Input, MaxPooling2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import mixed_precision


class PrecomputedDenoiseGenerator(tf.keras.utils.Sequence):
    def __init__(self, sample_indices, data_dir="precomputed_data",
                 batch_size=4, shuffle=True, augment=None):
        super().__init__()
        self.in_dir  = os.path.join(data_dir, "inputs")
        self.out_dir = os.path.join(data_dir, "outputs")
        self.indices = np.array(sample_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = shuffle if augment is None else augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    def _split_stereo(img):
        H = img.shape[0]
        F = H // 2
        return img[:F], img[F:]

    @staticmethod
    def _join_stereo(L, R):
        return np.concatenate([L, R], axis=0)

    def _augment_pair(self, x, y):
        x = x.copy()
        y = y.copy()

        if np.random.rand() < 0.5:
            xL, xR = self._split_stereo(x)
            yL, yR = self._split_stereo(y)
            x = self._join_stereo(xR, xL)
            y = self._join_stereo(yR, yL)

        if np.random.rand() < 0.7:
            shift = np.random.uniform(-0.15, 0.15)
            x = np.clip(x + shift, 0.0, 1.0)
            y = np.clip(y + shift, 0.0, 1.0)

        if np.random.rand() < 0.3:
            T = x.shape[1]
            w = np.random.randint(1, max(2, T // 20))
            t0 = np.random.randint(0, T - w)
            x[:, t0:t0 + w, :] = 0.0

        if np.random.rand() < 0.3:
            xL, xR = self._split_stereo(x)
            F = xL.shape[0]
            w = np.random.randint(1, max(2, F // 20))
            f0 = np.random.randint(0, F - w)
            xL[f0:f0 + w, :, :] = 0.0
            xR[f0:f0 + w, :, :] = 0.0
            x = self._join_stereo(xL, xR)

        if np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()

        return x, y

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x, batch_y = [], []
        for i in batch_indices:
            x = np.load(os.path.join(self.in_dir,  f"sample_{i}.npz"))["image"][..., 0:1]
            y = np.load(os.path.join(self.out_dir, f"sample_{i}.npz"))["image"][..., 0:1]

            if self.augment:
                x, y = self._augment_pair(x, y)

            batch_x.append(x)
            batch_y.append(y)
        return np.stack(batch_x, 0), np.stack(batch_y, 0)


def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['loss'], 'b-', label='Train Loss (MAE)')
    plt.plot(epochs, h['val_loss'], 'r-', label='Validation Loss (MAE)')
    plt.title('Loss Convergence (Overfitting check)')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    mae_key = 'mae' if 'mae' in h else 'mean_absolute_error'
    val_mae_key = 'val_mae' if 'val_mae' in h else 'val_mean_absolute_error'
    
    plt.plot(epochs, h[mae_key], 'b-', label='Train MAE')
    plt.plot(epochs, h[val_mae_key], 'r-', label='Validation MAE')
    plt.title('Metric Performance (MAE)')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()

def get_model(load_path=None):
    if load_path is not None:
        model = tf.keras.models.load_model(load_path, safe_mode=False)
    else:
        l2_val = 0.00005 
        inputs = Input(shape=(1026, 468, 1))
        image_channels = Lambda(lambda x: x[..., 0:2], name="slice_img")(inputs)
        x = GaussianNoise(0.01)(image_channels)

        weights = [64, 128, 256, 512]
        blocks = []
        for weight in weights:
            for layers in range(3):
                x = Conv2D(weight, 3, padding="same", kernel_regularizer=L2(l2_val))(x)
                x = GroupNormalization(groups=8) (x)
                x = LeakyReLU(negative_slope=0.1)(x)
            blocks.append(x)
            x = MaxPooling2D((2, 2))(x)

        x = Conv2D(1024, 3, padding="same", kernel_regularizer=L2(l2_val))(x)
        x = GroupNormalization(groups=8) (x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(1024, 3, padding="same", kernel_regularizer=L2(l2_val))(x)
        x = GroupNormalization(groups=8) (x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.2)(x)

        for i, weight in enumerate(reversed(weights)):
            x = Conv2DTranspose(weight, 3, strides=(2, 2), padding="same")(x)
            x = GroupNormalization(groups=8) (x)
            x = LeakyReLU(negative_slope=0.1)(x)
            
            skip_connection = blocks[-(i + 1)]
            
            height_diff = skip_connection.shape[1] - x.shape[1]
            width_diff = skip_connection.shape[2] - x.shape[2]
            
            if height_diff > 0 or width_diff > 0:
                x = ZeroPadding2D(padding=((0, height_diff), (0, width_diff)))(x)
            
            x = Concatenate(axis=-1)([x, skip_connection])
            for layers in range(3):
                x = Conv2D(weight, 3, padding="same", kernel_regularizer=L2(l2_val))(x)
                x = GroupNormalization(groups=8) (x)
                x = LeakyReLU(negative_slope=0.1)(x)

        mask = Conv2D(1, 3, padding="same", activation="sigmoid", dtype='float32')(x)
        outputs = Multiply()([inputs, mask]) 

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss='mae', metrics=['mae'])

    return model

def main(load_path=None):
    parser = argparse.ArgumentParser(description="Train model on precomputed spectrogram images")
    parser.add_argument("--load-path", default=None)
    args = parser.parse_args()

    mixed_precision.set_global_policy('mixed_float16')
    
    all_indices = np.arange(821)
    np.random.shuffle(all_indices)

    split_idx = int(len(all_indices) * 0.2)
    val_idx = all_indices[:split_idx]
    train_idx = all_indices[split_idx:]

    train_gen = PrecomputedDenoiseGenerator(train_idx, data_dir="precomputed_data", batch_size=4)
    val_gen = PrecomputedDenoiseGenerator(val_idx, data_dir="precomputed_data", batch_size=4, shuffle=False)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                        min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10,
                    restore_best_weights=True, verbose=1),
        ModelCheckpoint("best_model_v2.keras",
                        monitor="val_loss", save_best_only=True,
                        verbose=1),
        TensorBoard(log_dir="./logs", histogram_freq=0),
    ]
    model = get_model(load_path=args.load_path)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        callbacks=callbacks
    )

    plot_history(history)
 
if __name__ == "__main__":
    main(load_path=None)
