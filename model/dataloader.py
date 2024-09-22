import os, sys, random, math, time, glob

import cv2
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def png2npy(data_path):
    arr = Image.open(data_path).convert('L')
    arr = (np.array(arr) + 8) // 16

    # show
    plt.imshow(arr)
    plt.show()


def draw_gaussian(heatmap, center, sigma=10):
    radius = sigma * 3

    x = np.arange(0, heatmap.shape[1], 1, float)
    y = np.arange(0, heatmap.shape[0], 1, float)
    y = y[:, np.newaxis]
    x0, y0 = center

    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    heatmap += g

    return heatmap


def generate_gaussian_heatmap(points, canvas_size, sigma):
    width, height = canvas_size
    heatmap = np.zeros((height, width))
    
    for point in points:
        heatmap = draw_gaussian(heatmap, point, sigma)

    return heatmap


class SynthDataLoader(tf.keras.utils.Sequence):
    def __init__(self, root_dir, input_shape=128, batch_size=32, shuffle=True):
        self.data = glob.glob(os.path.join(root_dir, "raw", "*.png"))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.input_shape = input_shape

        if self.shuffle: random.shuffle(self.data)

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_raw = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        gen_batch = np.zeros((self.batch_size, self.input_shape, self.input_shape, 1))
        raw_batch = np.zeros((self.batch_size, self.input_shape, self.input_shape, 32))
        hmp_batch = np.zeros((self.batch_size, self.input_shape, self.input_shape, 1))

        # using the tf io to read the png file and convert to tensor
        for idx, data in enumerate(batch_raw):  # Modify this line
            gen_path = data.replace("raw", "gen")
            pts_path = data.replace("raw", "pts").replace(".png", ".txt")

            raw_im = tf.io.read_file(data)
            raw_im = tf.io.decode_png(raw_im, channels=1)
            raw_im = tf.image.convert_image_dtype(raw_im, tf.uint8)

            # gen_hmap via pts
            pts = load_data(pts_path)
            hmp_im = generate_gaussian_heatmap(pts, (self.input_shape, self.input_shape), sigma=1)

            gen_im = tf.io.read_file(gen_path)
            gen_im = tf.io.decode_png(gen_im, channels=1)
            gen_im = tf.image.convert_image_dtype(gen_im, tf.uint8)

            # one-hot 256x256x1 to 256x256x32
            raw_im = tf.one_hot(tf.squeeze(raw_im), 32)

            gen_batch[idx, :, :] = gen_im
            raw_batch[idx, :, :, :raw_im.shape[1]] = raw_im
            hmp_batch[idx, :, :] = np.expand_dims(hmp_im, axis=-1)

        return gen_batch, (raw_batch, hmp_batch)

    def on_epoch_end(self):
        if self.shuffle: random.shuffle(self.data)


if __name__ == "__main__":
    # data_dir = "data/test"
    # batch_size = 8
    # dataloader = SynthDataLoader(data_dir, 128, batch_size, shuffle=True)

    img_path = "data/train/*.png"
    im = random.choice(glob.glob(img_path))
    png2npy(im)

    print(len(dataloader))

    for batch in dataloader:
        gen_batch, (raw_batch, hmp_batch) = batch
        # check unique raw
        # print(np.unique(raw_batch[0, :, :, 0]))
        print(raw_batch.shape, gen_batch.shape, hmp_batch.shape)

        # draw pts on the generated image
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(raw_batch[0, :, :, 0], cmap='gray')
        plt.title('Raw Image')
        plt.subplot(1, 3, 2)
        plt.imshow(gen_batch[0, :, :, :], cmap='gray')
        plt.title('True Image')
        plt.subplot(1, 3, 3)
        plt.imshow(hmp_batch[0, :, :, 0], cmap='hot')
        plt.show()

        break

    # points = [(50, 50), (80, 80), (30, 30)]
    # points = load_data('data/test/pts/0.txt').tolist()

    # canvas_size = (256, 256)
    # sigma = 2

    # heatmap = generate_gaussian_heatmap(points, canvas_size, sigma)

    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Gaussian Heatmap')
    # plt.show()
