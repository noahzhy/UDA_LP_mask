import os, glob, random, time, math

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# # convert numpy array to image
# def array_to_image(array, img_path):
#     print(array.shape)
#     image = Image.fromarray(array)
#     # show image
#     plt.imshow(image)
#     plt.show()
#     # save as jpg with quality 95
#     # image.save(img_path, quality=95)


for path in glob.glob('data/train/image/*.png'):
    # image = np.load(path)
    # convert to jpg
    image = Image.open(path).convert('RGB')
    image.save(path.replace('.png', '.jpg'), quality=95)

#     # show via figure
#     plt.figure()
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.imshow(image[:, :, i])
#     plt.show()
#     break

    # print(image.shape)
    # array_to_image(image, path.replace('.npy', '.jpg'))
    # # os.remove(path)
