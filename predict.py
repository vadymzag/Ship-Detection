from model.unet_test import Unet
from tools.tool import test_generator
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt

threshold = 0.6
img_height = 256
img_width = 256
img_size = (img_height, img_width, 3)
test_path = '../python_udemy_projects/ship_detection/inputs/test_v2'
model_weights_name = 'seg_model_weights_best.hdf5'


montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


def main():

    # build model
    unet = Unet(filters=8, input_size=img_size, pretrained_weights=model_weights_name)
    unet.build()

    # create test generator
    test_gen = test_generator(test_path, 4)

    # show result by 2x2 test image and model prediction 2x2
    while True:
        test_imgs = next(test_gen)
        predictions = unet.predict(test_imgs)
        predictions = predictions >= threshold
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        result_imgs = montage_rgb(test_imgs)
        result_masks = montage_rgb(predictions)
        ax1.imshow(result_imgs)
        ax1.set_title(f"Images")
        ax2.imshow(result_masks)
        ax2.set_title("Predictions")
        plt.show()


if __name__ == "__main__":
    main()
