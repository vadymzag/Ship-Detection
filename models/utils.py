import numpy as np
import tensorflow.keras.backend as K
from skimage.io import imread


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# LOSS
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
#     print(loss.shape)
#     print(loss)
    return loss


# METRICS
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


# IMAGE GENERATOR
def make_img_gen(df, batch_size):
    unique_img_list = list(df[df["EncodedPixels"].notna()].groupby("ImageId"))
    img_gen = []
    mask_gen = []
    while True:
        np.random.shuffle(unique_img_list)
        for img, masks in unique_img_list:
            read_img = imread(f'inputs/train_v2/{img}')
            read_mask = masks_as_image(masks['EncodedPixels'].values)
            img_gen += [read_img]
            mask_gen += [read_mask]
            if len(img_gen) >= batch_size:
                yield np.stack(img_gen, 0) / 255, np.stack(mask_gen, 0)



