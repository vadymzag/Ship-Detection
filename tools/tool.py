import os
import pandas as pd
from skimage.io import imread
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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
    return img.reshape(shape).T         # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    '''
    in_mask_list => list with rle mask
    Returns mask as image with all masked ships
    '''
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def spliting(seg_path, train_image_dir):
    '''
    seg_path => path for segmentation csv
    Save csv processed file if doesn't exist
    if exists read it from file
    Returns train_df => balanced_df with all categories
    '''
    df = pd.read_csv(seg_path)
    if os.path.isfile('unique_img_ids.csv'):
        unique_img_ids = pd.read_csv('unique_img_ids.csv')
    else:
        df['ships'] = df["EncodedPixels"].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        unique_img_ids = df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        # some files are too small/corrupt
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                                       os.stat(os.path.join(train_image_dir,
                                                                                            c_img_id)).st_size / 1024)
        unique_img_ids = unique_img_ids[
            unique_img_ids['file_size_kb'] > 50]  # image files less than 50kb will be corrupt
        unique_img_ids['file_size_kb'].hist()
        df.drop(['ships'], axis=1, inplace=True)
        unique_img_ids.to_csv('unique_img_ids.csv')

    balanced_df = unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(3000) if len(x) > 3000 else x)

    train_df = pd.merge(df, balanced_df)

    return train_df


def make_image_gen(in_df,
                   train_image_dir,
                   batch_size):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            # try to open image, if cannot, miss it
            try:
                c_img = imread(rgb_path)
            except:
                # Uncoment line bellow, to show which image cannot open
                # print(f"\n!!! Cannot find file: {rgb_path}")
                continue

            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            c_img = c_img[::3, ::3]
            c_mask = c_mask[::3, ::3]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []


def create_aug_gen(in_gen, seed=None):
    dg_args = dict(featurewise_center=False,
                   samplewise_center=False,
                   rotation_range=45,
                   width_shift_range=0.1,
                   height_shift_range=0.1,
                   shear_range=0.01,
                   zoom_range=[0.9, 1.25],
                   horizontal_flip=True,
                   vertical_flip=True,
                   fill_mode='reflect',
                   data_format='channels_last')

    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)


def test_generator(test_path, size):
    test_img = []
    list_img = os.listdir(test_path)
    while True:
        np.random.shuffle(list_img)
        for img in list_img:
            img_path = os.path.join(test_path, img)
            rgb_img = imread(img_path)[::3, ::3] / 255.
            test_img.append(rgb_img)
            if len(test_img) >= size:
                yield np.stack(test_img, 0)
                test_img = []
