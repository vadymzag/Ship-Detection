import pandas as pd
from models.utils import jaccard_distance_loss, dice_coef, make_img_gen
from models.unet import Unet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# print(os.listdir("inputs"))
train_image_dir = 'inputs/train_v2'
test_image_dir = 'inputs/test_v2'
segment = 'inputs/train_ship_segmentations_v2.csv'

VALID_SIZE = 20
BATCH_SIZE = 2
STEP_COUNT = 30
EPOCHS = 5
IMG_SIZE = (768, 768, 3)

# get df from train_ship_segmentation
masks = pd.read_csv(segment)

# create train generator
gener = make_img_gen(masks, batch_size=BATCH_SIZE)
train_x, train_y = next(gener)

# # Show shape of x and y values
# print('x', train_x.shape, train_x.min(), train_x.max())
# print('y', train_y.shape, train_y.min(), train_y.max())

# # Show image(all images with ship)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
# ax1.imshow(train_x[0], cmap='gray')
# ax1.set_title('images')
# ax2.imshow(train_y[0, :, :, 0], cmap='gray_r')
# ax2.set_title('ships')
# plt.show()

# Create unet model
unet = Unet(IMG_SIZE)
model = unet.get_model()
model.summary()
# compile Unet
model.compile(optimizer='adam', loss=jaccard_distance_loss, metrics=[dice_coef])

valid_gener = gener(VALID_SIZE)
valid_x, valid_y = next(gener)

# create all helpful callbacks(earlysopping, reducelearningrate, checkpointer for model,
# checkpoint for model best loss, early stopper
early_stopper = EarlyStopping(patience=8, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=6, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
checkpointer_train = ModelCheckpoint('model_best_train.h5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(gener,
                    steps_per_epoch=STEP_COUNT,
                    epochs=EPOCHS,
                    callbacks=[early_stopper, checkpointer, checkpointer_train, reduce_learning_rate],
                    validation_data=(valid_x, valid_y))
