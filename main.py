from tools.tool import *
from model.unet_test import Unet

train_path = '../python_udemy_projects/ship_detection/inputs/train_v2'
test_path = '../python_udemy_projects/ship_detection/inputs/test_v2'
segment_path = '../python_udemy_projects/ship_detection/inputs/train_ship_segmentations_v2.csv'

model_weights_name = 'unet_weight_model.hdf5'

# FITTING PARAMETERS
TRAIN_BATCH_SIZE = 10
TRAIN_STEPS = 25
TRAIN_EPOCHS = 30
SAVE_FINALE_MODEL = True


def main():
    # get train_df -> DataFrame that has trainable labels
    train_df = spliting(segment_path, train_path)

    # create generator(X, Y) for fit model
    train_gen = create_aug_gen(make_image_gen(train_df, train_path, TRAIN_BATCH_SIZE))

    # get model and build it
    unet = Unet(filters=8, input_size=(256, 256, 3))
    unet.build()

    # callbacks => [checkpoint, early_stopping, reduceLR]
    model_callbacks = unet.callbacks()

    # fit model using generator
    unet.fit_generator(
        train_gen,
        steps_per_epoch=TRAIN_STEPS,
        epochs=TRAIN_EPOCHS,
        callbacks=model_callbacks,
        workers=1
    )

    # saving model weights
    if SAVE_FINALE_MODEL:
        unet.save_model(model_weights_name)


if __name__ == "__main__":
    main()
