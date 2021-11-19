from keras.models import Model
from keras import layers
from tensorflow.keras.optimizers import Adam
from model.utils import dice_coef, dice_loss
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class Unet(Model):

    def __init__(self, filters,  input_size, pretrained_weights=False):
        # Build U-Net model
        def upsample_simple(filters, kernel_size, strides, padding):
            return layers.UpSampling2D(strides)

        upsample = upsample_simple

        input_img = layers.Input(input_size, name='RGB_Input')
        pp_in_layer = input_img

        pp_in_layer = layers.AvgPool2D((1, 1))(pp_in_layer)

        pp_in_layer = layers.GaussianNoise(0.1)(pp_in_layer)
        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(c5)

        u6 = upsample(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c6)

        u7 = upsample(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c7)

        u8 = upsample(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c8)

        u9 = upsample(filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        outputs = layers.UpSampling2D((1, 1))(d)
        super(Unet, self).__init__(inputs=[input_img], outputs=outputs)

        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self, show_summary = True):
        self.compile(optimizer=Adam(1e-3, decay=1e-6), loss=dice_loss, metrics=[dice_coef])
        if show_summary:
            self.summary()

    def save_model(self, name):
        self.save_weights(name)

    def callbacks(self):
        weight_path = "{}_weights_best.hdf5".format('seg_model')
        checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)
        early_stoping = EarlyStopping(monitor='loss', mode='min', patience=5, verbose=2)
        reduce_lr= ReduceLROnPlateau(monitor='loss', factor=0.33,
                                           patience=1, verbose=1, mode='min',
                                           min_lr=1e-6)
        return [checkpoint, early_stoping, reduce_lr]
