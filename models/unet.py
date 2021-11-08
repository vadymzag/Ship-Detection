from tensorflow.keras import layers, models

class Unet():

    def __init__(self, input_shape):

        inputs = layers.Input(shape=input_shape)

        # s = layers.Lambda(lambda x: K.cast(x / 255., 'float32'))(inputs)
        c1 = layers.Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = models.Model(inputs=inputs, outputs=outputs)

    def get_model(self):
        return self.model


