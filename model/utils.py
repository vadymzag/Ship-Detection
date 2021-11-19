import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    # True Positive ==> intersection
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])

    # Sum of true and predictive
    true_sum = K.sum(K.square(y_true), axis=[1, 2, 3])
    pred_sum = K.sum(K.square(y_pred), axis=[1, 2, 3])

    # dice = 2|a*b|/|a|^2+|b|^2
    return K.mean((2. * intersection + smooth) / (true_sum + pred_sum + smooth), axis=0)


def dice_loss(y_true, y_pred):
    # return 1 - dice (higher dice, better model, task to maximize dice or minimize 1-dice)
    return 1 - dice_coef(y_true, y_pred)

