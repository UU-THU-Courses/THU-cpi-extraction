import sys
from keras import backend as K
from keras.losses import categorical_crossentropy

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice(y_true, y_pred):
    #print('y_true, y_pred99999999999', y_true.shape, y_pred.shape)
    #sys.exit(-5)
    alpha = 0.8
    numLabels = y_pred.shape[-1]
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,index], y_pred[:,index])
    return (1-alpha)*dice + alpha * categorical_crossentropy(y_true, y_pred)
