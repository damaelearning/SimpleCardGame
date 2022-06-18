from numpy.lib.function_base import hanning
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os
from game import DECK_NUM, FIELDS_NUM, HANDS_NUM
from const import MODEL_DIR

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (12,DECK_NUM,5)
DN_OUTPUT_SIZE = FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1

def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
        kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005))

def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f
        
def dual_network():
    if os.path.exists(MODEL_DIR/'best.h5'):
        return
    
    input = Input(shape=DN_INPUT_SHAPE)
    
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)
        
    x = GlobalAveragePooling2D()(x)
    
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005), activation='softmax', name='pi')(x)
    
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)
    
    model = Model(inputs=input, outputs=[p,v])
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR/'best.h5')
    
    K.clear_session()
    del model

if __name__ == '__main__':
    dual_network()