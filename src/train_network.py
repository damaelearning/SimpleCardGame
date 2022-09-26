import tensorflow as tf
from game import INITIAL_LIFE
from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
from const import HISTORY_DIR, MODEL_DIR
import platform
from shutil import copy


def load_data(history_path):
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def update_best_player(best_path, latest_path):
    copy(latest_path, best_path)
    print('Change BestPlayer')

def train_network(old_model_path, new_model_path, batch_size, epochs = 100, shape=DN_INPUT_SHAPE, 
        history_path = sorted(Path(HISTORY_DIR).glob('*.history'))[-1]):
    if platform.system() == "Darwin":
        from tensorflow.python.compiler.mlcompute import mlcompute
        mlcompute.set_mlc_device(device_name="gpu")
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")
    history = load_data(history_path)
    input, policies, values = zip(*history)

    height, width, channel = shape
    input = np.array(input)
    input = input.transpose(0, 2, 3, 1)
    input = input.reshape(len(input), height, width, channel)   
    policies = np.array(policies)
    values = np.array(values)

    model = load_model(old_model_path)

    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='SGD')

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
                print('\rTrain {}/{}'.format(epoch + 1,epochs), end=''))
    
    model.fit(input, [policies, values], batch_size=batch_size, epochs=epochs,
            verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    model.save(new_model_path)

    K.clear_session()
    del model

if __name__ == '__main__':
    train_network()