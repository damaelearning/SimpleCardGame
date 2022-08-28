from dual_network import dual_network
from self_play import multiProcessSelfPlay
from train_network import train_network
from evaluate_network import multi_process_evaluate_network, update_best_player
from evaluate_best_player import multi_process_evaluate_best_player
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
import platform

if __name__ == '__main__':
    if platform.system() == "Darwin":
        mlcompute.set_mlc_device(device_name="gpu")
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")
    dual_network()

    for i in range(3):
        print('Train',i,'====================')
        
        multiProcessSelfPlay(4)

        train_network()

        #update_best_player = multi_process_evaluate_network(5)
        update_best_player()

        #if update_best_player:
        multi_process_evaluate_best_player(4)