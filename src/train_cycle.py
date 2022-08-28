from dual_network import dual_network
from auto_play import multiProcessSelfPlay
from train_network import train_network
from evaluate_network import multi_process_evaluate_network, update_best_player
from evaluate_best_player import multi_process_evaluate_best_player
import tensorflow as tf
import platform

if __name__ == '__main__':
    dual_network()

    for i in range(3):
        print('Train',i,'====================')
        
        multiProcessSelfPlay(4)

        train_network()

        #update_best_player = multi_process_evaluate_network(5)
        update_best_player()

        #if update_best_player:
        multi_process_evaluate_best_player(4)