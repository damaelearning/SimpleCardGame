from dual_network import dual_network
from self_play import multiProcessSelfPlay
from train_network import train_network
from evaluate_network import multi_process_evaluate_network
from evaluate_best_player import multi_process_evaluate_best_player

if __name__ == '__main__':
    dual_network()

    for i in range(3):
        print('Train',i,'====================')
        
        multiProcessSelfPlay(5)

        train_network()

        update_best_player = multi_process_evaluate_network(5)

        if update_best_player:
            multi_process_evaluate_best_player(5)