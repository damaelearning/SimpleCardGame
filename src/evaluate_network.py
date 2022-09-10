import tensorflow as tf
from const import MODEL_DIR
from game import State, Actor
from pv_mcts import pv_ismcts, next_action_by
from model_wrapper import ModelWrapper
from train_network import update_best_player
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np
import multiprocessing
from const import MODEL_DIR
import platform


EN_GAME_COUNT = 100
EN_TEMPERATURE = 0.0

def first_player_point(ended_state):
    if ended_state.turn_owner.is_lose():
        return 0 if ended_state.turn_owner.is_first_player else 1
    if ended_state.enemy.is_lose():
        return 1 if ended_state.turn_owner.is_first_player else 0
    return 0.5

def play(next_actions):
    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()

    while True:
        if state.is_done():
            break;
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break

        next_action = next_actions[0] if state.turn_owner.is_first_player else next_actions[1]
        action, _ = next_action(state)

        state = state.next(action)
    
    return first_player_point(state)

def evaluate_network(batch, count, total_point):
    model0 = ModelWrapper(MODEL_DIR/'latest.h5')

    model1 = ModelWrapper(MODEL_DIR/'best.h5')

    next_action0 = next_action_by(pv_ismcts, model0, EN_TEMPERATURE)
    next_action1 = next_action_by(pv_ismcts, model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    for i in range(batch):
        if i % 2 == 0:
            point = play(next_actions)
        else:
            point = 1 - play(list(reversed(next_actions)))
        
        total_point.value += point
        count.value += 1
        print('\rEvaluate {}/{}'.format(count.value, EN_GAME_COUNT), end='')

    K.clear_session()
    del model0
    del model1

def multi_process_evaluate_network(process_num):
    manager = multiprocessing.Manager()
    count = manager.Value('i', 0)
    total_point = manager.Value('d', 0)
    processes = []
    
    batch = EN_GAME_COUNT//process_num
    for _ in range(process_num):
        process = multiprocessing.Process(target=evaluate_network, kwargs={'batch': batch, 'count':count, 'total_point': total_point})
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    
    print('')
    average_point = total_point.value / EN_GAME_COUNT
    print('AveragePoint', average_point)

    # if average_point > 0.51:
    #     update_best_player()
    #     return True
    # else:
    #     return False

if __name__ == '__main__':
    multi_process_evaluate_network(5)