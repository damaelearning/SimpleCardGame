import tensorflow as tf
from const import MODEL_DIR
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
from game import State, random_action, mcts_action
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import multiprocessing
from const import MODEL_DIR

EP_GAME_COUNT = 100

def first_player_point(ended_state):
    if ended_state.is_lose() or not ended_state.is_win():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

def play(next_actions):
    state = State()
    for _ in range(2):
        for _ in range(2):
            state = state.get_card_drawn_state()
        state = State(
            state.enemy_life, 
            state.enemy_fields, 
            state.enemy_hands, 
            state.enemy_deck, 
            state.life, 
            state.fields, 
            state.hands, 
            state.deck, 
            not state.is_first_player(),
            isLibraryOut=False,
            canPlayHand=True,
            isStartingTurn=True)

    while True:
        if state.is_done():
            break;
        state = state.start_turn() if state.is_starting_turn() else state
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

        print(state)
    
    return first_player_point(state)

def evaluate_algorithm_of(label, next_actions, batch, count, total_point):
    for i in range(batch):
        if i % 2 == 0:
            point = play(next_actions)
        else:
            point = 1 - play(list(reversed(next_actions)))
        
        total_point.value += point
        count.value += 1
        print('\rEvaluate {}/{}'.format(count.value, EP_GAME_COUNT), end='')
    

def evaluate_best_player(batch, count, total_point):
    model = load_model(MODEL_DIR/'best.h5')

    next_pv_mcts_action = pv_mcts_action(model, 0.0)

    #next_actions = (next_pv_mcts_action, random_action)
    #evaluate_algorithm_of('VS_Random', next_actions)

    #next_actions = (next_pv_mcts_action, alpha_beta_action)
    #evaluate_algorithm_of('VS_AlphaBeta', next_actions)

    next_actions = (next_pv_mcts_action, mcts_action)
    evaluate_algorithm_of('VS_MCTS', next_actions, batch, count, total_point)

    K.clear_session()
    del model

def multi_process_evaluate_best_player(process_num):
    manager = multiprocessing.Manager()
    count = manager.Value('i', 0)
    total_point = manager.Value('d', 0)
    processes = []

    batch = EP_GAME_COUNT//process_num
    for _ in range(process_num):
        process = multiprocessing.Process(target=evaluate_best_player, kwargs={'batch': batch, 'count':count, 'total_point': total_point})
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    
    print('')

    average_point = total_point.value / EP_GAME_COUNT
    print('VS_MCTS', average_point)

if __name__ == '__main__':
    model = load_model(MODEL_DIR/'best.h5')

    next_pv_mcts_action = pv_mcts_action(model, 0.0)

    next_actions = (next_pv_mcts_action, mcts_action)

    play(next_actions)
