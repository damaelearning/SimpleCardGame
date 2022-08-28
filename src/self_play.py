import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
import time
from game import INITIAL_LIFE, State, FIELDS_NUM, HANDS_NUM
from pv_mcts import pv_ismcts_scores
from dual_network import DN_OUTPUT_SIZE, DN_INPUT_SHAPE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import  numpy as np
import pickle
import os
import multiprocessing
from const import MODEL_DIR, HISTORY_DIR

SP_GAME_COUNT = 1000
SP_TEMPERATURE = 1.0

def first_player_value(ended_state):
    if ended_state.turn_owner.is_lose():
        return -1 if ended_state.turn_owner.is_first_player else 1
    if ended_state.enemy.is_lose():
        return 1 if ended_state.turn_owner.is_first_player else -1
    return 0

def write_data(history):
    now = datetime.now()
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = HISTORY_DIR/'{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

def play(model):
    history = []
    
    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()

    a, b, _ = DN_INPUT_SHAPE
    coef = 1/INITIAL_LIFE

    while True:
        if state.is_done():
            break
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break

        scores = pv_ismcts_scores(model, state, SP_TEMPERATURE)

        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(range(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1), scores):
            policies[action] = policy
        
        x = [state.resize_zero_padding(state.get_attack_list(state.fields), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.fields), b) * coef, 
        state.resize_zero_padding(state.get_attack_list(state.hands), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.hands), b) * coef,
        state.resize_zero_padding(state.get_attack_list(state.deck), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.deck), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.enemy_deck), b) * coef,
        state.resize_zero_padding(state.get_attack_list(state.enemy_deck), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.enemy_hands), b) * coef,
        state.resize_zero_padding(state.get_attack_list(state.enemy_hands), b) * coef,
        state.resize_zero_padding(state.get_health_list(state.enemy_fields), b) * coef,  
        state.resize_zero_padding(state.get_attack_list(state.enemy_fields), b) * coef]

        history.append([[x,
                [[state.life * coef for _ in range(b)] for _ in range(a)],
                [[state.enemy_life * coef for _ in range(b)] for _ in range(a)],
                [state.resize_zero_padding(state.get_attackable_list(state.fields), b) for _ in range(a)],
                [[float(state.can_play_hand()) for _ in range(b)] for _ in range(a)]], 
                policies, 
                state.is_first_player()])

        action = np.random.choice(range(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1), p=scores)

        state = state.next(action)
    
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value if history[i][2] else -value
    return history

def self_play(batch, count,  history):
    model = load_model(MODEL_DIR/'best.h5')
    for i in range(batch):
        h = play(model)
        history.extend(h)

        count.value += 1
        print('\rSelfPlay {}/{}'.format(count.value, SP_GAME_COUNT), end='')

    K.clear_session()
    del model

def multiProcessSelfPlay(process_num):
    manager = multiprocessing.Manager()
    history = manager.list()
    count = manager.Value('i', 0)
    processes = []
    
    batch = SP_GAME_COUNT//process_num
    for _ in range(process_num):
        process = multiprocessing.Process(target=self_play, kwargs={'batch': batch, 'count':count, 'history': history})
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()

    print('')
    history = list(history)
    write_data(history)


if __name__ == '__main__':
    multiProcessSelfPlay(5)