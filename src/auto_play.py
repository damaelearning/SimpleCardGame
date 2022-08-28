from faulthandler import cancel_dump_traceback_later
import imp
import tensorflow as tf
import time
from game import INITIAL_LIFE, State, Actor, FIELDS_NUM, HANDS_NUM
from pv_mcts import next_action_by, pv_mcts, pv_ismcts, convert_state_to_input
from dual_network import DN_OUTPUT_SIZE, DN_INPUT_SHAPE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import  numpy as np
import pickle
import os
import multiprocessing
import concurrent.futures as futures
from const import MODEL_DIR, HISTORY_DIR
import platform
from functools import partial

from search import mcts_action, ismcts_action

SP_GAME_COUNT = 1000
SP_TEMPERATURE = 1.0
model = load_model(MODEL_DIR/'best.h5')

def first_player_value(ended_state):
    if ended_state.turn_owner.is_lose():
        return -1 if ended_state.turn_owner.is_first_player else 1
    if ended_state.enemy.is_lose():
        return 1 if ended_state.turn_owner.is_first_player else -1
    return 0

def write_data(historys):
    now = datetime.now()
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = HISTORY_DIR/'{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(historys, f)

def play(next_actions, logging=False):
    # for logging
    history = []
    height, width, channel = DN_INPUT_SHAPE

    # initialize game state
    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()

    while True:
        if state.is_done():
            break
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break

        next_action = next_actions[0] if state.turn_owner.is_first_player else next_actions[1]
        action, policy = next_action(state)

        if logging:
            input = convert_state_to_input(state, height, width, channel)
            history.append([input, policy, state.turn_owner.is_first_player])

        state = state.next(action)
    
    value = first_player_value(state)
    if logging:
        for i in range(len(history)):
            history[i][2] = value if history[i][2] else -value
    return value, history

def play_with_action_name(action_names, temperature, logging=False):
    next_actions = []
    for action in action_names:
        if action == "mcts":
            next_actions.append(mcts_action)
        if action == "ismcts":
            next_actions.append(ismcts_action)
        if action == "pv_mcts": 
            next_actions.append(next_action_by(search=pv_mcts, model=model, temperature=temperature))
        if action == "pv_ismcts":
            next_actions.append(next_action_by(search=pv_ismcts, model=model, temperature=temperature))
    return play(next_actions, logging)


def self_play(process_num, game_count):
    canceled = False
    action_names = ("pv_mcts", "pv_mcts")
    with futures.ProcessPoolExecutor(max_workers=process_num) as executor:
        results = [executor.submit(play_with_action_name, action_names=action_names, temperature= 1.0, logging = True) for _ in range(game_count)]
        count = 0
        historys = []
        try:
            for result in futures.as_completed(results):
                _, history = result.result()
                historys.extend(history)

                count += 1
                print('\rSelfPlay {}/{}'.format(count, SP_GAME_COUNT), end='')
        except KeyboardInterrupt:
            for future in results:
                cancel = future.cancel()
                print(cancel)
            canceled = True
            for process in executor._processes.values():
                process.kill()

    if not canceled : write_data(historys)
    print("finished")

if __name__ == '__main__':
    self_play(2, 1000)