from faulthandler import cancel_dump_traceback_later
import tensorflow as tf
import time
from game import INITIAL_LIFE, State, Actor, FIELDS_NUM, HANDS_NUM
from pv_mcts import next_action_by, pv_mcts, pv_ismcts
from dual_network import DN_OUTPUT_SIZE, DN_INPUT_SHAPE
from datetime import datetime
from model_wrapper import ModelWrapper
from tensorflow.keras import backend as K
from pathlib import Path
import  numpy as np
import pickle
import multiprocessing
import concurrent.futures as futures
from const import MODEL_DIR, HISTORY_DIR
import platform
from functools import partial
import os

from search import mcts_action, ismcts_action

SP_GAME_COUNT = 1000
SP_TEMPERATURE = 1.0


class AutoPlay:
    def __init__(self):
        self.action1 = None
        self.action2 = None
        self.model_path1 = None
        self.model_path2 = None
    
    def set_action1(self, action, model_path=None):
        self.action1 = action
        self.model_path1 = model_path
    
    def set_action2(self, action, model_path=None):
        self.action2 = action
        self.model_path2 = model_path
    
    @classmethod
    def play(cls, action_names, model_paths, temperature, logging=False):
        actions = cls.get_actions(action_names, temperature, model_paths)
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

            next_action = actions[0] if state.turn_owner.is_first_player else actions[1]
            action, policy = next_action(state)

            if logging:
                input = ModelWrapper.convert_state_to_input(state, height, width, channel)
                history.append([input, policy, state.turn_owner.is_first_player])

            state = state.next(action)
        
        value = first_player_value(state)
        if logging:
            for i in range(len(history)):
                history[i][2] = value if history[i][2] else -value
        return value, history

    @staticmethod
    def get_actions(action_names, temperature, model_paths = [None, None]):
        next_actions = []
        for action, model_path in zip(action_names, model_paths):
            if action == "mcts":
                next_actions.append(mcts_action)
            if action == "ismcts":
                next_actions.append(ismcts_action)
            if action == "pv_mcts": 
                model = ModelWrapper(model_path)
                next_actions.append(next_action_by(search=pv_mcts, model=model, temperature=temperature))
            if action == "pv_ismcts":
                model = ModelWrapper(model_path)
                next_actions.append(next_action_by(search=pv_ismcts, model=model, temperature=temperature))
        return next_actions

    def multi_play(self, process_num, game_count, action_names, model_paths, 
                        temperature, logging=False, label="Completed", alt_player=True):
        def alternate_actions(action_names, i):
            if alt_player and i%2 == 1:
                return reversed(action_names) 
            return action_names
        
        count = 0
        value = 0
        historys = []
        canceled = False
        with futures.ProcessPoolExecutor(max_workers=process_num) as executor:
            results = [executor.submit(self.__class__.play, 
                    action_names=alternate_actions(action_names, i), temperature= temperature, 
                    model_paths=model_paths, logging = logging) 
                    for i in range(game_count)]
            try:
                for result in futures.as_completed(results):
                    value, history = result.result()
                    if logging:
                        historys.extend(history)
                    
                    value += value

                    count += 1
                    print('\r{} {}/{}'.format(label, count, game_count), end='')
            except KeyboardInterrupt:
                for future in results:
                    future.cancel()
                canceled = True
                for process in executor._processes.values():
                    process.kill()

        if not canceled and logging: write_data(historys)
        return value
    
    def make_play_log(self, process_num, game_count, temperature=1.0, 
                        process_name="Make play log"):
        print(process_name)
        action_names = (self.action1, self.action2)
        model_paths = (self.model_path1, self.model_path2)
        self.multi_play(process_num, game_count, action_names, model_paths, temperature, logging=True, alt_player=False)

    def calc_win_rate(self, process_num, game_count, temperature=0.0, 
                        process_name=None):
        if process_name == None:
            process_name = "Calc win rate {} VS {}".format(self.action1, self.action2)
        print(process_name)
        action_names = (self.action1, self.action2)
        model_paths = (self.model_path1, self.model_path2)
        value = self.multi_play(process_num, game_count, action_names, model_paths, temperature)
        print("Win rate : {}".format(value/game_count))
        return value


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

if __name__ == '__main__':
    auto_play = AutoPlay()
    auto_play.set_action1("pv_mcts", MODEL_DIR/'best.h5')
    auto_play.set_action2("pv_mcts", MODEL_DIR/'best.h5')
    auto_play.calc_win_rate(3, 1000, 1)
