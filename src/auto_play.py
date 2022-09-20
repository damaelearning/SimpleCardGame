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
        self.history_dir = HISTORY_DIR
    
    def set_action1(self, action, model_path=None):
        self.action1 = action
        self.model_path1 = model_path
    
    def set_action2(self, action, model_path=None):
        self.action2 = action
        self.model_path2 = model_path
    
    def set_history_dir(self, dir):
        self.history_dir = dir

    @classmethod
    def play(cls, action_names, model_paths, temperature, logging=False, minus_value=False):
        actions = cls.get_actions(action_names, temperature, model_paths)
        # for logging
        history = []
        shape = DN_INPUT_SHAPE

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
                history = cls._add_history(history, state, policy, shape)

            state = state.next(action)
        
        value = first_player_value(state)
        if logging:
            for i in range(len(history)):
                history[i][2] = value if history[i][2] else -value
        if minus_value:
            value = -value
        return value, history

    @staticmethod
    def get_actions(action_names, temperature, model_paths = [None, None]):
        next_actions = []
        print(model_paths)
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

    @staticmethod
    def _add_history(history, state, policy, shape):
        input = ModelWrapper.convert_state_to_input(state, *shape)
        return history + [[input, policy, state.turn_owner.is_first_player]]

    def multi_play(self, process_num, game_count, action_names, model_paths, 
                        temperature, logging=False, label="Completed", alt_player=False):
        def alternate_actions(action_names, i):
            if alt_player and i%2 == 0:
                return action_names[::-1] 
            return action_names
        
        def alternate_model_path(model_path, i):
            if alt_player and i%2 == 0:
                return model_path[::-1] 
            return model_path

        def alternate_value(i):
            return alt_player and i%2==0
        
        count = 0
        total_value = 0
        historys = []
        canceled = False
        with futures.ProcessPoolExecutor(max_workers=process_num) as executor:
            results = [executor.submit(self.__class__.play, 
                    action_names=alternate_actions(action_names, i), temperature= temperature, 
                    model_paths=alternate_model_path(model_paths, i), 
                    logging = logging, minus_value=alternate_value(i)) 
                    for i in range(game_count)]
            try:
                for result in futures.as_completed(results):
                    value, history = result.result()
                    if logging:
                        historys.extend(history)
                    total_value += value

                    count += 1
                    print('\r{} {}/{}'.format(label, count, game_count), end='')
            except KeyboardInterrupt:
                for future in results:
                    future.cancel()
                canceled = True
                for process in executor._processes.values():
                    process.kill()

        if not canceled and logging: write_data(self.history_dir, historys)
        return total_value
    
    def make_play_log(self, process_num, game_count, temperature=1.0, 
                        process_name="Make play log", alt_player=False):
        print(process_name)
        action_names = [self.action1, self.action2]
        model_paths = [self.model_path1, self.model_path2]
        self.multi_play(process_num, game_count, action_names, model_paths, temperature, logging=True, alt_player=alt_player)

    def calc_win_rate(self, process_num, game_count, temperature=0.0, 
                        process_name=None, alt_player=False):
        if process_name == None:
            process_name = "Calc win rate {} VS {}".format(self.action1, self.action2)
        print(process_name)
        action_names = (self.action1, self.action2)
        model_paths = (self.model_path1, self.model_path2)
        value = self.multi_play(process_num, game_count, action_names, model_paths, temperature, alt_player=alt_player)
        value = value/game_count+0.5
        print("Win rate : {}".format(value))
        return value


def first_player_value(ended_state):
    if ended_state.turn_owner.is_lose():
        return -1 if ended_state.turn_owner.is_first_player else 1
    if ended_state.enemy.is_lose():
        return 1 if ended_state.turn_owner.is_first_player else -1
    return 0

def write_data(history_dir, historys):
    now = datetime.now()
    os.makedirs(history_dir, exist_ok=True)
    path = history_dir/'{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(historys, f)

if __name__ == '__main__':
    auto_play = AutoPlay()
    auto_play.set_action1("pv_ismcts", MODEL_DIR/'latest.h5')
    auto_play.set_action2("pv_ismcts", MODEL_DIR/'best.h5')
    auto_play.calc_win_rate(5, 30, 0.0, alt_player=True)