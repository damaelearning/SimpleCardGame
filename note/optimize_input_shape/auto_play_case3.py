from statistics import mode
import sys
import os
sys.path.append(os.path.abspath("src"))
from dual_network import dual_network
from auto_play import AutoPlay
from model_wrapper import ModelWrapper
from const import MODEL_DIR
from train_network import train_network
from game import State, Actor, INITIAL_LIFE, HANDS_NUM, DECK_NUM
from pv_mcts import next_action_by, pv_mcts, pv_ismcts
from search import mcts_action, ismcts_action
from math import ceil
import numpy as np


class ModelWrapperCase3(ModelWrapper):
    def predict(self, state):
        height, width, channel = (40, 5, 2)
        input = self.convert_state_to_input(state, height, width, channel)
        input = input.transpose(1, 2, 0)
        input = input.reshape(1, height, width, channel)
        output = self.model.predict_on_batch(input)
        policies = output[0][0][list(state.legal_actions())]
        policies /= sum(policies) if sum(policies) else 1
        
        value = output[1][0][0]
        return policies, value

    @classmethod
    def convert_state_to_input(cls, state, height, width, channel):
        turn_owner = state.turn_owner
        enemy = state.enemy
        h_hands = ceil(HANDS_NUM/width)
        h_deck = ceil(DECK_NUM/width)
        coef = 1/INITIAL_LIFE
        sizes = (width, h_hands, h_deck)
        funcs = [cls._get_attack_list, cls._get_health_list, cls._get_play_point_list, cls._get_card_type_list, cls._get_attackable_list]
        input=[]
        input.append(cls._get_channel(funcs, turn_owner, *sizes))
        input.append(cls._get_channel(funcs, enemy, *sizes))
        return np.stack(input)*coef

    @classmethod
    def _get_card_para_layer(cls, get_para_func, cards, size):
        return [cls._resize_zero_padding(get_para_func(cards), size)]

    @classmethod
    def _get_channel(cls, funcs, player, width, h_hands, h_deck):
        channel = []
        for func in funcs:
            channel.extend(cls._get_card_para_layer(func, player.fields, width))
        for func in funcs[:4]:
            channel.extend(cls._get_card_para_layer(func, player.hands, (h_hands, width)))
        for func in funcs[:4]:
            channel.extend(cls._get_card_para_layer(func, player.deck, (h_deck, width)))
        channel.extend([np.full(width, player.life)])
        channel.extend([np.full(width, player.max_play_point)])
        channel.extend([np.full(width, player.play_point)])
        return np.vstack(channel)

class AutoPlayCase3(AutoPlay):
    @staticmethod
    def get_actions(action_names, temperature, model_paths = [None, None]):
        next_actions = []
        for action, model_path in zip(action_names, model_paths):
            if action == "mcts":
                next_actions.append(mcts_action)
            if action == "ismcts":
                next_actions.append(ismcts_action)
            if action == "pv_mcts": 
                model = ModelWrapperCase3(model_path)
                next_actions.append(next_action_by(search=pv_mcts, model=model, temperature=temperature))
            if action == "pv_ismcts":
                model = ModelWrapperCase3(model_path)
                next_actions.append(next_action_by(search=pv_ismcts, model=model, temperature=temperature))
        return next_actions

    @staticmethod
    def _add_history(history, state, policy, shape):
        input = ModelWrapperCase3.convert_state_to_input(state, 40, 5, 2)
        return history+[[input, policy, state.turn_owner.is_first_player]]

if __name__ == "__main__":
    self_play = AutoPlayCase3()
    self_play.set_action1("ismcts")
    self_play.set_action2("ismcts")
    self_play.make_play_log(5, 10, 1.0, "Self Play")