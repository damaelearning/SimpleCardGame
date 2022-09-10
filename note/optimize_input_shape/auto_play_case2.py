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


class ModelWrapperCase2(ModelWrapper):
    def predict(self, state):
        height, width, channel = (9, 5, 16)
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
        sizes = [width, (h_hands, width), (h_deck, width)]
        funcs = [cls._get_attack_list, cls._get_health_list, cls._get_play_point_list, cls._get_card_type_list]
        #for turn owner
        input=[]
        cards_list = [turn_owner.fields, turn_owner.hands, turn_owner.deck]
        for func in funcs:
            input.append(cls._get_card_para_channel(func, cards_list, sizes)*coef)
        input.append(cls._get_attackable_channel(turn_owner.fields, height, width))
        input.extend(cls._get_player_para_channel(turn_owner, height, width, coef))
        #for enemy
        cards_list = [enemy.fields, enemy.hands, enemy.deck]
        for func in funcs:
            input.append(cls._get_card_para_channel(func, cards_list, sizes)*coef)
        input.append(cls._get_attackable_channel(enemy.fields, height, width))
        input.extend(cls._get_player_para_channel(enemy, height, width, coef))
        return np.stack(input)
    
    @classmethod
    def _get_attackable_channel(cls, cards, height, width):
        channel = [cls._resize_zero_padding(cls._get_attackable_list(cards), width)]
        channel.extend([np.zeros((width)) for _ in range(height-1)])
        return np.vstack(channel)

    @classmethod
    def _get_player_para_channel(cls, player, height, width, coef):
        channel = [np.full((height, width), player.life*coef)]
        channel.append(np.full((height, width), player.max_play_point*coef))
        channel.append(np.full((height, width), player.play_point*coef))
        return channel

class AutoPlayCase2(AutoPlay):
    @staticmethod
    def get_actions(action_names, temperature, model_paths = [None, None]):
        next_actions = []
        for action, model_path in zip(action_names, model_paths):
            if action == "mcts":
                next_actions.append(mcts_action)
            if action == "ismcts":
                next_actions.append(ismcts_action)
            if action == "pv_mcts": 
                model = ModelWrapperCase2(model_path)
                next_actions.append(next_action_by(search=pv_mcts, model=model, temperature=temperature))
            if action == "pv_ismcts":
                model = ModelWrapperCase2(model_path)
                next_actions.append(next_action_by(search=pv_ismcts, model=model, temperature=temperature))
        return next_actions

    @staticmethod
    def _add_history(history, state, policy, shape):
        input = ModelWrapperCase2.convert_state_to_input(state, 9, 5, 16)
        return history+[[input, policy, state.turn_owner.is_first_player]]

if __name__ == "__main__":
    model_path = os.path.dirname(os.path.abspath(__file__))+"/case2_best.h5"
    model = ModelWrapperCase2(model_path)
    player = Actor(is_first_player=True)
    enemy = Actor(is_first_player=False)
    state = State(player, enemy)
    state = state.game_start()
    model.predict(state)