import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from dual_network import DN_INPUT_SHAPE
import numpy as np
from game import INITIAL_LIFE, HANDS_NUM, DECK_NUM
from math import ceil
from tensorflow.keras.models import load_model

# enable to use GPU and limit memory
import platform
if platform.system() == "Darwin":
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name="gpu")
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")



class ModelWrapper():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, state):
        height, width, channel = DN_INPUT_SHAPE
        input = self.convert_state_to_input(state, height, width, channel)

        input = input.transpose(1, 2, 0)
        input = input.reshape(1, height, width, channel)
        output = self.model.predict_on_batch(input)
        policies = output[0][0][list(state.legal_actions())]
        policies /= sum(policies) if sum(policies) else 1
        
        value = output[1][0][0]
        return policies, value

    # To convert card parameters to channel ##################################
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
            input.append(cls._get_card_para_channel(func, cards_list, sizes))
        input.append(cls._get_player_para_channel(turn_owner, width))
        #for enemy
        cards_list = [enemy.fields, enemy.hands, enemy.deck]
        for func in funcs:
            input.append(cls._get_card_para_channel(func, cards_list, sizes))
        input.append(cls._get_player_para_channel(enemy, width))
        input = np.stack(input)
        return input*coef

    @classmethod
    def _get_card_para_channel(cls, get_para_func, cards_list, sizes):
        channel = [cls._resize_zero_padding(get_para_func(cards), size) 
            for cards, size in zip(cards_list, sizes)]
        return np.vstack(channel)

    @classmethod
    def _get_player_para_channel(cls, player, width):
        channel = [cls._resize_zero_padding(cls._get_attackable_list(player.fields), width)]
        channel.extend([np.full((width), player.life) for _ in range(2)])
        channel.extend([np.full((width), player.max_play_point) for _ in range(2)])
        channel.extend([np.full((width), player.play_point) for _ in range(2)])
        channel.extend([np.full((width), 0) for _ in range(2)])
        return np.vstack(channel)

    @staticmethod
    def _resize_zero_padding(input_list, size):
        return_array = np.array(input_list)
        return_array.resize(size, refcheck=False)
        return return_array

    @staticmethod
    def _get_card_type_list(input_list):
        return [int(card.has_fanfare) for card in input_list]

    @staticmethod
    def _get_play_point_list(input_list):
        return [card.play_point for card in input_list]

    @staticmethod
    def _get_attack_list(input_list):
        return [card.attack for card in input_list]

    @staticmethod
    def _get_health_list(input_list):
        return [card.health for card in input_list]

    @staticmethod
    def _get_attackable_list(input_list):
        return [int(card.is_attackable) for card in input_list]
    
