from game import DECK_NUM, TOTAL_ACTION, State, Actor, FIELDS_NUM, INITIAL_LIFE, HANDS_NUM
from dual_network import DN_INPUT_SHAPE
from math import sqrt, log10, ceil
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import tensorflow as tf
import copy
import time
from const import MODEL_DIR

# enable toe use GPU and limit memory
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

PV_EVALUATE_COUNT = 10


def predict(model, state):
    height, width, channel = DN_INPUT_SHAPE
    input = convert_state_to_input(state, height, width, channel)

    input = input.transpose(1, 2, 0)
    input = input.reshape(1, height, width, channel)
    output = model.predict_on_batch(input)
    policies = output[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1
    
    value = output[1][0][0]
    return policies, value

# To convert card parameters to channel ##################################
def convert_state_to_input(state, height, width, channel):
    turn_owner = state.turn_owner
    enemy = state.enemy
    h_hands = ceil(HANDS_NUM/width)
    h_deck = ceil(DECK_NUM/width)
    coef = 1/INITIAL_LIFE
    sizes = [width, (h_hands, width), (h_deck, width)]
    funcs = [get_attack_list, get_health_list, get_play_point_list, get_card_type_list]
    #for turn owner
    input=[]
    cards_list = [turn_owner.fields, turn_owner.hands, turn_owner.deck]
    for func in funcs:
        input.append(get_card_para_channel(func, cards_list, sizes))
    input.append(get_player_para_channel(turn_owner, width))
    #for enemy
    cards_list = [enemy.fields, enemy.hands, enemy.deck]
    for func in funcs:
        input.append(get_card_para_channel(func, cards_list, sizes))
    input.append(get_player_para_channel(enemy, width))
    input = np.stack(input)
    return input/coef


def get_card_para_channel(get_para_func, cards_list, sizes):
    channel = [resize_zero_padding(get_para_func(cards), size) 
        for cards, size in zip(cards_list, sizes)]
    return np.vstack(channel)

def get_player_para_channel(player, width):
    channel = [resize_zero_padding(get_attackable_list(player.fields), width)]
    channel.extend([np.full((width), player.life) for _ in range(2)])
    channel.extend([np.full((width), player.max_play_point) for _ in range(2)])
    channel.extend([np.full((width), player.play_point) for _ in range(2)])
    return np.vstack(channel)

def resize_zero_padding(input_list, size):
    return_array = np.array(input_list)
    return_array.resize(size, refcheck=False)
    return return_array

def get_card_type_list(input_list):
    return [int(card.has_fanfare) for card in input_list]

def get_play_point_list(input_list):
    return [card.play_point for card in input_list]

def get_attack_list(input_list):
    return [card.attack for card in input_list]

def get_health_list(input_list):
    return [card.health for card in input_list]

def get_attackable_list(input_list):
    return [int(card.is_attackable) for card in input_list]

def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# Search algolithms ##########################################################
def pv_mcts(model, state, temperature):
    class node:
        def __init__(self, state, p):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = None
        
        def evaluate(self):
            if self.state.is_done():
                if self.state.turn_owner.is_lose():
                    value = -1
                if self.state.enemy.is_lose():
                    value = 1
                
                self.w += value
                self.n += 1
                return value
            
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                
                self.w += value
                self.n += 1
                
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(node(self.state.next(action), policy))
                return value
        
            else:
                next_child_node = self.next_child_node()
                if next_child_node.state.turn_owner.is_first_player == self.state.turn_owner.is_first_player:
                    value = next_child_node.evaluate()
                else:
                    value = -next_child_node.evaluate()
                
                self.w += value
                self.n += 1
                return value
                
        def next_child_node(self):
            t = sum(nodes_to_scores(self.child_nodes))
            C_PUCT = log10((1+t+19652)/19652+1.25)
            pucb_values = []
            for child_node in self.child_nodes:
                w = -child_node.w if child_node.state.turn_owner.is_first_player != self.state.turn_owner.is_first_player else child_node.w
                pucb_values.append((w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            
            return self.child_nodes[np.argmax(pucb_values)]
            
    root_node = node(state, 0)
    
    if len(state.legal_actions()) == 1:
        policy = [0]*(TOTAL_ACTION)
        policy[-1] = 1
    else:
        for _ in range(PV_EVALUATE_COUNT):
            root_node.evaluate()
        scores = nodes_to_scores(root_node.child_nodes)
        policy = [0] * TOTAL_ACTION
        for action, score in zip(state.legal_actions(), scores):
            policy[action] = score
    
    if temperature == 0:
        action = np.argmax(policy)
        policy = np.zeros(TOTAL_ACTION)
        policy[action] = 1
    else:
        policy = boltzman(policy, temperature)
    return policy

def next_action_by(search, model, temperature=0):
    def next_action(state):
        policy = search(model, state, temperature)
        return np.random.choice(range(TOTAL_ACTION), p=policy), policy
    return next_action

def pv_ismcts(model, state, temperature):
    class node:
        def __init__(self):
            self.w = 0
            self.n = 0
            self.child_nodes = None
        
        def evaluate(self, state):
            if state.is_done():
                if state.turn_owner.is_lose():
                    value = -1
                if state.enemy.is_lose():
                    value = 1
                
                self.w += value
                self.n += 1
                return value
            
            if not self.child_nodes:
                _, value = predict(model, state)
                
                self.w += value
                self.n += 1
                
                self.child_nodes = [node() for _ in range(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1)]
                return value
        
            else:
                action = self.next_action(state)
                next_state = state.next(action)
                next_state = next_state.start_turn() if next_state.is_starting_turn else next_state
                next_child_node = self.child_nodes[action]
                if next_state.turn_owner.is_first_player == state.turn_owner.is_first_player:
                    value = next_child_node.evaluate(next_state)
                else:
                    value = -next_child_node.evaluate(next_state)
                
                self.w += value
                self.n += 1
                return value
                
        def next_action(self, state):
            t = sum(nodes_to_scores(self.child_nodes))
            C_PUCT = log10((1+t+19652)/19652+1.25)
            policies, _ = predict(model, state)

            legal_actions = state.legal_actions()

            pucb_values = []
            for action, policy in zip(legal_actions, policies):
                child_node = self.child_nodes[action]
                child_state = state.next(action)
                w = -child_node.w if child_state.turn_owner.is_first_player != state.turn_owner.is_first_player else child_node.w
                pucb_values.append((w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * policy * sqrt(t) / (1 + child_node.n))
            
            return legal_actions[np.argmax(pucb_values)]
    
    if len(state.legal_actions()) == 1:
        scores = [0]*(TOTAL_ACTION)
        scores[-1] = 1
    else:
        root_node = node()
        for _ in range(PV_EVALUATE_COUNT):
            root_node.evaluate(state)
        scores = nodes_to_scores(root_node.child_nodes)
        
    if temperature == 0:
        action = np.argmax(scores)
        policy = np.zeros(len(scores))
        policy[action] = 1
    else:
        policy = boltzman(scores, temperature)
    
    return policy

def boltzman(xs, temperature):
    xs = [x ** (1/ temperature) for x in xs]
    return [x / sum(xs) for x in xs]

if __name__ == '__main__':
    path = sorted(Path(MODEL_DIR).glob('best.h5'))[-1]
    model = load_model(str(path))
    

    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()
    
    next_action = next_action_by(pv_ismcts, model, 1.0)

    while True:
        if state.is_done():
            break;
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break
        
        action = next_action(state)
        
        state = state.next(action)
        
        print(state)