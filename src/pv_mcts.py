from game import DECK_NUM, PASS_NUM, TOTAL_ACTION, State, Actor, FIELDS_NUM, INITIAL_LIFE, HANDS_NUM
from model_wrapper import ModelWrapper
from dual_network import DN_INPUT_SHAPE
from math import sqrt, log10, ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import tensorflow as tf
import copy
import time
import sys
from const import MODEL_DIR

PV_EVALUATE_COUNT = 100

class Node:
    def __init__(self, state=None, p=0, is_certain_node=True):
        if is_certain_node:
            if not state: 
                print("Certeain node class need arg 'state'.")
                sys.exit()
            self.set_as_certain_node()
        else:
            self.set_as_uncertain_node()
        
        self.state = state
        self.p = p
        self.w = 0
        self.n = 0
        self.child_nodes = None
    
    def evaluate(self, model, state):
        if state.is_done():
            if state.turn_owner.is_lose():
                value = -1
            if state.enemy.is_lose():
                value = 1
            
            self.w += value
            self.n += 1
            return value
        
        if not self.child_nodes:
            policies, value = model.predict(state)
            
            self.w += value
            self.n += 1
            
            self.expand_node_func(state, policies)
            return value
    
        else:
            next_child_node, next_state = self.select_node_func(state, model)

            if next_state.turn_owner.is_first_player == state.turn_owner.is_first_player:
                value = next_child_node.evaluate(model, next_state)
            else:
                value = -next_child_node.evaluate(model, next_state)
            
            self.w += value
            self.n += 1
            return value
    
    def nodes_to_scores(self):
        scores = []
        for c in self.child_nodes:
            scores.append(c.n)
        return scores
            
    def select_certain_node(self, state, model):
        t = sum(self.nodes_to_scores())
        C_PUCT = log10((1+t+19652)/19652+1.25)
        pucb_values = []
        for child_node in self.child_nodes:
            w = -child_node.w if child_node.state.turn_owner.is_first_player != self.state.turn_owner.is_first_player else child_node.w
            pucb_values.append((w / child_node.n if child_node.n else 0.0) +
                C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
        
        next_child_node = self.child_nodes[np.argmax(pucb_values)]
        
        return next_child_node, next_child_node.state
    
    def select_uncertain_node(self, state, model):
        t = sum(self.nodes_to_scores())
        C_PUCT = log10((1+t+19652)/19652+1.25)
        policies, _ = model.predict(state)

        legal_actions = state.legal_actions()

        pucb_values = []
        for action, policy in zip(legal_actions, policies):
            child_node = self.child_nodes[action]
            w = -child_node.w if action == PASS_NUM else child_node.w
            pucb_values.append((w / child_node.n if child_node.n else 0.0) +
                C_PUCT * policy * sqrt(t) / (1 + child_node.n))
        
        action = legal_actions[np.argmax(pucb_values)]
        next_state = state.next(action)
        next_state = next_state.start_turn() if next_state.is_starting_turn else next_state
        return self.child_nodes[action], next_state
        
    def expand_certain_node(self, state, policies):
        self.child_nodes = []
        for action, policy in zip(state.legal_actions(), policies):
            self.child_nodes.append(self.__class__(state.next(action), policy, True))

    def expand_uncertain_node(self, state, policies):
        # state and policies are not used. They are defined just to much interface.
        self.child_nodes = []
        self.child_nodes = [self.__class__(is_certain_node=False) for _ in range(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1)]
    
    def set_as_certain_node(self):
        self.select_node_func = self.select_certain_node
        self.expand_node_func = self.expand_certain_node
    
    def set_as_uncertain_node(self):
        self.select_node_func = self.select_uncertain_node
        self.expand_node_func = self.expand_uncertain_node

# Search algolithms ##########################################################
def next_action_by(search, model, temperature=0):
    def next_action(state):
        policy = search(model, state, temperature)
        return np.random.choice(range(TOTAL_ACTION), p=policy), policy
    return next_action

def pv_mcts(model, state, temperature):
    if len(state.legal_actions()) == 1:
        policy = [0]*(TOTAL_ACTION)
        policy[-1] = 1
    else:
        root_node = Node(state, 0)
        root_node.set_as_certain_node()
        for _ in range(PV_EVALUATE_COUNT):
            root_node.evaluate(model, state)
        scores = root_node.nodes_to_scores()
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


def pv_ismcts(model, state, temperature):
    if len(state.legal_actions()) == 1:
        scores = [0]*(TOTAL_ACTION)
        scores[-1] = 1
    else:
        root_node = Node(is_certain_node=False)
        for _ in range(PV_EVALUATE_COUNT):
            root_node.evaluate(model, state)
        scores = root_node.nodes_to_scores()
        
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
    current_dir = os.getcwd()
    latest_path = current_dir+"/note/optimize_input_shape/case1_best.h5"
    best_path = current_dir+"/note/optimize_input_shape/case1_best.h5"
    latest_model = ModelWrapper(latest_path)
    best_model  = ModelWrapper(best_path)
    from search import ismcts_action

    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()
    
    next_actions = (next_action_by(pv_ismcts, best_model, 0.0), ismcts_action)

    while True:
        if state.is_done():
            break
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break
        
        action, _ = next_actions[0](state) if state.turn_owner.is_first_player else next_actions[1](state)
        
        state = state.next(action)
        
        print(state)