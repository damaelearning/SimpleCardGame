from game import DECK_NUM, State, FIELDS_NUM, INITIAL_LIFE, HANDS_NUM
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import tensorflow as tf
import copy
from const import MODEL_DIR

PV_EVALUATE_COUNT = 50

def predict(model, state):
    

    a, b, c = DN_INPUT_SHAPE
    x = np.array([
            state.resize_zero_padding(state.get_status_list(state.fields), [a, b]), 
            state.resize_zero_padding(state.get_status_list(state.enemy_fields), [a, b]), 
            state.resize_zero_padding(state.get_status_list(state.hands), [a, b]),
            state.resize_zero_padding(state.get_status_list(state.enemy_hands), [a, b]),
            state.resize_zero_padding(state.get_status_list(state.deck), [a, b]),
            state.resize_zero_padding(state.get_status_list(state.enemy_deck), [a, b]),
            state.resize_zero_padding(state.get_attackable_list(state.fields), [a, b]),
            [[state.life for _ in range(b)] for _ in range(a)],
            [[state.enemy_life for _ in range(b)] for _ in range(a)],
            [[float(state.can_play_hand()) for _ in range(b)] for _ in range(a)]])
    x = x.transpose(1, 2, 0)
    x = x.reshape(1, a, b, c)
    x = x / INITIAL_LIFE
    y = model.predict(x, batch_size=1)
    
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1
    
    value = y[1][0][0]
    return policies, value

def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores
    
def pv_mcts_scores(model, state, temperature):
    class node:
        def __init__(self, state, p):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = None
        
        def evaluate(self):
            if self.state.is_done():
                if self.state.is_lose():
                    value = -1
                if self.state.is_win():
                    value = 1
                else:
                    value = 0
                
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
                if next_child_node.state.is_first_player() == self.state.is_first_player():
                    value = next_child_node.evaluate()
                else:
                    value = -next_child_node.evaluate()
                
                self.w += value
                self.n += 1
                return value
                
        def next_child_node(self):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                w = -child_node.w if child_node.state.is_first_player() != self.state.is_first_player() else child_node.w
                pucb_values.append((w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            
            return self.child_nodes[np.argmax(pucb_values)]
            
    root_node = node(state, 0)
    
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()
    
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores

def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

def boltzman(xs, temperature):
    xs = [x ** (1/ temperature) for x in xs]
    return [x / sum(xs) for x in xs]

if __name__ == '__main__':
    path = sorted(Path(MODEL_DIR).glob('*.h5'))[-1]
    model = load_model(str(path))
    
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
    
    next_action = pv_mcts_action(model, 1.0)
    
    while True:
        if state.is_done():
            break;
        state = state.start_turn() if state.is_starting_turn() else state
        if state.is_done():
            break
        
        action = next_action(state)
        
        state = state.next(action)
        
        print(state)