from pydoc import importfile
import random
from game import PASS_NUM, FIELDS_NUM, HANDS_NUM, TOTAL_ACTION
import math

def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)], None


def playout(state):
    if state.is_done():
        if state.turn_owner.is_lose():
            return -1
        if state.enemy.is_lose():
            return 1

    drawn_state = state.start_turn() if state.is_starting_turn else state
    if drawn_state.turn_owner.is_lose():
        return -1
    
    action, _ = random_action(drawn_state)
    next_state = drawn_state.next(action)
    
    if next_state.turn_owner.is_first_player == drawn_state.turn_owner.is_first_player:
        return playout(next_state)
    else:
        return -playout(next_state)

def argmax(collection, key=None):
    return collection.index(max(collection))

def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.state = state
            self.w = 0
            self.n = 0
            self.child_nodes = None
        
        def evaluate(self):
            if self.state.is_done():
                if self.state.turn_owner.is_lose():
                    value = -1
                if self.state.enemy.is_lose():
                    value = 1
                else:
                    value = 0
                
                self.w += value
                self.n += 1
                return value
            
            if not self.child_nodes:
                value = playout(self.state)
                
                self.w += value
                self.n += 1
                
                if self.n == 10:
                    self.expand()
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
            
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))
        
        def next_child_node(self):
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                w = -child_node.w if child_node.state.turn_owner.is_first_player != self.state.turn_owner.is_first_player else child_node.w
                ucb1_values.append(w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return self.child_nodes[argmax(ucb1_values)]
    
    if len(state.legal_actions()) == 1:
        policy = [0]*(TOTAL_ACTION)
        policy[-1] = 1
        return FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM, policy

    root_node = Node(state)
    root_node.expand()
    
    for _ in range(10000):
        root_node.evaluate()
    
    legal_actions = state.legal_actions()
    n_list = [0]*TOTAL_ACTION
    for action in legal_actions:
        c = root_node.child_nodes[action]
        n_list[action] = c.n
    return legal_actions[argmax(n_list)], [n/sum(n_list) for n in n_list]

def ismcts_action(state):
    class Node:
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
                else:
                    value = 0
                
                self.w += value
                self.n += 1
                return value
            
            if not self.child_nodes:
                value = playout(state)
                
                self.w += value
                self.n += 1
                
                if self.n == 10:
                    self.expand()
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
  
        def expand(self):
            self.child_nodes = [Node() for _ in range(TOTAL_ACTION)]
  
        def next_action(self, state):
            legal_actions = state.legal_actions()
            for action in legal_actions:
                if self.child_nodes[action].n == 0:
                    return action
            
            t = 0
            for action in legal_actions:
                t += self.child_nodes[action].n
            ucb1_values = []
            for action in legal_actions:
                child_node = self.child_nodes[action]
                w = -child_node.w if action == PASS_NUM else child_node.w
                ucb1_values.append(w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return legal_actions[argmax(ucb1_values)]
    
    if len(state.legal_actions()) == 1:
        policy = [0]*(TOTAL_ACTION)
        policy[-1] = 1
        return PASS_NUM, policy
    
    root_node = Node()
    root_node.expand()
    for _ in range(10000):
        root_node.evaluate(state)
    
    legal_actions = state.legal_actions()
    n_list = [0]*TOTAL_ACTION
    for action in legal_actions:
        c = root_node.child_nodes[action]
        n_list[action] = c.n
    return argmax(n_list), [n/sum(n_list) for n in n_list]