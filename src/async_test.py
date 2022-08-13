from functools import total_ordering
from game import State, Actor, playout, argmax, ismcts_action
from game import FIELDS_NUM, HANDS_NUM, DECK_NUM, INITIAL_LIFE, LIMIT_PP
from game import INITIAL_DECK, TOTAL_ACTION, PASS_NUM
import time
import math
import concurrent.futures
import asyncio

def damy(reward):
        return reward

def async_ismcts_action(state):
    class Node:
        def __init__(self, is_first_player):
            self.w = 0
            self.n = 0
            self.is_first_player = is_first_player
            self.child_nodes = None
        
        def evaluate(self, state, route, executor, loop, task):
            if state.is_done():
                self.n += 1
                reward = -1 if state.turn_owner.is_lose() else 1
                task.append(loop.run_in_executor(executor, damy, reward))
                task.append(state.turn_owner.is_first_player)
            
            elif not self.child_nodes:
                self.n += 1
                
                if self.n == 10:
                    self.expand()
                task.append(loop.run_in_executor(executor, playout, state))
                task.append(state.turn_owner.is_first_player)
            
            else:
                self.n += 1
                action = self.next_action(state)
                route.append(action)
                next_state = state.next(action)
                next_state = next_state.start_turn() if next_state.is_starting_turn else next_state
                next_child_node = self.child_nodes[action]
                
                next_child_node.evaluate(next_state, route, executor, loop, task)

        def expand(self):
            self.child_nodes = [Node(not state.turn_owner.is_first_player if i == PASS_NUM 
                        else state.turn_owner.is_first_player) for i in range(TOTAL_ACTION)]

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
                w = -child_node.w if action==PASS_NUM else child_node.w
                ucb1_values.append(w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return legal_actions[argmax(ucb1_values)]
        
        def rewarding(self, reward, route, child_node_is_first):
            self.w += reward if child_node_is_first == self.is_first_player else -reward
            if len(route) != 0:
                action = route.pop()
                self.child_nodes[action].rewarding(reward, route, child_node_is_first)

    async def develop_game_tree(root_node):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(1250):
                tasks = []
                routes = []
                for _ in range(8):
                    task = []
                    route = []
                    root_node.evaluate(state, route, executor, loop, task)
                    tasks.append(task)
                    routes.append(route)
                    
                for task, route in zip(tasks, routes):
                    reward = await task[0]
                    root_node.rewarding(reward, route[::-1], task[1])
    
    if len(state.legal_actions()) == 1:
        return PASS_NUM
    
    root_node = Node(state.turn_owner.is_first_player)
    root_node.expand()
    asyncio.run(develop_game_tree(root_node))

    legal_actions = state.legal_actions()
    n_list = []
    for action in legal_actions:
        c = root_node.child_nodes[action]
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

if __name__ == '__main__':
    # first_player = Actor(is_first_player=True)
    # second_player = Actor(is_first_player=False)
    # state = State(first_player, second_player)
    # state = state.game_start()
    # state = state.start_turn()
    # state = state.next(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM)
    # state = state.start_turn()
    # state = state.next(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM)
    # state = state.start_turn()
    # time1 = time.time()
    # async_ismcts_action(state)
    # time2 = time.time()
    # print(time2 - time1)

    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()

    while True:
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break
        
        if state.turn_owner.is_first_player:
            time1 = time.time()
            state = state.next(async_ismcts_action(state))
            time2 = time.time()
            print("multiprocess time ", time2 - time1)
        else:
            time1 = time.time()
            state = state.next(ismcts_action(state))
            time2 = time.time()
            print("singleprocess time ", time2 - time1)
        
        print(state)
        print()