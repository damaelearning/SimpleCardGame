from game import State, Actor
from search import random_action, mcts_action, ismcts_action
from pathlib import Path
import numpy as np
import multiprocessing
from const import MODEL_DIR

EP_GAME_COUNT = 100

def first_player_point(ended_state):
    if ended_state.turn_owner.is_lose():
        return 0 if ended_state.turn_owner.is_first_player else 1
    if ended_state.enemy.is_lose():
        return 1 if ended_state.turn_owner.is_first_player else 0
    return 0.5

def play(next_actions):
    first_player = Actor(is_first_player=True)
    second_player = Actor(is_first_player=False)
    state = State(first_player, second_player)
    state = state.game_start()

    while True:
        if state.is_done():
            break;
        state = state.start_turn() if state.is_starting_turn else state
        if state.is_done():
            break

        next_action = next_actions[0] if state.turn_owner.is_first_player else next_actions[1]
        action, _ = next_action(state)

        state = state.next(action)
    
    return first_player_point(state)

def evaluate_algorithm_of(next_actions, batch, count, total_point):
    for i in range(batch):
        point = play(next_actions)
        
        total_point.value += point
        count.value += 1
        print('\rEvaluate {}/{}'.format(count.value, EP_GAME_COUNT), end='')
    

def evaluate_best_player(batch, count, total_point):

    next_actions = (ismcts_action, ismcts_action)
    evaluate_algorithm_of(next_actions, batch, count, total_point)


def multi_process_evaluate_best_player(process_num):
    manager = multiprocessing.Manager()
    count = manager.Value('i', 0)
    total_point = manager.Value('d', 0)
    processes = []

    batch = EP_GAME_COUNT//process_num
    for _ in range(process_num):
        process = multiprocessing.Process(target=evaluate_best_player, kwargs={'batch': batch, 'count':count, 'total_point': total_point})
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    
    print('')

    average_point = total_point.value / EP_GAME_COUNT
    print('First player win rate', average_point)

if __name__ == '__main__':
    multi_process_evaluate_best_player(5)