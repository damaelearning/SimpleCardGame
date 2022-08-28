from ast import Pass
from cmath import e
import random
from unittest import case
import numpy as np
import math
import copy
import sys
from card import Card, CardWithDraw
import concurrent.futures
import time

FIELDS_NUM = 5
HANDS_NUM = 9
DECK_NUM = 15
INITIAL_LIFE = 20
LIMIT_PP = 10
TOTAL_ACTION = FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1
PASS_NUM = FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM
INITIAL_DECK = [Card(1, 1, 2), Card(1, 1, 2), Card(1, 1, 2),
                Card(2, 2, 2),Card(2, 2, 2), Card(2, 2, 2), Card(2, 3, 1), Card(2, 1, 3),
                Card(3, 2, 3), Card(3, 2, 3), Card(3, 2, 4), Card(3, 4, 1),
                CardWithDraw(1, 1, 1), CardWithDraw(2, 1, 2), CardWithDraw(2, 2, 1), CardWithDraw(3, 2, 3), CardWithDraw(3, 2, 3)]
class State:
    def __init__(self, turn_owner, enemy, is_starting_turn=False):
        self.__turn_owner = turn_owner
        self.__enemy = enemy
        self.__is_starting_turn = is_starting_turn
        self.effect_dict = {"draw" : self.iterate_draw}
    
    @property
    def turn_owner(self):
        return self.__turn_owner

    @property
    def enemy(self):
        return self.__enemy
    
    @property
    def is_starting_turn(self):
        return self.__is_starting_turn
    
    #for Game State transition
    def is_done(self):
        return self.__turn_owner.is_lose() or self.__enemy.is_lose()
    
    def game_start(self):
        turn_owner = self.__turn_owner
        enemy = self.__enemy
        #add more two card to second player as handy
        for _ in range(2):
            turn_owner = turn_owner.draw_card()
        for _ in range(4):
            enemy = enemy.draw_card()
        return State(turn_owner, enemy, is_starting_turn=True)

    # @profile
    def next(self, action):
        #pass
        if action == FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM:
            return State(self.__enemy, self.__turn_owner, is_starting_turn=True) 
        #play hand
        elif action >= FIELDS_NUM*(FIELDS_NUM+1):
            hand_index = action - FIELDS_NUM*(FIELDS_NUM+1)
            play_card, turn_owner = self.__turn_owner.play_hand(hand_index)
            state = State(turn_owner, self.__enemy)
            if play_card.has_fanfare: state = state.activate_effect(play_card.fanfare())
            return state
        #battle
        else:
            attacker = action // (FIELDS_NUM+1)
            subject = action % (FIELDS_NUM+1)

            #life damege
            if subject == FIELDS_NUM:
                enemy = self.__enemy.damage_to_actor(self.__turn_owner.fields[attacker].attack)
                turn_owner = self.__turn_owner.make_card_non_attackable(attacker)
            #battle
            elif subject < FIELDS_NUM:
                turn_owner = self.__turn_owner.damage_to_card(attacker, self.__enemy.fields[subject].attack) 
                enemy = self.__enemy.damage_to_card(subject, self.__turn_owner.fields[attacker].attack)
                turn_owner = turn_owner.make_card_non_attackable(attacker)
                if turn_owner.fields[attacker].health <= 0:
                    _, turn_owner = turn_owner.pop_card(subject="fields", index=attacker)

                if enemy.fields[subject].health <= 0:
                    _, enemy = enemy.pop_card(subject="fields", index=subject)
                
            return State(turn_owner, enemy)

    # @profile
    def legal_actions(self):
        actions = []
        #battle
        for i, card in enumerate(self.__turn_owner.fields):
            if card.is_attackable:
                for j, _ in enumerate(self.__enemy.fields):
                    actions.append(i*(FIELDS_NUM+1)+j)
                #attack player
                actions.append(i*(FIELDS_NUM+1)+FIELDS_NUM)
        #hand
        if len(self.__turn_owner.fields) < FIELDS_NUM:
            for i, card in enumerate(self.__turn_owner.hands):
                if card.play_point <= self.__turn_owner.play_point:
                    actions.append(FIELDS_NUM*(FIELDS_NUM+1)+i)
        #pass
        actions.append(PASS_NUM)
        return actions
    
    def start_turn(self):
        turn_owner = self.__turn_owner.draw_card()
        turn_owner = turn_owner.make_all_cards_attackable()
        turn_owner = turn_owner.inclease_max_PP() if turn_owner.max_play_point < LIMIT_PP else turn_owner
        turn_owner = turn_owner.set_play_point(turn_owner.max_play_point)

        return State(turn_owner, self.__enemy, is_starting_turn=False)
    
    def activate_effect(self, effect):
        return self.effect_dict[effect[0]](effect[1])

    def iterate_draw(self, iterate_num):
        turn_owner = self.__turn_owner
        for i in range(iterate_num):
            turn_owner = turn_owner.draw_card()

        return State(turn_owner, self.__enemy)
    
    
    #to display Game State
    def __str__(self):
        def getStr(first_player, second_player):
            def getCardsStr(cards, max_num):
                str = ''
                for card in cards:
                    str += '[{0:1d}:{1:1d}/{2:1d}] '.format(card.play_point, card.attack, card.health)
                
                for i in range(max_num - len(cards)):
                    str += '[   ] '
                return str
            str = '         {0:2d}  {1:1d}/{2:1d}\n'.format(second_player.life, second_player.play_point, second_player.max_play_point)
            str += getCardsStr(second_player.hands, HANDS_NUM) + '\n'
            str += getCardsStr(second_player.fields, FIELDS_NUM) + '\n'
            str += getCardsStr(first_player.fields, FIELDS_NUM) + '\n'
            str += getCardsStr(first_player.hands, HANDS_NUM) + '\n'
            str += '         {0:2d}  {1:1d}/{2:1d}\n'.format(first_player.life, first_player.play_point, first_player.max_play_point)
            return str
        if self.__turn_owner.is_first_player:
            return getStr(self.__turn_owner, self.__enemy)
        else:
            return getStr(self.__enemy, self.__turn_owner)

class Actor:
    def __init__(self, life=INITIAL_LIFE, max_play_point=0, play_point=0, fields=[], hands=[], deck=INITIAL_DECK, is_first_player=False,
            is_library_out=False):
        self.__life = life
        self.__max_play_point = max_play_point
        self.__play_point = play_point
        self.__fields = fields
        self.__hands = hands
        self.__deck = deck
        self.__is_first_player = is_first_player
        self.__is_library_out = is_library_out

    @property
    def life(self):
        return self.__life
    
    @property
    def max_play_point(self):
         return self.__max_play_point
    
    @property
    def play_point(self):
        return self.__play_point

    @property
    def fields(self):
        return self.__fields
    
    @property
    def hands(self):
        return self.__hands
    
    @property
    def deck(self):
        return self.__deck
    
    @property
    def is_first_player(self):
        return self.__is_first_player
    
    @property
    def is_library_out(self):
        return self.__is_library_out

    def __pick_random_card(self, cards):
        next_cards = [card.copy() for card in cards]
        np.random.shuffle(next_cards)
        choiced_card = next_cards.pop()
        
        return next_cards, choiced_card
    
    def __add_card(self, cards, card):
        next_cards = [card.copy() for card in cards]
        next_cards.append(card)
               
        return next_cards

    def is_lose(self):
        if self.__life > 0 and not self.__is_library_out:
            return False
        return True

    def pop_card(self, subject, index):
        if subject == "fields":
            cards = [card.copy() for card in self.__fields]
            card = cards.pop(index)
            actor = Actor(self.__life,
                    self.__max_play_point,
                    self.__play_point,
                    cards,
                    self.__hands,
                    self.__deck,
                    self.__is_first_player)
        if subject == "hands":
            cards = [card.copy() for card in self.__hands]
            actor = Actor(self.__life,
                    self.__max_play_point,
                    self.__play_point,
                    self.__fields,
                    cards,
                    self.__deck,
                    self.__is_first_player)
        
        return card, actor

    def draw_card(self):
        if len(self.__deck) > 0:    
            next_deck, picked_card = self.__pick_random_card(self.__deck)
            next_hands = self.__add_card(self.__hands, picked_card) if len(self.__hands) < HANDS_NUM else self.__hands
            actor = Actor(
                self.__life,
                self.__max_play_point,
                self.__play_point,
                self.__fields,
                next_hands,
                next_deck,
                self.__is_first_player,
                is_library_out=False)
        else:
            actor = Actor(
                self.__life,
                self.__max_play_point,
                self.__play_point,
                self.__fields,
                self.__hands,
                self.__deck,
                self.__is_first_player,
                is_library_out=True)
        
        return actor
    
    # @profile
    def play_hand(self, hand_index):
        hands = [card.copy() for card in self.__hands]
        fields = [card.copy() for card in self.__fields]
        card = hands.pop(hand_index)
        fields.append(card)
        next_play_point = self.__play_point - card.play_point
        return card, Actor(self.__life,
                    self.__max_play_point,
                    next_play_point,
                    fields, 
                    hands,
                    self.__deck,
                    self.__is_first_player)

    def damage_to_actor(self, damage):
        return Actor(self.__life - damage,
                    self.__max_play_point,
                    self.__play_point,
                    self.__fields, 
                    self.__hands,
                    self.__deck,
                    self.__is_first_player)
    
    # @profile
    def damage_to_card(self, subject, damage):
        fields = [card.copy() for card in self.__fields]
        fields[subject].damage(damage)
        return Actor(self.__life,
                    self.__max_play_point,
                    self.__play_point,
                    fields, 
                    self.__hands,
                    self.__deck,
                    self.__is_first_player)

    def make_all_cards_attackable(self):
        fields = [card.copy() for card in self.__fields]
        for card in fields:
            card.become_attackable()
        return Actor(
                    self.__life,
                    self.__max_play_point,
                    self.__play_point,
                    fields, 
                    self.__hands,
                    self.__deck,
                    self.__is_first_player,
                    self.__is_library_out)
    
    def make_card_non_attackable(self, index):
        fields = [card.copy() for card in self.__fields]
        fields[index].become_non_attackable()
        return Actor(self.__life,
                    self.__max_play_point,
                    self.__play_point,
                    fields, 
                    self.__hands,
                    self.__deck,
                    self.__is_first_player,
                    self.__is_library_out)
    
    def inclease_max_PP(self):
        return Actor(
            self.__life,
            self.__max_play_point+1,
            self.__play_point,
            self.__fields, 
            self.__hands,
            self.__deck,
            self.__is_first_player,
            self.__is_library_out)
    
    def set_play_point(self, next_play_point):
        return Actor(
            self.__life,
            self.__max_play_point,
            next_play_point,
            self.__fields, 
            self.__hands,
            self.__deck,
            self.__is_first_player,
            self.__is_library_out)

def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]


def playout(state):
    if state.is_done():
        if state.turn_owner.is_lose():
            return -1
        if state.enemy.is_lose():
            return 1

    drawn_state = state.start_turn() if state.is_starting_turn else state
    if drawn_state.turn_owner.is_lose():
        return -1
    
    next_state = drawn_state.next(random_action(drawn_state))
    
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
                if next_child_node.state.turn_owner.is_first_player == self.state.turnowner.is_first_player:
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
        return FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM

    root_node = Node(state)
    root_node.expand()
    
    for _ in range(10000):
        root_node.evaluate()
    
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

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
        return PASS_NUM
    
    root_node = Node()
    root_node.expand()
    for _ in range(10000):
        root_node.evaluate(state)
    
    legal_actions = state.legal_actions()
    n_list = []
    for action in legal_actions:
        c = root_node.child_nodes[action]
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

if __name__ == '__main__':
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
        
        if state.turn_owner.is_first_player:
            state = state.next(ismcts_action(state))
        else:
            state = state.next(ismcts_action(state))
        
        
        print(state)
        print()


