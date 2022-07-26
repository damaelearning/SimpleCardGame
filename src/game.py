import random
import numpy as np
import math
import copy
import sys
from card import Card

FIELDS_NUM = 3
HANDS_NUM = 5
DECK_NUM = 15
INITIAL_LIFE = 20
class State:
    def __init__(self, life = None,
                fields=None,
                hands=None,
                deck=None,
                enemy_life = None,
                enemy_fields=None,
                enemy_hands=None,
                enemy_deck=None,
                isFirstPlayer=True,
                isLibraryOut=False,
                canPlayHand=True,
                isStartingTurn=True):
        self.life = life if life != None else INITIAL_LIFE
        self.enemy_life = enemy_life if enemy_life != None else INITIAL_LIFE
        self.fields = fields if fields != None else []
        self.enemy_fields = enemy_fields if enemy_fields != None else []
        if deck == None:
            self.deck = [Card(i+1, j+1) for i in range(3) for j in range(3)]
            self.deck.extend([Card(i+1, j+1) for i in range(2) for j in range(3)])
        else: 
            self.deck = deck
        if enemy_deck == None:
            self.enemy_deck = [Card(i+1, j+1) for i in range(3) for j in range(3)]
            self.enemy_deck.extend([Card(i+1, j+1) for i in range(2) for j in range(3)])
        else: 
            self.enemy_deck = enemy_deck
        self.hands = hands if hands != None else []
        self.enemy_hands = enemy_hands if enemy_hands != None else []
        self.__isFirstPlayer = isFirstPlayer
        self.__isLibraryOut = isLibraryOut
        self.__canPlayHand = canPlayHand
        self.__isStartingTurn = isStartingTurn
    
    def pick_random_card(self, cards):
        next_cards = [Card(card.attack, card.health, card.is_attackable) for card in cards]
        np.random.shuffle(next_cards)
        choiced_card = next_cards.pop()
        
        return next_cards, choiced_card
    
    def add_card(self, cards, card):
        next_cards = [Card(card.attack, card.health, card.is_attackable) for card in cards]
        next_cards.append(card)
               
        return next_cards
    
    def get_card_drawn_state(self):
        if len(self.deck) > 0:    
            next_deck, picked_card = self.pick_random_card(self.deck)
            next_hands = self.add_card(self.hands, picked_card) if len(self.hands) < HANDS_NUM else self.hands
            random.shuffle(next_hands)
            state = State(
                self.life,
                self.fields,
                next_hands,
                next_deck,
                self.enemy_life,
                self.enemy_fields,
                self.enemy_hands,
                self.enemy_deck,
                self.is_first_player(),
                isLibraryOut=False,
                canPlayHand=self.can_play_hand(),
                isStartingTurn=self.is_starting_turn()
                )
        else:
            state = State(
                self.life,
                self.fields,
                self.hands,
                self.deck,
                self.enemy_life,
                self.enemy_fields,
                self.enemy_hands,
                self.enemy_deck,
                self.is_first_player(),
                isLibraryOut=True,
                canPlayHand=self.can_play_hand(),
                isStartingTurn=self.is_starting_turn())
        
        return state
    
    def is_lose(self):
        if self.life > 0 and not self.is_library_out():
            return False
        return True

    def is_win(self):
        if self.enemy_life > 0:
            return False
        return True
    
    def is_draw(self):
        return False
    
    def is_done(self):
        return self.is_win() or self.is_lose() or self.is_draw()
    
    def next(self, action):
        hands = [Card(card.attack, card.health, card.is_attackable) for card in self.hands]
        fields = [Card(card.attack, card.health, card.is_attackable) for card in self.fields]
        enemy_fields = [Card(card.attack, card.health, card.is_attackable) for card in self.enemy_fields]
        enemy_life = self.enemy_life
        #pass
        if action == FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM:
            return State(
                enemy_life,
                enemy_fields,
                self.enemy_hands,
                self.enemy_deck,
                self.life,
                fields,
                hands,
                self.deck,
                not self.is_first_player(),
                isLibraryOut=False,
                canPlayHand=True,
                isStartingTurn=True) 
        #play hand
        elif action >= FIELDS_NUM*(FIELDS_NUM+1):
            handId = action - FIELDS_NUM*(FIELDS_NUM+1)
            card = hands.pop(handId)
            fields.append(card)
            return State(
                    self.life,
                    fields, 
                    hands,
                    self.deck,
                    enemy_life,
                    enemy_fields,
                    self.enemy_hands,
                    self.enemy_deck,
                    self.is_first_player(),
                    self.is_library_out(),
                    canPlayHand=False,
                    isStartingTurn=False)
        #battle
        else:
            attacker = action // (FIELDS_NUM+1)
            subject = action % (FIELDS_NUM+1)
            fields[attacker].is_attackable = False

            #life damege
            if subject == FIELDS_NUM:
                enemy_life -= fields[attacker].attack
            #battle
            elif subject < FIELDS_NUM:
                fields[attacker].damage(enemy_fields[subject].attack) 
                enemy_fields[subject].damage(fields[attacker].attack)
                if fields[attacker].health <= 0:
                    del fields[attacker]

                if enemy_fields[subject].health <= 0:
                    del enemy_fields[subject]
            
            return State(
                    self.life,
                    fields, 
                    hands,
                    self.deck,
                    enemy_life,
                    enemy_fields,
                    self.enemy_hands,
                    self.enemy_deck,
                    self.is_first_player(),
                    self.is_library_out(),
                    canPlayHand=self.can_play_hand(),
                    isStartingTurn=False)
    
    def legal_actions(self):
        actions = []
        #battle
        for i, card in enumerate(self.fields):
            if card.is_attackable:
                for j, _ in enumerate(self.enemy_fields):
                    actions.append(i*(FIELDS_NUM+1)+j)
                #attack player
                actions.append(i*(FIELDS_NUM+1)+FIELDS_NUM)
        #hand
        if self.can_play_hand() and len(self.fields) < FIELDS_NUM:
            for i, _ in enumerate(self.hands):
                actions.append(FIELDS_NUM*(FIELDS_NUM+1)+i)
        #pass
        actions.append(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM)
        return actions
    
    def is_first_player(self):
        return self.__isFirstPlayer
    
    def is_library_out(self):
        return self.__isLibraryOut
    
    def is_starting_turn(self):
        return self.__isStartingTurn
    
    def can_play_hand(self):
        return self.__canPlayHand

    def make_all_cards_attackable(self):
        fields = [Card(card.attack, card.health, card.is_attackable) for card in self.fields]
        for card in fields:
            card.is_attackable = True
        return State(
                    self.life,
                    fields, 
                    self.hands,
                    self.deck,
                    self.enemy_life,
                    self.enemy_fields,
                    self.enemy_hands,
                    self.enemy_deck,
                    self.is_first_player(),
                    self.is_library_out(),
                    canPlayHand=self.can_play_hand(),
                    isStartingTurn=self.is_starting_turn())

    def start_turn(self):
        self.__isStartingTurn=False
        state = self.get_card_drawn_state()
        state = state.make_all_cards_attackable()
        return state
    
    
    def resize_zero_padding(self, input_list, size):
        return_array = np.array(input_list)
        return_array.resize(size, refcheck=False)
        return return_array
    
    def get_attack_list(self, input_list):
        return [card.attack for card in input_list]

    def get_health_list(self, input_list):
        return [card.health for card in input_list]

    def get_attackable_list(self, input_list):
        return [float(card.is_attackable) for card in input_list]
    
    def __str__(self):
        def getStr(firstPlayerFields, firstPlayerLife, firstPlayerHands, secondPlayerFields, secondPlayerLife, secondPlayerHands):
            def getCardsStr(cards, max_num):
                str = ''
                for card in cards:
                    str += '[{0:1d}/{1:1d}] '.format(card.attack, card.health)
                
                for i in range(max_num - len(cards)):
                    str += '[   ] '
                return str
            str = '         {0:2d}\n'.format(secondPlayerLife)
            str += getCardsStr(secondPlayerHands, HANDS_NUM) + '\n'
            str += getCardsStr(secondPlayerFields, FIELDS_NUM) + '\n'
            str += getCardsStr(firstPlayerFields, FIELDS_NUM) + '\n'
            str += getCardsStr(firstPlayerHands, HANDS_NUM) + '\n'
            str += '         {0:2d}\n'.format(firstPlayerLife)
            return str
        if self.is_first_player():
            return getStr(self.fields, self.life, self.hands, self.enemy_fields, self.enemy_life, self.enemy_hands)
        else:
            return getStr(self.enemy_fields, self.enemy_life, self.enemy_hands, self.fields, self.life, self.hands)

def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

def playout(state):
    drawn_state = state.start_turn() if state.is_starting_turn() else state
    
    if drawn_state.is_win():
        return 1

    if drawn_state.is_lose():
        return -1
    
    if drawn_state.is_draw():
        return 0
    
    next_state = drawn_state.next(random_action(drawn_state))
    
    if next_state.is_first_player() == drawn_state.is_first_player():
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
                value = playout(self.state)
                
                self.w += value
                self.n += 1
                
                if self.n == 10:
                    self.expand()
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
                w = -child_node.w if child_node.state.is_first_player() != self.state.is_first_player() else child_node.w
                ucb1_values.append(w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return self.child_nodes[argmax(ucb1_values)]
    
    if len(state.legal_actions()) == 1:
        scores = [0]*(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1)
        scores[-1] = 1
    else:
        root_node = Node(state)
        root_node.expand()
    
    for _ in range(1000):
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
                if state.is_lose():
                    value = -1
                if state.is_win():
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
                next_state = next_state.start_turn() if next_state.is_starting_turn() else next_state
                next_child_node = self.child_nodes[action]
                if next_state.is_first_player() == state.is_first_player():
                    value = next_child_node.evaluate(next_state)
                else:
                    value = -next_child_node.evaluate(next_state)

                self.w += value
                self.n += 1
                return value
            
        def expand(self):
            self.child_nodes = [Node() for _ in range(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM+1)]
            
        
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
                child_state = state.next(action)
                w = -child_node.w if child_state.is_first_player() != state.is_first_player() else child_node.w
                ucb1_values.append(w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return legal_actions[argmax(ucb1_values)]
    
    root_node = Node()
    root_node.expand()
    
    for _ in range(1000):
        root_node.evaluate(state)
    
    legal_actions = state.legal_actions()
    n_list = []
    for action in legal_actions:
        c = root_node.child_nodes[action]
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

if __name__ == '__main__':
    state = State()
    for _ in range(2):
        for _ in range(2):
            state = state.get_card_drawn_state()
        state = State(state.enemy_life, state.enemy_fields, state.enemy_hands, state.enemy_deck, state.life, state.fields, state.hands, state.deck, not state.is_first_player())

    while True:
        state = state.start_turn() if state.is_starting_turn() else state
        if state.is_done():
            break
        
        if state.is_first_player:
            state = state.next(mcts_action(state))
        else:
            state = state.next(ismcts_action(state))
        
        
        print(state)
        print()