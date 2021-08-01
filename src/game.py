import random
import numpy as np
import math
import copy
import sys

FIELDS_NUM = 3
HANDS_NUM = 5
DECK_NUM = 15
INITIAL_LIFE = 10
class State:
    def __init__(self, life = None,
                fields=None,
                hands=None,
                deck=None,
                enemy_life = None,
                enemy_fields=None,
                enemy_hands=None,
                enemy_deck=None,
                _isFirstPlayer=True,
                isLibraryOut=False):
        self.life = life if life != None else INITIAL_LIFE
        self.enemy_life = enemy_life if enemy_life != None else INITIAL_LIFE
        self.fields = fields if fields != None else [[0, 0] for _ in range(FIELDS_NUM)]
        self.enemy_fields = enemy_fields if enemy_fields != None else [[0 ,0] for _ in range(FIELDS_NUM)]
        self.deck = deck if deck != None else [[random.randint(1, 3), random.randint(1,3)] for _ in range(DECK_NUM)]
        self.enemy_deck = enemy_deck if enemy_deck != None else [[random.randint(1, 3), random.randint(1,3)] for _ in range(DECK_NUM)]
        self.hands = hands if hands != None else [[0, 0] for _ in range(HANDS_NUM)]
        self.enemy_hands = enemy_hands if enemy_hands != None else [[0, 0] for _ in range(HANDS_NUM)]
        self._isFirstPlayer = _isFirstPlayer
        self._isLibraryOut = isLibraryOut
    
    def pick_random_card(self, cards):
        next_cards = copy.deepcopy(cards)
        choiced_card = random.choice([[i, card] for i, card in enumerate(next_cards) if card[1] > 0])
        
        next_cards[choiced_card[0]] = [0, 0]
        
        return next_cards, choiced_card[1]
    
    def add_card(self, cards, card):
        next_cards = copy.deepcopy(cards)
        if self.card_count(next_cards) < len(next_cards):
            vacant_index = [i for i, _ in enumerate(next_cards) if not _[0] > 0][0]
            next_cards[vacant_index] = card
        else:
            print("The cards don't have any [0, 0] zone, input cards which have at least one vacant zone.", file=sys.stderr)
        
        return next_cards
    
    def get_card_drawn_state(self):
        if self.card_count(self.deck) > 0:    
            next_deck, picked_card = self.pick_random_card(self.deck)
            next_hands = self.add_card(self.hands, picked_card) if self.card_count(self.hands) < HANDS_NUM else self.hands
            state = State(self.life, self.fields, next_hands, next_deck, self.enemy_life, self.enemy_fields, self.enemy_hands, self.enemy_deck, self.is_first_player(), isLibraryOut=False)
        else:
            state = State(self.life, self.fields, self.hands, self.deck, self.enemy_life, self.enemy_fields, self.enemy_hands, self.enemy_deck, self.is_first_player(), isLibraryOut=True)
        return state

    def card_count(self, cards):
        return sum([i[1] > 0 for i in cards])
    
    def is_lose(self):
        if self.life > 0 and not self.is_library_out():
            return False
        return True
    
    def is_draw(self):
        return False
    
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    def next(self, action):
        hands = copy.deepcopy(self.hands)
        fields = copy.deepcopy(self.fields)
        enemy_fields = copy.deepcopy(self.enemy_fields)
        enemy_life = self.enemy_life
        #pass
        if action == FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM:
            pass
        #play hand
        elif action >= FIELDS_NUM*(FIELDS_NUM+1):
            handId = action - FIELDS_NUM*(FIELDS_NUM+1)
            fieldId = [i for i, x in enumerate(fields) if x[1] == 0][0]
            fields[fieldId][0] = hands[handId][0]
            fields[fieldId][1] = hands[handId][1]
            for i in range(handId+1, HANDS_NUM):
                hands[i-1][0] = hands[i][0]
                hands[i-1][1] = hands[i][1]
            hands[HANDS_NUM-1][0] = 0
            hands[HANDS_NUM-1][1] = 0
        else:
            attacker = action // (FIELDS_NUM+1)
            subject = action % (FIELDS_NUM+1)         
 
            #life damege
            if subject == FIELDS_NUM:
                enemy_life -= fields[attacker][0]
            #battle
            elif subject < FIELDS_NUM:
                fields[attacker][1] -= enemy_fields[subject][0]
                enemy_fields[subject][1] -= fields[attacker][0]
                if fields[attacker][1] <= 0:
                    for i in range(attacker+1, FIELDS_NUM):
                        fields[i-1][0] = fields[i][0]
                        fields[i-1][1] = fields[i][1]
                    fields[FIELDS_NUM-1][0] = 0
                    fields[FIELDS_NUM-1][1] = 0

                if enemy_fields[subject][1] <= 0:
                    for i in range(subject+1, FIELDS_NUM):
                        enemy_fields[i-1][0] = enemy_fields[i][0]
                        enemy_fields[i-1][1] = enemy_fields[i][1]
                    enemy_fields[FIELDS_NUM-1][0] = 0
                    enemy_fields[FIELDS_NUM-1][1] = 0
        
        return State(enemy_life, enemy_fields, self.enemy_hands, self.enemy_deck, self.life, fields, hands, self.deck, not self.is_first_player(), self.is_library_out())
    
    def legal_actions(self):
        actions = []
        #battle
        for i, card in enumerate(self.fields):
            if card[1] > 0:
                for j, enemy_card in enumerate(self.enemy_fields):
                    if enemy_card[1] > 0:
                        actions.append(i*(FIELDS_NUM+1)+j)
                actions.append(i*(FIELDS_NUM+1)+FIELDS_NUM)
        #hand
        if self.card_count(self.fields) < FIELDS_NUM:
            for i, card in enumerate(self.hands):
                if card[1] > 0:
                    actions.append(FIELDS_NUM*(FIELDS_NUM+1)+i)
        #pass
        actions.append(FIELDS_NUM*(FIELDS_NUM+1)+HANDS_NUM)
        return actions
    
    def is_first_player(self):
        return self._isFirstPlayer
    
    def is_library_out(self):
        return self._isLibraryOut
    
    def resize_zero_padding(self, input_list, size):
        return_array = np.array(input_list)
        return_array.resize(size, refcheck=False)
        return return_array
    
    def __str__(self):
        def getStr(firstPlayerFields, firstPlayerLife, firstPlayerHands, secondPlayerFields, secondPlayerLife, secondPlayerHands):
            def getCardsStr(cards):
                str = ''
                for card in cards:
                    if card[1] > 0:
                        str += '[{0:1d}/{1:1d}] '.format(card[0], card[1])
                    else:
                        str += '[   ] '
                return str
            str = '         {0:2d}\n'.format(secondPlayerLife)
            str += getCardsStr(secondPlayerHands) + '\n'
            str += getCardsStr(secondPlayerFields) + '\n'
            str += getCardsStr(firstPlayerFields) + '\n'
            str += getCardsStr(firstPlayerHands) + '\n'
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
    drawn_state = state.get_card_drawn_state()
    if drawn_state.is_lose():
        return -1
    
    if drawn_state.is_draw():
        return 0
    return -playout(drawn_state.next(random_action(drawn_state)))

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
                value = -1 if self.state.is_lose() else 0
                
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
                value = -self.next_child_node().evaluate()

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
                ucb1_values.append(-child_node.w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
            
            return self.child_nodes[argmax(ucb1_values)]
    
    root_node = Node(state)
    root_node.expand()
    
    for _ in range(100):
        root_node.evaluate()
    
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

if __name__ == '__main__':
    state = State()
    for _ in range(2):
        for _ in range(2):
            state = state.get_card_drawn_state()
        state = State(state.enemy_life, state.enemy_fields, state.enemy_hands, state.enemy_deck, state.life, state.fields, state.hands, state.deck, not state.is_first_player())

    while True:
        state = state.get_card_drawn_state()
        if state.is_done():
            break
        state = state.next(mcts_action(state))
        
        print(state)
        print()