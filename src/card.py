import random

MAX_STATUS = 3

class Card:
    def __init__(self, play_point, attack, health, is_attackable=False):
        self._play_point = play_point
        self._attack = attack
        self._health = health
        self._is_attackable = is_attackable
        self._has_fanfare = False
    
    @property
    def play_point(self):
        return self._play_point

    @property
    def attack(self):
        return self._attack
    
    @property
    def health(self):
        return self._health
    
    @property
    def is_attackable(self):
        return self._is_attackable

    @property
    def has_fanfare(self):
        return self._has_fanfare

    def copy(self):
        return self.__class__(self._play_point, self._attack, self._health, self._is_attackable)

    def damage(self, value):
        self._health -= value

    def become_non_attackable(self):
        self._is_attackable = False

    def become_attackable(self):
        self._is_attackable = True

    def fanfare(self):
        return None


class CardWithDraw(Card):
    def __init__(self, play_point, attack, health, is_attackable=False):
        super().__init__(play_point, attack, health, is_attackable)
        self._has_fanfare = True

    def fanfare(self):
        return ["draw", 1]

class CardHasStorm(Card):
    def __init__(self, play_point, attack, health, is_attackable=True):
        super().__init__(play_point, attack, health, is_attackable)

