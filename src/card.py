import random

MAX_STATUS = 3

class Card:
    def __init__(self, playPoint=None, attack=None, health=None, is_attackable=None):
        self.playPoint = playPoint
        self.attack = attack if attack != None else random.randint(1, MAX_STATUS)
        self.health = health if health != None else random.randint(1, MAX_STATUS)
        self.is_attackable = is_attackable if is_attackable != None else True
        self.has_fanfare = False
    
    def copy(self):
        return Card(self.playPoint, self.attack, self.health, self.is_attackable)

    def damage(self, value):
        self.health -= value

    def fanfare(self):
        return None


class CardWithDraw(Card):
    def __init__(self, playPoint=None, attack=None, health=None, is_attackable=None):
        super().__init__(playPoint, attack, health, is_attackable)
        self.has_fanfare = True

    def copy(self):
        return CardWithDraw(self.playPoint, self.attack, self.health, self.is_attackable)

    def fanfare(self):
        return ["draw", 1]
