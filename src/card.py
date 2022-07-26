import random

MAX_STATUS = 3

class Card:
    def __init__(self, attack=None, health=None, is_attackable=None):
        self.attack = attack if attack != None else random.randint(1, MAX_STATUS)
        self.health = health if health != None else random.randint(1, MAX_STATUS)
        self.is_attackable = is_attackable if is_attackable != None else True
    
    def damage(self, value):
        self.health -= value


class CardWithPP:
    def __init__(self, playPoint=None, attack=None, health=None, is_attackable=None):
        self.playPoint = playPoint
        self.attack = attack if attack != None else random.randint(1, MAX_STATUS)
        self.health = health if health != None else random.randint(1, MAX_STATUS)
        self.is_attackable = is_attackable if is_attackable != None else True
        self.has_fanfare = False
    
    def damage(self, value):
        self.health -= value

    def fanfare(self):
        return None


class CardWithDraw(CardWithPP):
    def __init__(self, playPoint=None, attack=None, health=None, is_attackable=None):
        super().__init__(playPoint, attack, health, is_attackable)
        self.has_fanfare = True

    def fanfare(self):
        return ["draw", 1]
