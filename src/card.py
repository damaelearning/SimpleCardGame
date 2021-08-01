import random

MAX_STATUS = 3

class Card:
    def __init__(self, attack=None, health=None):
        self.attack = attack if attack != None else random.randint(1, MAX_STATUS)
        self.health = health if health != None else random.randint(1, MAX_STATUS)
        self.is_attackable = True
    
    def damage(self, value):
        self.health -= value