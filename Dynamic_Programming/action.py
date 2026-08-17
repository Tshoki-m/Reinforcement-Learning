

#Define the basic actions. Give each action a numerical value. 
# the def __Str__(self): just means that for example if later you run
#action = Action.DOWN 
#print(action)  will print "Down" instead of "Action.Down"
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __str__(self):
        return self.name.capitalize()

