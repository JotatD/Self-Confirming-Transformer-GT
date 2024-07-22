import gym
from agents import MemXY
from functools import partial
import random
env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')

env.reset()

t1 = MemXY(1, 2, 'dcCDCDDCDD')

print(t1.first_moves)
print(t1.dictionary_values)
print(t1.dictionary)

for i in range(10):
    act = t1.get_action()
    rand = random.choice([0, 1])
    t1.update(act, rand)
    print(act, rand)


