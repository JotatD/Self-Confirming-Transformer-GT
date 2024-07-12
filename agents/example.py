import gym
from agents import TestAgent
from functools import partial
env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')

env.reset()

t1 = TestAgent()
t2 = TestAgent()

for i in range(15):
    a1 = t1.get_action()
    a2 = t2.get_action()
    _, rew, _, _ = env.step(a1, a2)
    
    print(rew)
    
    t1.update(a1, a2)
    t2.update(a2, a1)
    
    
print("t1's own history", t1.own_history)
print("t1's opponent history", t1.opponent_history)
print("t1's counter", t1.counter)
print("t2's own history", t2.own_history)
print("t2's opponent history", t2.opponent_history)
print("t2's counter", t2.counter)

t1_history = [1, 1, 1, 1]
t2_history = [0, 0, 0, 0]

t1.force_history(t1_history, t2_history)
t2.force_history(t2_history, t1_history)

print("t1's own history", t1.own_history)
print("t1's opponent history", t1.opponent_history)
print("t1's counter", t1.counter)
print("t2's own history", t2.own_history)
print("t2's opponent history", t2.opponent_history)
print("t2's counter", t2.counter)

variable_class = partial(TestAgent, counter=10)
instance = variable_class()
print("instance counter", instance.counter)

