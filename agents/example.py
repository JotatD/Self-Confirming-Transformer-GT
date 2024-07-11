import gym

env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')

env.reset()

_, rew1, _, info = env.step(0, 0)
_, rew2, _, info = env.step(0, 1)
_, rew3, _, info = env.step(1, 0)
_, rew4, _, info = env.step(1, 1)


print(rew1)
print(rew2)
print(rew3)
print(rew4)
