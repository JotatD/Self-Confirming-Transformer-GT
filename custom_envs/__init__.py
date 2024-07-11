from gym.envs.registration import register

register(
    id='IteratedPrisoner-v0',
    entry_point='custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner')