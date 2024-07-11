import gym 
import numpy as np
class IteratedPrisoner(gym.Env):
    def __init__(self, payoffs=None, horizon=100):
        if payoffs is None:
            payoffs = {
                'T': 5,
                'R': 3,
                'P': 1,
                'S': 0
            }
        assert payoffs['T'] > payoffs['R'] > payoffs['P'] > payoffs['S'] \
            and 2 * payoffs['R'] > (payoffs['T'] + payoffs['S'])

        self.payoffs = self.dictionary_to_matrix(payoffs)
        
        self.horizon = horizon
        self.timestep = 0
        
    def reset(self):
        self.timestep = 0
        return 0
    
    def step(self, player_1, player_2):
        self.timestep += 1
        
        rewards = {
            "player_1": self.payoffs[0][player_1, player_2],
            "player_2": self.payoffs[1][player_1, player_2]
        }
        
        return 0, rewards, self.timestep == self.horizon, {'timestep': self.timestep}
    
    def dictionary_to_matrix(self, payoffs):
        p1_matrix = np.array([[payoffs['R'], payoffs['S']], [payoffs['T'], payoffs['P']]])
        p2_matrix = np.array([[payoffs['R'], payoffs['T']], [payoffs['S'], payoffs['P']]])
        return np.stack([p1_matrix, p2_matrix], axis=0)