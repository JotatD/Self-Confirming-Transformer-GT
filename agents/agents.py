import numpy as np
import gym

class PrisonerAgent():
    def __init__(self):
        self.own_history = []
        self.opponent_history = []
    
    def get_action():
        raise NotImplementedError()
    
    def update(self, own_action, opponent_action, rewards=None):
        self.own_history.append(own_action)
        self.opponent_history.append(opponent_action)
        
    def reset(self):
        self.__init__()
        
    def replace_history(self, own_history, opponent_history):
        self.reset()
        for own_action, opponent_action in zip(own_history, opponent_history):
            self.update(own_action, opponent_action)
            
    def force_history(self, own_history, opponent_history):
        self.own_history = own_history
        self.opponent_history = opponent_history
            
class TestAgent(PrisonerAgent):
    def __init__(self, counter=0):
        super().__init__()
        self.counter = counter
        
    def get_action(self):
        return int(self.counter % 2)
    
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        self.counter += 1
        
class AllC(PrisonerAgent):
    def get_action(self):
        return 0

class AllD(PrisonerAgent):
    def get_action(self):
        return 1
    
class TitForTat(PrisonerAgent):
    def get_action(self):
        if len(self.opponent_history) == 0:
            return 0
        return self.opponent_history[-1]

class Spiteful(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.opponent_defected = False
        
    def get_action(self):
        if (self.opponent_defected):
            return 1
        else:
            return 0
        
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        if opponent_action == 1:
            self.opponent_defected = True

#TODO: Verify if this is correct
class SoftMajo(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.num_oponent_defection = 0
        self.num_oponent_cooperation = 0
    

    def get_action(self):
        num_oponent_defection = sum(self.opponent_history)
        num_oponent_cooperation = len(self.opponent_history) - num_oponent_defection
        if len(self.own_history) == 0 or \
            num_oponent_cooperation >= num_oponent_defection:
            return 0
        else:
            return 1
    
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        if opponent_action == 1:
            self.num_oponent_defection += 1
        else:
            self.num_oponent_cooperation += 1
        
#TODO: Verify if this is correct
class HardMajo(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.num_oponent_defection = 0
        self.num_oponent_cooperation = 0
        
    def get_action(self):
        if len(self.own_history) == 0 or \
            self.num_oponent_defection >= self.num_oponent_cooperation:
            return 1
        else:
            return 0
    
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        if opponent_action == 1:
            self.num_oponent_defection += 1
        else:
            self.num_oponent_cooperation += 1
        
class PerDDC(PrisonerAgent):
    def get_action(self):
        modulo = len(self.own_history) % 3
        if modulo == 0:
            return 1
        elif modulo == 1:
            return 1
        else:
            return 0
        
class PerCCD(PrisonerAgent):
    def get_action(self):
        modulo = len(self.own_history) % 3
        if modulo == 0:
            return 0
        elif modulo == 1:
            return 0
        else:
            return 1
        
class Mistrust(PrisonerAgent):
    def get_action(self):
        if len(self.opponent_history) == 0:
            return 1
        else:
            return self.opponent_history[-1]
        
class PerCD(PrisonerAgent):
    def get_action(self):
        modulo = len(self.own_history) % 2
        if modulo == 0:
            return 0
        else:
            return 1
        
class Pavlov(PrisonerAgent):
    def get_action(self):
        if len(self.own_history) == 0:
            return 0
        else:
            if self.own_history[-1] != self.opponent_history[-1]:
                        return 1
            else:
                return 0

class TF2T(PrisonerAgent):
    def get_action(self):
        if len(self.own_history) in [0, 1]:
            return 0
        else:
            if self.opponent_history[-1] == 1 and self.opponent_history[-2] == 1:
                return 1
            else:
                return 0
            
class HardTFT(PrisonerAgent):
    def get_action(self):
        if len(self.own_history) in [0, 1]:
            return 0
        else:
            if self.opponent_history[-1] == 1 or self.opponent_history[-2] == 1:
                return 1
            else:
                return 0
            
class SlowTFT(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.defect = False
    
    def get_action(self):
        return int(self.defect)
        
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        if len(self.own_history) > 1:
            if self.opponent_history[-1] == 1 and self.opponent_history[-2] == 1:
                self.defect = True
            
            elif self.opponent_history[-1] == 0 and self.opponent_history[-2] == 0:
                self.defect = False

#TODO: Verify if this is correct
class Gradual(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.num_defections_before_peace = 0
        self.two_time_cooperation_flag = False
        self.two_time_cooperation_counter = 0
        
    def get_action(self):
        return int(self.num_defections_before_peace > 0)
                
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        self.num_oponent_defections = sum(self.opponent_history)
                
        if self.num_defections_before_peace == 0:
            assert own_action == 0
            if self.two_time_cooperation_flag:
                self.two_time_cooperation_counter += 1
                if self.two_time_cooperation_counter == 2:
                    self.two_time_cooperation_flag = False
                    self.two_time_cooperation_counter = 0
            elif opponent_action == 1:
                self.num_defections_before_peace = self.num_oponent_defections
        else:
            assert own_action == 1
            self.num_defections_before_peace -= 1
            
            if self.num_defections_before_peace == 0:
                self.two_time_cooperation_flag = True
            
class Prober(PrisonerAgent):
    def __init__(self):
        super().__init__()
        self.is_baited = False
    
    def get_action(self):
        if len(self.own_history) == 0:
                return 1
        elif len(self.own_history) in [1, 2]:
                return 0
        elif self.is_baited:
                return 1
        else:
            return self.opponent_history[-1]
            
    
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        if len(self.own_history) == 3:
            if self.opponent_history[1] == 0 and self.opponent_history[2] == 0:
                self.is_baited = True
        
class Mem2(PrisonerAgent):
    def __init__(self, env_dict):
        super().__init__()
        self.sub_player = TitForTat()
        self.counter = 0
        self.all_d_chosen = 0 
        self.env_dict = env_dict
        
        self.own_matrix = np.array([[env_dict['R'], env_dict['S']], [env_dict['T'], env_dict['P']]])
                
    
    def get_action(self):
        return self.sub_player.get_action()
    
    def update(self, own_action, opponent_action, rewards=None):
        super().update(own_action, opponent_action)
        self.sub_player.update(own_action, opponent_action)
        self.counter += 1
        
        if self.all_d_chosen < 2 and self.counter % 2 == 0 and self.counter > 0:
            p_own_action = self.own_history[-2]
            p_opponent_action = self.opponent_history[-2]
            own_2_payoff = self.own_matrix[p_own_action, p_opponent_action] +\
                self.own_matrix[own_action, opponent_action]

            if own_2_payoff == (2 * self.env_dict['R']):
                self.sub_player = TitForTat()
            elif own_2_payoff == (self.env_dict['T'] + self.env_dict['S']):
                self.sub_player = TF2T()
            else:
                self.sub_player = AllD()
                self.all_d_chosen += 1
            self.sub_player.force_history(self.own_history, self.opponent_history)
    
    def reset(self):
        self.__init__(self.env_dict)
            
class MemXY(PrisonerAgent):
    def __init__(self, x, y, genotype):
        super().__init__()
        self.x = x
        self.y = y
        self.genotype = genotype
        counter = 0
        for char in genotype:
            if char in ['c', 'd']:
                counter += 1
            else:
                break
        for char in genotype[counter:]:
            assert char in ['C', 'D']
        
        self.first_moves = list(map(lambda x: int(x == 'd'), genotype[:counter]))
        self.dictionary_values = list(map(lambda x: int(x == 'D'), genotype[counter:]))
        
        assert len(self.first_moves) == max(x, y)
        assert len(self.dictionary_values) == 2 ** (x+y)
        
        self.dict_counter = 0
        self.dictionary = {}
        self.build_dictionary('', 1)
        
    def build_dictionary(self, current_str, depth):
        if depth == self.x + self.y:
            self.dictionary[current_str +'0'] = self.dictionary_values[self.dict_counter]
            self.dict_counter += 1
            self.dictionary[current_str +'1'] = self.dictionary_values[self.dict_counter]
            self.dict_counter += 1
        else:
            self.build_dictionary(current_str + '0', depth + 1)
            self.build_dictionary(current_str + '1', depth + 1)
            
    def get_action(self):
        if (len(self.first_moves) > 0):
            return int(self.first_moves.pop(0) == 1)
        else:
            key = "".join(map(str, self.own_history[-self.x:])) \
                + "".join(map(str, self.opponent_history[-self.y:]))
            return int(self.dictionary[key] == 1)     