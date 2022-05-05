import random
import numpy
import copy

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, done):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent using Monte Carlo Rollout Planning.
"""
class MonteCarloRolloutPlanner(Agent):

    def __init__(self, params):
        super(MonteCarloRolloutPlanner, self).__init__(params)
        self.env = params["env"]
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.simulations = params["simulations"]  
            
    def calc_discounted_return(self, env, random_plan):
        discounted_return = 0
        # simulate generateive model actions
        time_step = 0
        for action in random_plan:
            _, reward, _, _ = env.step(action)
            # calculate discount return 
            discounted_return += (self.gamma**time_step)*reward
            time_step += 1
        return discounted_return
        
    def policy(self, state):
        """ Policy = Strategie/Verhaltensmuster
            Gegeben einer policy z.B. random werden zufällig Pläne erzeugt.
            Die returns aggregiert und dann den Durchschnittswert bildet.
            Am Ende wird auf den Durchschnittswerten (Q-Values) eine Entscheidung trifft.
        """
        # Tracks Q-values of each action (in the first step)
        Q_values = numpy.zeros(self.nr_actions)
        # Tracks number of each action selections (in the first step).
        action_counts = numpy.zeros(self.nr_actions)
        
        for _ in range(self.simulations):
            # Copying the current environment provides a simulator for planning.
            generative_model = copy.deepcopy(self.env)
            # Create random plan with length H
            # 1) Create random plan
            random_plan = [random.randint(0, self.nr_actions-1) for n in range(self.horizon)]
            # 2) Calculate discount return
            discounted_return = self.calc_discounted_return(generative_model, random_plan)
            Q_values[random_plan[0]] = discounted_return
            action_counts[random_plan[0]] += 1

        Q_values = numpy.divide(Q_values, action_counts)
        return numpy.argmax(Q_values)
