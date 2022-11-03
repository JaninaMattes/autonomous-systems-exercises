import random
import numpy
import copy
from multi_armed_bandits import random_bandit, epsilon_greedy, boltzmann, ucb1


class Agent:
    """ Base class of an autonomously acting and learning agent. """

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    def policy(self, state):
        """ Behavioral strategy of the agent. Maps states to actions. """
        pass

    def update(self, state, action, reward, next_state, done):
        """ Learning method of the agent. Integrates experience into the agent's current knowledge. """
        pass
        

class RandomAgent(Agent):
    """ Randomly acting agent. """

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))


class MonteCarloRolloutPlanner(Agent):
    """ Autonomous agent using Monte Carlo Rollout Planning. """

    def __init__(self, params):
        super(MonteCarloRolloutPlanner, self).__init__(params)
        self.env = params["env"]
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.simulations = params["simulations"]
        
    def policy(self, state):
        # track Q-values of each action (in the first step)
        q_values = numpy.zeros(self.nr_actions)
        # track action selections (in the first step)
        action_counts = numpy.zeros(self.nr_actions)
        for _ in range(self.simulations):
            # copy the current environment provides a simulator for planning.
            generative_model = copy.deepcopy(self.env)

            random_plan = numpy.random.randint(0, self.nr_actions, self.horizon)
            discounted_return = .0
            done = False
            for t, action in enumerate(random_plan):
                if not done:
                    _, reward, done, _ = generative_model.step(action)
                    discounted_return += reward*(self.gamma**t)
            first_action = random_plan[0]
            action_counts[first_action] += 1
            q_values[first_action] += discounted_return
        return numpy.argmax(q_values/action_counts)


class MonteCarloTreeSearchNode:
    """ Represents a (state) node in Monte Carlo Tree Search. """
    
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.nr_actions = params["nr_actions"]
        self.q_values = numpy.zeros(self.nr_actions)
        self.action_counts = numpy.zeros(self.nr_actions)
        self.children = []

    def select(self, q_values, action_counts):
        """
         Selects an action according to a Multi-armed Bandit strategy.
         Returns the selected action.
        """
        return ucb1(q_values, action_counts)

    def expand(self):
        """ Appends a new child node to self.children. """
        self.children.append(MonteCarloTreeSearchNode(self.params))

    def rollout(self, generative_model, depth):
        """
         Performs a rollout for self.horizon-depth time steps.
         Returns the obtained discounted return.
        """
        random_plan = numpy.random.randint(0, self.nr_actions, self.horizon-depth)
        discounted_return = .0
        done = False
        for t, action in enumerate(random_plan):
            if not done:
                _, reward, done, _ = generative_model.step(action)
                discounted_return += reward*(self.gamma**t)
        return discounted_return

    def backup(self, discounted_return, action):
        """ Updates the Q-values of this node according to the observed discounted return and the selected action. """
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] = (n - 1) * self.q_values[action] + discounted_return
        self.q_values[action] /= n

    def final_decision(self):
        """
         Makes a final decision based on the currently learned Q-values.
         @return the action with the highest Q-value.
        """
        return numpy.argmax(self.q_values)

    def is_leaf(self):
        """
         Indicates if this node is still a leaf node.
         Returns true, if this leaf is not fully expanded yet, and false otherwise.
        """
        return len(self.children) < self.nr_actions

    def simulate(self, generative_model, depth):
        """
         Performs a simulation step in this node.
         Returns the discounted return observed from this node.
        """
        if depth >= self.horizon:
            return 0
        if self.is_leaf():
            self.expand()
            selected_action = len(self.children) - 1  # Select action that leads to new child node
            _, reward, done, _ = generative_model.step(selected_action)
            return self.simulate_with_rollout(generative_model, selected_action, depth)
        selected_action = self.select(self.q_values, self.action_counts)
        return self.simulate_with_selection(generative_model, selected_action, depth)

    def simulate_with_rollout(self, generative_model, action, depth):
        """
         Simulates and evaluates an action with a subsequent rollout.
         Returns the discounted return observed from this node.
        """
        return self.simulate_action(generative_model, action, depth, self.rollout)

    def simulate_with_selection(self, generative_model, action, depth):
        """
         Simulates and evaluates an action with a simulation at a child node.
         Returns the discounted return observed from this node.
        """
        return self.simulate_action(generative_model, action, depth, self.children[action].simulate)

    def simulate_action(self, generative_model, action, depth, eval_func):
        """
         Simulates and evaluates an action with a subsequent evaluation function.
         Returns the discounted return observed from this node.
        """
        _, reward, done, _ = generative_model.step(action)
        delayed_return = 0
        if not done:
            delayed_return = eval_func(generative_model, depth+1)
        discounted_return = reward + self.gamma*delayed_return
        self.backup(discounted_return, action)
        return discounted_return
        

class MonteCarloTreeSearchPlanner(Agent):
    """ Autonomous agent using Monte Carlo Tree Search for Planning. """

    def __init__(self, params):
        super(MonteCarloTreeSearchPlanner, self).__init__(params)
        self.params = params
        self.env = params["env"]
        self.simulations = params["simulations"]

    def policy(self, state):
        root = MonteCarloTreeSearchNode(self.params)
        for _ in range(self.simulations):
            generative_model = copy.deepcopy(self.env)
            root.simulate(generative_model, depth=0)
        return root.final_decision()


class SARSALearner(Agent):
    """ Autonomous agent using SARSA. """

    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.q_values = {} # contains q-table - state:action value
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        
    def q_table(self, state):
        state = numpy.array2string(state)
        if state not in self.q_values:
            self.q_values[state] = numpy.zeros(self.nr_actions)
        return self.q_values[state]

    def policy(self, state):
        q_values = self.q_table(state)
        # return epsilon_greedy(q_values, None, epsilon=self.epsilon)
        return boltzmann(q_values, self.nr_actions, epsilon=self.epsilon)

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table(state)[action]
        next_action = self.policy(next_state) # take next action "on policy"
        td_target = reward + self.gamma * self.q_table(next_state)[next_action]
        td_error = td_target - current_q
        # update
        self.q_table(state)[action] = current_q + self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        

class QLearner(Agent):
    """  Autonomous agent using Q-Learning. """

    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        
    def q_table(self, state):
        state = numpy.array2string(state)
        if state not in self.q_values:
            self.q_values[state] = numpy.zeros(self.nr_actions)
        return self.q_values[state]

    def policy(self, state):
        q_values = self.q_table(state)
        # return epsilon_greedy(q_values, None, epsilon=self.epsilon)
        action_count = numpy.zeros(self.nr_actions)
        return boltzmann(q_values, action_count)
        
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table(state)[action]
        td_target = reward + self.gamma * max(self.q_table(next_state)) # choose next action greedy (max, off policy)
        td_error = td_target - current_q
        # update
        self.q_table(state)[action] = current_q + self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) 
