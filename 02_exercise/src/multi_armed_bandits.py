import random
import numpy
import math


def random_bandit(Q_values, action_counts):
    return random.choice(range(len(Q_values)))


def epsilon_greedy(q_values, action_counts, epsilon=0.1):
    if numpy.random.rand() <= epsilon:
        return random.choice(range(len(q_values)))
    else:
        return numpy.argmax(q_values)


def boltzmann(q_values, action_counts, temperature=1.0):
    e = numpy.exp(q_values/temperature)
    return numpy.random.choice(range(len(action_counts)), p=e/sum(e))


def ucb1(q_values, action_counts, exploration_constant=1):
    ucb1_values = []
    n_total = sum(action_counts)
    for q, n in zip(q_values, action_counts):
        if n == 0:
            ucb1_values.append(math.inf)
        else:
            exploration_term = exploration_constant
            exploration_term *= numpy.sqrt(numpy.log(n_total)/n)
            ucb1_values.append(q + exploration_term)
    return numpy.argmax(ucb1_values)
    
