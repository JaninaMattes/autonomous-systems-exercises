import rooms
import agent as a
import matplotlib.pyplot as plot


def episode(env, ag, params, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = ag.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        ag.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward*(params["gamma"]**time_step)
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


parameters = {}
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
parameters["nr_actions"] = env.action_space.n
# parameters["simulations"] = 100  # only relevant for planning
# parameters["horizon"] = 50  # only relevant for planning
# parameters["env"] = env  # only relevant for planning
parameters["alpha"] = 0.3  # learning rate
parameters["gamma"] = 0.99  # discount factor
parameters["epsilon"] = 1.0  # exploration rate (at the start of learning)
parameters["epsilon_min"] = 0.01  # minimal exploration rate
parameters["epsilon_decay"] = 0.0001  # epsilon decay
parameters["training_episodes"] = 500  # training duration

# agent = a.RandomAgent(parameters)
#### TASK 1-2 ####
agent = a.SARSALearner(parameters)
# agent = a.QLearner(parameters) 
#################

returns = [episode(env, agent, parameters, ep) for ep in range(parameters["training_episodes"])]

x = range(parameters["training_episodes"])
y = returns

plot.plot(x, y)
plot.title("SARSA, alpha=0.3")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

# env.save_video()
