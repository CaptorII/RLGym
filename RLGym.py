# Setup/imports
import gymnasium
from gymnasium import wrappers
import os
import numpy as np

wrapped_env = gymnasium.make("FrozenLake-v1", render_mode="rgb_array")  # create the environment used for the game
env = wrappers.RecordVideo(wrapped_env, 'game_screenshots')  # wrap environment in recorder to view output
if not os.path.exists('game_screenshots'):  # create directory for storing videos
    os.makedirs('game_screenshots')

# Define hyperparameters
number_of_runs = 10000  # takes about 3 seconds
learning_rate = 0.1
discount_factor = 0.99
initial_exploration = 1.0
min_exploration = 0.01
exploration_decay = 0.001
report_interval = 1000
report = 'Average: %.2f, 100-run average: %.2f, Best average: %.2f (Run %d)'

# Reset learned values, rewards and best streak
q_table = np.zeros((env.observation_space.n, env.action_space.n))  # stores learned values
rewards = []
best_streak = 0.0

# Start training
for run in range(number_of_runs):
    observation, info = env.reset()
    done = False
    run_reward = 0
    exploration_rate = max(min_exploration, initial_exploration * np.exp(
        -exploration_decay * run))  # decrease exploration rate every run
    while not done:
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()  # Take random actions
        else:
            action = np.argmax(q_table[observation, :])  # Take learned action

        new_observation, reward, terminated, truncated, _ = env.step(action)

        q_table[observation, action] = (1 - learning_rate) * q_table[observation, action] + learning_rate * \
                                       (reward + discount_factor * np.max(q_table[new_observation, :]))

        run_reward += reward
        observation = new_observation

        if (run + 1) % 100 == 0:  # check if last 100 run average was the best so far
            current_streak = np.mean(rewards[-100:])
            if current_streak > best_streak:
                best_streak = current_streak

        if terminated or truncated:
            done = True
            rewards.append(run_reward)
            if (run + 1) % report_interval == 0:  # every 500 runs, print a report showing progress
                print(report % (np.mean(rewards), np.mean(rewards[-100:]), best_streak, run + 1))
env.close()
