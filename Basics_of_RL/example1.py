import gymnasium as gym

#Create environment
env = gym.make("LunarLander-v3")

#Sample Action
sample_action = env.action_space.sample()
print("Sample action :", sample_action)

#Sample Observation
sample_observation = env.observation_space.sample()
print("Sample observation :", sample_observation)