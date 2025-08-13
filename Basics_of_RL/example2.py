import gymnasium as gym
import imageio


env = gym.make("LunarLander-v3", render_mode="rgb_array")
frames=[]


obs,info=env.reset()

for step in range(200):
    frames.append(env.render())
    obs,reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
    
env.close()

imageio.mimsave("lander.gif", frames, fps=30)
