import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import random

class Bandit():
    def __init__(self, n_arms):
        self.n_arms = n_arms #number of arms in the bandit
        self.q_stars= [np.random.normal(0,1)for _ in range(n_arms)]# list of means of rewards of each arm
        self.stds = np.ones(n_arms) #standard deviation is 1 for all arms
        
    def pull_arm(self,arm_index): #Pull an arm to get a reward. This reward dist follows the textbook
        """Pull an arm and obtain the reward according to a normal distribution centred around the mean and having a std deviation 1"""
        reward = np.random.normal(self.q_stars[arm_index], self.stds[arm_index])
        return reward

    def print_bandit_info(self):
        """Gives us the info about each arm and its corresponding reward distribution"""
        for i in range(self.n_arms):
            print(f"Arm{i}: mean = {self.q_stars[i]}, stdd={self.stds[i]}")

    def visualize_reward_distributions(self, samples_per_arm=10000, save_path = "bandit_reward_distributions.svg"):
        """Visualize the reward distributions of all arms using violin plots."""
        data = []
        labels = []

        #Generate samples for each arm
        for arm_index in range(self.n_arms):
            rewards = np.random.normal(self.q_stars[arm_index],self.stds[arm_index],samples_per_arm)
            data.extend(rewards)
            labels.extend([f"Arm {arm_index}" for _ in range(samples_per_arm)])

        #create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=labels, y=data, inner="quartile", palette="gray")
        plt.title("Bandit Reward Distributions")
        plt.ylabel("Reward")
        plt.xlabel("Arms")
        plt.xticks(rotation=0)

        # Save as SVG
        plt.tight_layout()
        plt.savefig(save_path, format="svg")
        plt.show()
        print(f"Distribution plot saved to {save_path}")


class EpsilonGreedy():
    def __init__(self,epsilon):
        self.epsilon = epsilon

    def initialize(self, n_arms):
        self.values = [0.0 for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]

    def pick_arm(self):
        #exploit
        if random.random()>self.epsilon:
            return np.argmax(self.values)

        #explore
        else:
            return np.random.randint(len(self.values))

    def run_epsilon_greedy(self, n_steps, bandit):
        rewards = []
        for step in range(n_steps):

            arm = self.pick_arm()
            reward = bandit.pull_arm(arm)
            rewards.append(reward)#where am I adding the fact that"if arm i is pulled, reward corresponding to arm i is generated and others are zero"
            self.counts[arm] +=1
            #should i average out the reward? as in should i write Q = r1+r2+r3.../n
            self.values[arm]+=(reward-self.values[arm])/self.counts[arm]

            return rewards

def simulate(n_runs=2000, n_steps=1000, n_arms = 7, epsilons=[0.0,0.01,0.05,0.1, 0.5, 1]):
    avg_rewards = {eps: np.zeros(n_steps) for eps in epsilons} #dictionary storing the average rewards per run
    optimal_actions = {eps: np.zeros(n_steps) for eps in epsilons} #dictionary storing the optimal actions per run


    for eps in epsilons:
        for run in range(n_runs):
            bandit = Bandit(n_arms=n_arms)
            agent = EpsilonGreedy(epsilon=eps)
            agent.initialize(n_arms)

            best_arm = np.argmax(bandit.q_stars)#should it be q_stars or bandit.values?

            for step in range(n_steps):
                arm = agent.pick_arm()
                reward = bandit.pull_arm(arm)
                avg_rewards[eps][step]+=reward

                #track if arm pulled is optimal
                if arm == best_arm:
                    optimal_actions[eps][step]+=1

                agent.counts[arm]+=1
                agent.values[arm]+=(reward-agent.values[arm])/agent.counts[arm]

        avg_rewards[eps]/=n_runs
        optimal_actions[eps] = (optimal_actions[eps] / n_runs) * 100.0

    return avg_rewards, optimal_actions
     

    
if __name__=="__main__":
    
    bandit = Bandit(n_arms = 7)
    bandit.visualize_reward_distributions()
    avg_rewards, optimal_actions = simulate()

    # Plot Average Rewards
    plt.figure(figsize=(10, 6))
    for eps, rewards in avg_rewards.items():
        plt.plot(rewards, label=f"ε = {eps}")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Average reward vs steps")
    plt.legend()
    plt.savefig(fname="reward_convergence_plots.svg")
    plt.show()

    # Plot Optimal Action %
    plt.figure(figsize=(10, 6))
    for eps, optimal in optimal_actions.items():
        plt.plot(optimal, label=f"ε = {eps}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.title("Optimal action percentage vs steps")
    plt.legend()
    plt.savefig(fname="optimal_actions.svg")
    plt.show()
     
       
