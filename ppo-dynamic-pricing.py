import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PriceEnvironment:
    """
    Simulated environment for dynamic pricing optimization.
    """
    def __init__(self, base_demand=100, price_elasticity=-1.5, competitor_prices=None, 
                 seasons=None, noise_level=0.1, max_steps=30):
        self.base_demand = base_demand
        self.price_elasticity = price_elasticity
        self.competitor_prices = competitor_prices or [10.0, 12.0, 9.5]
        self.seasons = seasons or np.sin(np.linspace(0, 2*np.pi, max_steps))
        self.noise_level = noise_level
        self.max_steps = max_steps
        self.current_step = 0
        self.current_price = 10.0
        self.price_history = []
        self.profit_history = []
        
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.current_price = 10.0
        self.price_history = []
        self.profit_history = []
        return self._get_state()
    
    def _get_demand(self, price):
        """Calculate demand based on price, competitor prices, and seasonal factors."""
        price_impact = (price / np.mean(self.competitor_prices)) ** self.price_elasticity
        seasonal_factor = 1.0 + 0.2 * self.seasons[self.current_step]
        noise = np.random.normal(1.0, self.noise_level)
        
        demand = self.base_demand * price_impact * seasonal_factor * noise
        return max(0, demand)
    
    def _get_state(self):
        """Return the current state observation."""
        # State includes: current price, competitor prices, seasonal factor, 
        # recent price history, recent profits
        state = [self.current_price]
        state.extend(self.competitor_prices)
        state.append(self.seasons[self.current_step])
        
        # Add price history (last 3 prices or zeros if not enough history)
        price_history = self.price_history[-3:] if len(self.price_history) > 0 else []
        while len(price_history) < 3:
            price_history.append(0.0)
        state.extend(price_history)
            
        # Add profit history (last 3 profits or zeros if not enough history)
        profit_history = self.profit_history[-3:] if len(self.profit_history) > 0 else []
        while len(profit_history) < 3:
            profit_history.append(0.0)
        state.extend(profit_history)
            
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        Action is interpreted as a price adjustment percentage.
        """
        # Convert action to price change percentage (-10% to +10%)
        price_change_pct = action * 0.1  # Scale to ±10%
        
        # Apply price change
        new_price = self.current_price * (1 + price_change_pct)
        new_price = max(5.0, min(20.0, new_price))  # Keep price in reasonable bounds
        
        # Calculate demand and profit
        demand = self._get_demand(new_price)
        cost = 5.0  # Simplified: fixed cost per unit
        profit = (new_price - cost) * demand
        
        # Update state
        self.price_history.append(self.current_price)
        self.profit_history.append(profit)
        self.current_price = new_price
        self.current_step += 1
        
        # Calculate reward (normalized profit)
        reward = profit / 1000.0  # Normalize reward scale
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {"profit": profit, "demand": demand}


class Actor(nn.Module):
    """
    Policy network for PPO algorithm.
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Mean and standard deviation of the action distribution
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.forward(state)
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Sample action
        action = normal.sample()
        
        # Calculate log probability of the action
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action.detach().numpy(), log_prob.detach().numpy()
    
    def evaluate(self, state, action):
        mean, std = self.forward(state)
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Calculate log probability of the action
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value network for PPO algorithm.
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class PPO:
    """
    Proximal Policy Optimization algorithm.
    """
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, 
                 clip_epsilon=0.2, k_epochs=10, batch_size=64):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory buffer for storing trajectories
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def select_action(self, state):
        """Select an action from the current policy."""
        action, log_prob = self.actor.get_action(state)
        
        # Store experience in memory
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action
    
    def store_transition(self, reward, is_terminal):
        """Store reward and terminal flag for the latest transition."""
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
    
    def update(self):
        """Update policy and value networks using PPO algorithm."""
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Calculate advantage
            values = self.critic(old_states)
            advantages = discounted_rewards - values.detach()
            
            # Get new log probabilities and entropy
            new_log_probs, entropy = self.actor.evaluate(old_states, old_actions)
            
            # Calculate PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Calculate critic loss
            critic_loss = nn.MSELoss()(values, discounted_rewards)
            
            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def save(self, filename):
        """Save model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """Load model parameters."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def train_ppo_agent():
    """Train a PPO agent on the dynamic pricing environment."""
    # Create environment
    env = PriceEnvironment()
    
    # Get dimensions
    state_dim = len(env.reset())
    action_dim = 1  # Price adjustment percentage
    
    # Create PPO agent
    agent = PPO(state_dim, action_dim)
    
    # Training parameters
    num_episodes = 200
    max_steps = 30
    
    # Logging
    episode_rewards = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action[0])
            
            # Store transition
            agent.store_transition(reward, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update agent
        agent.update()
        
        # Log performance
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Save model
    agent.save("ppo_pricing_model.pth")
    
    return agent, episode_rewards


def evaluate_ppo_agent(agent, num_episodes=10, visualize=False):
    """Evaluate trained PPO agent."""
    env = PriceEnvironment()
    total_profits = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_profit = 0
        prices = []
        demands = []
        
        while True:
            # Select action without storing in memory
            action, _ = agent.actor.get_action(state)
            
            # Take action
            next_state, _, done, info = env.step(action[0])
            
            # Update metrics
            prices.append(env.current_price)
            demands.append(info["demand"])
            episode_profit += info["profit"]
            
            # Update state
            state = next_state
            
            if done:
                break
        
        total_profits.append(episode_profit)
        
        print(f"Episode {episode+1}, Total Profit: ${episode_profit:.2f}")
        
    print(f"Average Profit over {num_episodes} episodes: ${np.mean(total_profits):.2f}")
    
    if visualize and num_episodes > 0:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot price and demand for the last episode
        plt.subplot(2, 1, 1)
        plt.plot(prices, label='Price')
        plt.xlabel('Step')
        plt.ylabel('Price ($)')
        plt.title('Price Adjustments')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(demands, label='Demand')
        plt.xlabel('Step')
        plt.ylabel('Units Demanded')
        plt.title('Customer Demand')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ppo_pricing_evaluation.png')
        
    return total_profits


if __name__ == "__main__":
    # Train agent
    print("Training PPO agent for dynamic pricing optimization...")
    agent, rewards = train_ppo_agent()
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    profits = evaluate_ppo_agent(agent, num_episodes=5, visualize=True)
    
    # Plot training curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('PPO Training Progress')
    plt.grid(True)
    plt.savefig('ppo_training_curve.png')
    
    print("Training and evaluation complete. Results saved to files.")
