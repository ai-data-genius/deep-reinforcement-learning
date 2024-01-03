from torch import cat, from_numpy, log, multinomial, softmax, tensor
from torch.nn import Module, Linear
from torch.optim import Adam


class PolicyNetwork(Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = Linear(input_size, output_size)

    def forward(self, x):
        return softmax(self.fc(x), dim=-1)


class ReinforceWithBaseline:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state):
        state = from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = multinomial(probs, 1).item()
        return action

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            rewards = []
            done = False

            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                log_prob = log(self.policy(from_numpy(state).float())[action])
                log_probs.append(log_prob)
                rewards.append(reward)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns = tensor(returns)
            baseline = returns.mean()  # Calcul de la baseline moyenne
            returns = (returns - baseline) / (returns.std() + 1e-9)  # Normalisation avec la baseline

            policy_loss = []
            for log_prob, Gt in zip(log_probs, returns):
                policy_loss.append(-log_prob * Gt)

            self.optimizer.zero_grad()
            policy_loss = cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
