from torch import cat, from_numpy, log, multinomial, softmax, tensor
from torch.nn import Module, Linear, MSELoss
from torch.optim import Adam


class PolicyNetwork(Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = Linear(input_size, output_size)

    def forward(self, x):
        return softmax(self.fc(x), dim=-1)


class CriticNetwork(Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        self.fc = Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


class ReinforceWithLearnedBaseline:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.critic = CriticNetwork(env.observation_space.shape[0])
        self.optimizer_policy = Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.critic_loss = MSELoss()

    def choose_action(self, state):
        state = from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = multinomial(probs, 1).item()

        return action

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            done = False

            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                log_prob = log(self.policy(from_numpy(state).float())[action])
                log_probs.append(log_prob)
                values.append(self.critic(from_numpy(state).float()))
                rewards.append(reward)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = tensor(returns)

            policy_loss = []
            value_loss = []
            for log_prob, value, Gt in zip(log_probs, values, returns):
                advantage = Gt - value.item()
                policy_loss.append(-log_prob * advantage)
                value_loss.append(self.critic_loss(value, tensor([Gt])))

            self.optimizer_policy.zero_grad()
            self.optimizer_critic.zero_grad()
            policy_loss = cat(policy_loss).sum()
            value_loss = cat(value_loss).mean()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_critic.step()