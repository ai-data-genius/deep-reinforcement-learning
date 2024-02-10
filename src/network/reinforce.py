from torch import cat, from_numpy, log, multinomial, softmax, tensor
from torch.nn import Module, Linear
from torch.optim import Adam


class PolicyNetwork(Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = Linear(input_size, output_size)

    def forward(self, x):
        return softmax(self.fc(x), dim=-1)


class Reinforce:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state):
        state = from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = multinomial(probs, 1).item()

        # return False if random() >= 1 - (i * .05) else action
        return action

    def train_chose_action(self):
        pass

    def predict(self, state: list, chose_action_state: list, actions: list): # -> Union[bool, action]
        #fn(chose_action_state) => keep_playing: bool  # prédit si on doit continuer de jouer
        #keep_playing == True && fn(state) => action to play  # prédit l'action à jouer
        pass

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            rewards = []
            done = False

            self.train_chose_action()

            # i = 0
            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)

                """
                [
                    confiance: 100, est_tombé: 0, nb_avancement_fait_avant_de_tomber: 0,
                ]
                [
                    confiance: 100, est_tombé: 1, nb_avancement_fait_avant_de_tomber: 5,
                ]
                [
                    confiance: 95, est_tombé: 0, nb_avancement_fait_avant_de_tomber: 0,
                ]
                [
                    confiance: 95, est_tombé: 1, nb_avancement_fait_avant_de_tomber: 7,
                ]
                [
                    confiance: 88, est_tombé: 0, nb_avancement_fait_avant_de_tomber: 0,
                ]
                """

                """
                [
                    way_id:0, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    way_id:1, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5,
                    way_id:2, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:3, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 9, 2, 9,
                    way_id:4, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:5, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5,
                    way_id:6, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    roll_dices: 0, 0, 0, 0, 0, 0,
                ]
                [
                    way_id:0, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    way_id:1, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5, 
                    way_id:2, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:3, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 9, # reward: 9 - 7 = 2 
                    way_id:4, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:5, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5,
                    way_id:6, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    roll_dices: 2, 4, 5, 3, 4, 2,
                ]
                [
                    way_id:0, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    way_id:1, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5, 
                    way_id:2, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:3, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 9, # reward: 9 - 3 = 6 
                    way_id:4, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:5, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5,
                    way_id:6, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    roll_dices: 2, 4, 5, 3, 4, 2,
                ]
                [
                    way_id:0, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    way_id:1, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5, 
                    way_id:2, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:3, is_won:1, won_id_player:1, player_id_distance_to_the_end:1, 0, 2, 9, # reward: 9 - 0 = 9
                    way_id:4, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 7, 2, 7,
                    way_id:5, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 5, 2, 5,
                    way_id:6, is_won:0, won_id_player:0, player_id_distance_to_the_end:1, 3, 2, 3,
                    roll_dices: 2, 4, 5, 3, 4, 2,
                ]
                """

                log_prob = log(self.policy(from_numpy(state).float())[action])
                log_probs.append(log_prob)
                rewards.append(reward)

            returns = []
            G = 0
            for r in reversed(rewards):  # [9,6,2]
                G = r + self.gamma * G  # 9 + (.99 * 0)  # 6 + (.99 * 9) = 14.91
                returns.insert(0, G)

            returns = tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalisation

            policy_loss = []
            for log_prob, Gt in zip(log_probs, returns):
                policy_loss.append(-log_prob * Gt)

            self.optimizer.zero_grad()
            policy_loss = cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
