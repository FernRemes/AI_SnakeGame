import torch
import random
import numpy as np

from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plotty

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,123,3)
        self.trainer = QTrainer(self.model, learning_rate = LR, gamma = self.gamma) 

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            batch_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*batch_sample)
        self.trainer.train(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random actions
        self.epsilon = 100 - self.n_games
        final_action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_action[move] = 1
        else:
            prediction_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(prediction_state) 
            move = torch.argmax(prediction).item()
            final_action[move] = 1
        
        return final_action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()

    game = SnakeGame()

    while True:
        # get current state
        curr_state = agent.get_state(game)

        # get action
        final_action = agent.get_action(curr_state)

        # perform action
        reward, game_over, score = game.play_step(final_action)
        next_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(curr_state, final_action, reward, next_state, game_over)

        # remember
        agent.remember(curr_state, final_action, reward, next_state, game_over)

        if game_over:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_model()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            # plot results
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            plotty(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
        
