import argparse
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from tetris_school.games import Tetris
from tetris_school.model import Fraser
from helper import plot


def train(learning_rate: float = 0.001, temperature: float = 1.0, ui: bool = True, file: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_scores = []
    plot_mean_scores = []
    n_games = 0
    total_score = 0
    record = 0

    game = Tetris(ui=ui)
    model = Fraser(hidden_size=32, layer_number=6, num_actions=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    while True:

        # sample action from model
        output = model(game.state)
        log_probs = F.log_softmax( output["logits"] / temperature, dim=-1)
        action = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(dim=-1)

        # perform action and calculate loss
        game.play_step(action)
        loss = -log_probs[action] * game.reward

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if game.done:
            n_games += 1
            print(f'Game {n_games} Score {game.score} Record {record}')

            plot_scores.append(game.score)
            total_score += game.score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)
            game.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a tetris agent")
    parser.add_argument("--file", type=str, help="File to load model from")
    parser.add_argument("--ui", action="store_true", help="Use the UI")

    train(**vars(parser.parse_args()))
