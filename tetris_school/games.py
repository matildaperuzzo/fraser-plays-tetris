from typing import Optional
from enum import Enum
import torch

import pygame
import gymnasium as gym
from gymnasium import spaces

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100


class Tetris(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    actions = Enum("Actions", "NOTHING RIGHT LEFT ROTATE", start=0)

    def __init__(self, width: int = 5, height: int = 5, render_mode: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # define size of game board
        self.width = width
        self.height = height

        # we have 4 actions, corresponding to "nothing", "right", "left", "rotate"
        self.action_space = spaces.Discrete(4)

        self.placedBlocks = torch.zeros((self.width, self.height), dtype=torch.int, device=self.device)
        self._reward = torch.tensor(0, dtype=torch.float, device=self.device)
        self.height_range = torch.arange(self.height, device=self.device)

        # define starting shape
        self._x = torch.tensor([self.width//2], dtype=torch.int, device=self.device)
        self._y = torch.tensor([self.height], dtype=torch.int, device=self.device)

        self.shape = {"x": self._x.clone(), "y": self._y.clone() }

        self.window = None
        self.clock = None

    def _get_obs(self):
        state = self.placedBlocks.clone()
        state[self.shape_inview] = 2
        return state.flatten()
    
    def _get_info(self):
        return {
            "shape": self.shape,
            "score": self.score,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # needed to seed self.np_random
        
        self._new_shape()
        self.placedBlocks.mul_(0)
        self.score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _new_shape(self):
        """Create a new shape above the view"""
        self.shape["x"], self.shape["y"] = self._x.clone(), self._y.clone()
        return self.shape
    
    @property
    def terminated(self) -> torch.Tensor:
        return self.placedBlocks[:, 1].any()
    
    @property
    def reward(self) -> torch.Tensor:
        return self._reward
    
    @reward.setter
    def reward(self, value: float):
        self._reward.fill_(value)

    def step(self, action: int):
        self.reward = 0  # type: ignore

        # 2. move
        x, y = self.move_shape(action)
        self.reward += 1 if not self.placedBlocks[x,0].any() else -1

        # gravity
        if self.y.min() > 0:  # boundary check
            if not self.placedBlocks[x, y - 1].any():  # collision check
                self.y -= 1

        # place shape if it hits the bottom or other blocks
        x, y = self.shape_inview
        if y.min() == 0 or self.placedBlocks[x, y - 1].any():
            self.placedBlocks[x, y] = 1
            self._clear_rows()
            self._new_shape()

        terminated = self.terminated
        reward = self.reward.clone()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def move_shape(self, action: int):
        x, y = self.shape_inview

        # possible actions
        if action == self.actions.RIGHT.value:
            if self.x.max() < self.width-1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    self.x += 1

        elif action == self.actions.LEFT.value:
            if self.x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    self.x -= 1

        elif action == self.actions.ROTATE.value:
            self.rotate_shape()
        
        return self.shape_inview
    
    def rotate_shape(self):
        """Rotate shape clockwise"""

        x_median = self.x.median()
        y_median = self.y.median()

        # rotate shape
        x_rotated = x_median - (self.y - y_median)
        y_rotated = y_median + (self.x - x_median)

        if x_rotated.max() < self.width-1 and x_rotated.min() > 0:  # boundary check
            # check for collisions
            if not self.placedBlocks[x_rotated[y_rotated < self.height], y_rotated[y_rotated < self.height]].any():

                self.x = x_rotated
                self.y = y_rotated

    @property
    def _inview(self) -> torch.Tensor:
        """mask for shape points that are in view. This
        is required as shapes are initially created above the
        view and then fall into view"""
        return self.shape["y"] < self.height

    @property
    def shape_inview(self) -> tuple:
        return self.x[self._inview], self.y[self._inview]

    @property
    def x(self) -> torch.Tensor:
        """x index of shape points"""
        return self.shape["x"]

    @x.setter
    def x(self, value: torch.Tensor):
        self.shape["x"] = value

    @property
    def y(self) -> torch.Tensor:
        """y index of shape points"""
        return self.shape["y"]

    @y.setter
    def y(self, value: torch.Tensor):
        self.shape["y"] = value

    def _clear_rows(self):
        isfull = self.placedBlocks.all(axis=0)
        num_full = sum(isfull)

        updatedBlocks = self.placedBlocks[:, ~isfull]
        self.placedBlocks.fill_(value=0)

        self.placedBlocks[:, :self.height-num_full] = updatedBlocks
        self.score += num_full

        # reward for clearing rows
        # self.reward += torch.log10(num_full+1)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()

            self.font = pygame.font.SysFont('arial', 25)            
            self.window = pygame.display.set_mode(
                (BLOCK_SIZE*self.width, BLOCK_SIZE*self.height)
            )
            pygame.display.set_caption('Tetris')

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((BLOCK_SIZE*self.width, BLOCK_SIZE*self.height))
        canvas.fill(BLACK)

        for idx in self.placedBlocks.argwhere():
            x, y = idx

            pygame.draw.rect(canvas, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1)*BLOCK_SIZE + 4, 12, 12))

        for x,y in zip(self.shape["x"], self.shape["y"]):
            pygame.draw.rect(canvas, RED, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, RED2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1) * BLOCK_SIZE + 4, 12, 12))

        x, y = self.shape["x"][0], self.shape["y"][0]
        pygame.draw.rect(canvas, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render(f"Score: {self.score}", True, WHITE)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(text, [0, 0])
        self.window.blit(canvas, canvas.get_rect())
        
        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()