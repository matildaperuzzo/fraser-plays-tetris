import pygame
from enum import Enum
import torch

class Actions(Enum):
    NOTHING = 0
    RIGHT = 1
    LEFT = 2
    ROTATE = 3


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100


class Tetris:
    def __init__(self, width: int = 10, height: int = 10, ui: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define size of game board
        self.width = width
        self.height = height
        
        self.ui = ui
        if self.ui:  # setup ui properties
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)

            self.block_size = BLOCK_SIZE
            self.display = pygame.display.set_mode(
                (BLOCK_SIZE*self.width, BLOCK_SIZE*self.height)
            )
            pygame.display.set_caption('Tetris')
            self.clock = pygame.time.Clock()

        self.placedBlocks = torch.zeros((self.width, self.height), dtype=torch.int, device=self.device)
        self._reward = torch.tensor(0, dtype=torch.float, device=self.device)
        self.height_range = torch.arange(self.height, device=self.device)

        # define starting shape
        self._x = torch.tensor([self.width//2], dtype=torch.int, device=self.device)
        self._y = torch.tensor([self.height], dtype=torch.int, device=self.device)
        self.shape = {"x": self._x.clone(), "y": self._y.clone() }

        self.total_cleared_lines = 0
        self.reset()

    @property
    def reward(self) -> torch.Tensor:
        return self._reward
    
    @reward.setter
    def reward(self, value: float):
        self._reward.fill_(value)

    def reset(self):

        # init game state
        self.done = False
        self._new_shape()
        self.placedBlocks.mul_(0)

        # init game stats
        self.score = 0

    @property
    def state(self):
        state = self.placedBlocks.clone()
        state[self.x[self.shape_inview], self.y[self.shape_inview]] = 2
        return state

    def play_step(self, action: torch.Tensor):
        self.reward = 0  # reset reward at each step

        # 1. collect user input
        if self.ui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. move
        self._move(action)

        # 3. check if game over
        self._check_game_over()

        if self.ui:  # 4. update ui and clock
            self._update_ui()
            self.clock.tick(SPEED)

    def _update_ui(self):
        self.display.fill(BLACK)

        for idx in self.placedBlocks.argwhere():
            x, y = idx

            pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1)*BLOCK_SIZE + 4, 12, 12))

        for x,y in zip(self.shape["x"], self.shape["y"]):
            pygame.draw.rect(self.display, RED, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, RED2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1) * BLOCK_SIZE + 4, 12, 12))

        x, y = self.shape["x"][0], self.shape["y"][0]
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: torch.Tensor):

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # possible actions
        if action == Actions.RIGHT.value:
            if self.x.max() < self.width-1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    self.x += 1

        elif action == Actions.LEFT.value:
            if self.x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    self.x -= 1

        elif action == Actions.ROTATE.value:
            self.rotate_shape()
        
        # update shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # gravity
        if self.y.min() > 0:  # boundary check
            if not self.placedBlocks[x, y - 1].any():  # collision check
                self.y -= 1
            else:
                self._place_shape()
                self._clear_rows()
                self._new_shape()

        else:
            self._place_shape()
            self._clear_rows()
            self._new_shape()

    @property
    def shape_inview(self) -> torch.Tensor:
        """mask for shape points that are in view. This
        is required as shapes are initially created above the
        view and then fall into view"""
        return self.shape["y"] < self.height

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

    def _place_shape(self):

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # place shape
        self.placedBlocks[x, y] = 1

        # reward for filling out board horizontally
        self.reward += 1 if self.y.min() == 0 else -1

    def _clear_rows(self):
        isfull = self.placedBlocks.all(axis=0)
        num_full = sum(isfull)

        updatedBlocks = self.placedBlocks[:, ~isfull]

        self.placedBlocks.fill_(value=0)
        self.placedBlocks[:, :self.height-num_full] = updatedBlocks

        self.score += num_full
        self.total_cleared_lines += num_full

        # reward for clearing rows
        self.reward += torch.log10(num_full+1)
  
    def _new_shape(self):
        """Create a new shape above the view"""
        self.shape["x"] = self._x.clone()
        self.shape["y"] = self._y.clone()

        return self.shape

    def _check_game_over(self):
        """check if any of the placed blocks hit the top of the screen"""
        self.done = self.placedBlocks[:, -1].any()
        return self.done

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
