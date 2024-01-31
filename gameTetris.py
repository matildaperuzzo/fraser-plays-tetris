import pygame
from enum import Enum
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)


class Actions(Enum):
    NOTHING = 0
    RIGHT = 1
    LEFT = 2
    ROTATE = 3


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100


class TetrisAI:
    def __init__(self, width: int = 6, height: int = 30):

        # define size of game board
        self.width = width
        self.height = height

        # setup ui properties
        self.block_size = BLOCK_SIZE
        self.display = pygame.display.set_mode(
            (BLOCK_SIZE*self.width, BLOCK_SIZE*self.height)
        )

        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()

        self.reset()
        self.n_cleared_lines = 0

    def reset(self):
        # init game state
        self._new_shape()

        self.score = 0
        self.placedBlocks = np.zeros((self.width, self.height), dtype=np.int8)

        self.frame_iteration = 0
        self._update_ui()

    @property
    def state(self):
        state = self.placedBlocks.copy()
        state[self.x[self.shape_inview], self.y[self.shape_inview]] = 2
        return state

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)

        # 3. check if game over
        reward = 0
        game_over = False
        if self._check_game_over():
            game_over = True
            reward = -1

            return reward, game_over, self.score

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(BLACK)

        for x, y in zip(*np.where(self.placedBlocks)):
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1)*BLOCK_SIZE + 4, 12, 12))

        for point in self.shape:
            x, y = point
            pygame.draw.rect(self.display, RED, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1) * BLOCK_SIZE + 4, 12, 12))

        x, y = self.shape[0]
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: Actions):

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # possible actions
        if np.array_equal(action, [1, 0, 0, 0]):
            if self.x.max() < self.width-1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    self.x += 1

        elif np.array_equal(action, [0, 1, 0, 0]):
            if self.x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    self.x -= 1

        elif np.array_equal(action, [0, 0, 1, 0]):
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
    def shape_inview(self) -> np.ndarray:
        """mask for shape points that are in view. This
        is required as shapes are initially created above the
        view and then fall into view"""
        return self.shape["y"] < self.height

    @property
    def x(self) -> np.ndarray:
        """x index of shape points"""
        return self.shape["x"]

    @x.setter
    def x(self, value: int):
        self.shape["x"] = value

    @property
    def y(self) -> np.ndarray:
        """y index of shape points"""
        return self.shape["y"]

    @y.setter
    def y(self, value: int):
        self.shape["y"] = value

    def _place_shape(self):

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # place shape
        self.placedBlocks[x, y] = 1

    def _clear_rows(self):
        updatedBlocks = np.zeros((self.width, self.height), dtype=np.int8)

        isfull = self.placedBlocks.all(axis=0)
        updatedBlocks[:, :self.height-sum(isfull)] = np.delete(self.placedBlocks, np.where(isfull), axis=1)

        self.placedBlocks = updatedBlocks

    def _new_shape(self):
        """Create a new shape above the view"""
        self.shape = np.array([
                (self.width//2, self.height+1),
                (self.width//2+1, self.height+1),
                (self.width//2-1, self.height+1),
                (self.width//2, self.height)
            ],
            dtype=[('x', np.int8), ('y', np.int8)]
        )
        return self.shape

    def _check_game_over(self):
        """check if any of the placed blocks hit the top of the screen"""
        return self.placedBlocks[:, -1].any()

    def rotate_shape(self):
        """Rotate shape clockwise"""

        x_median = np.median(self.x).astype(int)
        y_median = np.median(self.y).astype(int)

        x_rel = self.x - x_median
        y_rel = self.y - y_median

        # rotate shape
        x_rotated = x_median - y_rel
        y_rotated = y_median + x_rel

        # bounce back if rotated shape is out of bounds
        if x_rotated.max() >= self.width:
            x_rotated -= 1
        elif x_rotated.min() < 0:
            x_rotated += 1

        if y_rotated.max() >= self.height:
            y_rotated -= 1
        elif y_rotated.min() < 0:
            y_rotated += 1

        # check for collisions
        if not self.placedBlocks[x_rotated[self.shape_inview], y_rotated[self.shape_inview]].any():

            self.x = x_rotated
            self.y = y_rotated
