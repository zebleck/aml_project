import numpy as np

N = 2
FREE = 0
SNAKE_HEAD = 1
SNAKE_SEGMENT = 2
APPLE = 3

LEFT = 4
RIGHT = 6
UP = 8
DOWN = 2

class Snake:
    def __init__(self):
        self.board = np.zeros((N, N))
        self.apple_location = None
        self.snake_segment_locations = []

        self.snake_head_location = np.random.randint((N, N))
        self.board[tuple(self.snake_head_location)] = SNAKE_HEAD
        self.place_apple()
        self.board[tuple(self.apple_location)] = APPLE

        self.snake_direction = -1
        self.score = 0
        self.finished = False

    def update_board(self):
        self.board = np.zeros((N, N))
        self.board[tuple(self.apple_location)] = APPLE
        self.board[tuple(self.snake_head_location)] = SNAKE_HEAD

        if len(self.snake_segment_locations) > 0:
            idx0 = np.array(self.snake_segment_locations)[:, 0]
            idx1 = np.array(self.snake_segment_locations)[:, 1]
            self.board[(idx0, idx1)] = SNAKE_SEGMENT

    def place_apple(self):
        free_tiles = np.asarray(np.where(self.board == 0)).T
        if len(free_tiles) == 0:
            self.finished = True
            return
        idx = np.random.randint(free_tiles.shape[0])
        self.apple_location = free_tiles[idx]

    def run_step(self, direction):
        new_snake_head_location = self.snake_head_location.copy()
        if direction == UP:
            new_snake_head_location[0] -= 1
        elif direction == DOWN:
            new_snake_head_location[0] += 1
        elif direction == LEFT:
            new_snake_head_location[1] -= 1
        elif direction == RIGHT:
            new_snake_head_location[1] += 1
        else:
            raise Exception("no valid direction")

        if (not (0 <= new_snake_head_location[0] < N)) or (not (0 <= new_snake_head_location[1] < N)):
            self.finished = True
            return

        self.snake_segment_locations.insert(0, tuple(self.snake_head_location))
        self.snake_head_location = new_snake_head_location.copy()

        if np.all(self.snake_head_location == self.apple_location):
            self.place_apple()
            self.score += 1
        else:
            self.snake_segment_locations.pop()

        self.update_board()
        if np.sum(self.board == FREE) + np.sum(self.board == APPLE) == 0:
            self.finished = True
        if tuple(self.snake_head_location) in self.snake_segment_locations:
            self.finished = True

    def run_game_visual(self):
        print(self.board)
        while not self.finished:
            direction = int(input())
            while direction not in [UP, DOWN, LEFT, RIGHT]:
                direction = int(input())
            self.run_step(direction)
            print(self.score)
            print(self.board)
        print("Game Over, final score:", self.score)

    def run_game(self, algorithm):
        while not self.finished:
            direction = algorithm(self.board)
            self.run_step(direction)
        print("Game Over, final score:", self.score)

if __name__ == "__main__":
    game = Snake()
    game.run_game_visual()