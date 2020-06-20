import sys
sys.path.append("..")  # 把上级目录加入到变量中
import numpy as np
import torch


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class ExpectiIdiotAgent(Agent):

    def __init__(self, game, p_idiot=None, s_idiot=None, display=None):
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move
        self.p_idiot = p_idiot
        self.s_idiot = s_idiot

    def step(self, p_idiot=0, s_idiot=0):
        # ExpectiIdiotAgent has a probability of p_idiot to choose a random wrong direction
        if self.p_idiot is not None:
            p_idiot = self.p_idiot
        if self.s_idiot is not None:
            s_idiot = self.s_idiot

        while np.sum(self.game.board) < s_idiot:
            self.game.move(self.best_step())
        
        if np.random.random() < p_idiot * 4/3:
            direction = np.random.randint(0, 4)
            self.game.move(direction)

        direction = self.search_func(self.game.board)
        assert direction == self.best_step(), "not the best step."
        return direction

    def best_step(self):
        direction = self.search_func(self.game.board)
        return direction


class MyAgent(Agent):
    def __init__(self, game, display=None, model_path='../model_related/model_score_best.ckpt'):
        super().__init__(game, display)
        from model_related.model import ConvNet as model
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            assert False

    def step(self):
        from data.dataprocess import one_hot
        self.model.eval()
        with torch.no_grad():
            board = self.game.board
            board[board == 0] = 1
            board = np.log2(board)
            directions = torch.zeros(4, dtype=torch.int)
    
            for rot in range(4):
                b, d = board_rot(board, rot)
                d = d.to(self.device)
                b_torch = torch.tensor(b.copy()).long().unsqueeze(dim=0)  # copy()
                b_torch = one_hot(b_torch).float().to(self.device)
                outputs = self.model(b_torch.to(self.device))
                # print(outputs, d)
                tmp = outputs.data.squeeze().gather(dim=0, index=d)
                # print(tmp)
                tmp = torch.argmax(tmp)
                directions[tmp] += 1
    
                b_t, d_t = board_transpostion(b, d)
                b_t_torch = torch.tensor(b_t.copy()).long().unsqueeze(dim=0)  # copy()
                b_t_torch = one_hot(b_t_torch).float()
                outputs_t = self.model(b_t_torch.to(self.device))
                # print(outputs_t, d_t)
                tmp_t = outputs_t.data.squeeze().gather(dim=0, index=d_t).to(self.device)
                # print(tmp_t)
                tmp_t = torch.argmax(tmp_t)
                directions[tmp_t] += 1
    
            return torch.argmax(directions).item()


class TSAgent(MyAgent):
    def __init__(self, game, display=None, model_path='../model_related/checkpoint.ckpt'):
        # import Expectimax
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        from .expectimax import board_to_move
        self.search_func = board_to_move

        # import MyAgent
        super().__init__(game, display, model_path)

    # give the best direction of the situation of AI
    def best_step(self):
        return self.search_func(self.game.board)


def board_rot(board, rot, directions=torch.tensor([0, 1, 2, 3])):
    """
    :param board:
    :param rot:
    :param directions:
    :return: rot_board, directions
    """
    rot_board = np.rot90(board, -rot)
    rot_directions = (directions + rot) % 4  # copy()
    return rot_board, rot_directions


def board_transpostion(board, directions=torch.tensor([0, 1, 2, 3])):
    return board.T.copy(), directions ^ 3

