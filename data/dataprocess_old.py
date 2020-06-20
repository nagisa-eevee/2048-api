import sys
sys.path.append("..")  # 把上级目录加入到变量中
import game2048.game as gm
from game2048.game import Game
from game2048.displays import Display
from game2048.agents import ExpectiIdiotAgent, TSAgent
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


is_suspend = False


class Data(ExpectiIdiotAgent):
    """
    搜集一些 ExpectiMaxAgent 从初始情况产生的最优棋盘 & 随机的恶劣情况
    board先保存为log_board格式
    """
    def __init__(self, display=None):
        game = Game()
        super().__init__(game, display)

    @staticmethod
    def board_rot(board, direction, rot=0):
        return np.rot90(board, -rot).copy(), (direction + rot) % 4

    @staticmethod
    def board_transpostion(board, direction):
        return board.T.copy(), direction ^ 3

    def generator(self, epoches_n=0, epoches_r=0, p_idiot=0., s_idiot=0):
        data_s = []

        # csv_saver
        def csv_saver(path):
            with open(path, 'a') as f:
                np.savetxt(f, data_s, fmt='%d', delimiter=',')
            # # create data flow
            # cycle_last_lines(filename=path, count=len(data_s))

        # save board&step
        def saver(direction, board_s):
            def mini_saver(log_board, dire):
                log_board = log_board.flatten()
                log_board = np.concatenate((np.array([dire]), log_board), axis=0)
                data_s.append(log_board)

            # print(board_s)
            for i in range(4):
                b, d = Data.board_rot(board_s, direction, i)
                b[b == 0] = 1
                log_board_s = np.log2(b).astype(int)

                mini_saver(log_board_s, d)
                # # TODO test do no use data agumentation
                # break
                mini_saver(*Data.board_transpostion(log_board_s, d))

        def run(p_idiot=0., s_idiot=0):
            if self.game.end:  # win or lose
                return

            cnt_step = 0
            verbose_flag = True
            # end(self):0: continue, 1: lose, 2: win
            while not self.game.end and self.game.score <= 512:
                step = self.step(p_idiot, s_idiot)
                if verbose_flag:
                    print("p_idiot:{}, s_idiot:{}"
                          .format(p_idiot, s_idiot))
                    print("\ntotal_score:{}, board:\n{}"
                          .format(np.sum(self.game.board), self.game.board))
                    verbose_flag = False
                
                saver(step, self.game.board)
                self.game.move(step)

                cnt_step += 1
                if (cnt_step + 1) % 100 == 0:
                    print("cnt_steps: %d, scores: %d" % (cnt_step, self.game.score))
            print("final_steps:%d" % cnt_step)

        # normal
        for _ in range(epoches_n):
            # score_to_win 可以变 至少比model最大情形大一倍
            self.game = Game(score_to_win=2**11, random=False, enable_rewrite_board=True)
            run(p_idiot, s_idiot)

        # # random
        # if epoches_r != 0:
        #     rand_range = gm.rand_range
        #
        #     gm.rand_range = [0, 10]
        #     for _ in range(epoches_r):
        #         self.game = Game(score_to_win=2**12, random=True, enable_rewrite_board=True)
        #         run(p_idiot, s_idiot)
        #
        #     gm.rand_range = [4, 15]
        #     for _ in range(epoches_r):
        #         self.game = Game(score_to_win=2**16, random=True, enable_rewrite_board=True)
        #         run(p_idiot, s_idiot)
        #
        #     gm.rand_range = rand_range
        #     # assert False, "breakpoint"

        path = "data_{}.csv".format('eval_full')  # "train_below_", key
        print("saved to {}.".format(path))
        csv_saver(path)


def one_hot(board):
    """
    把4*4的板子转换成16位的one-hot(0-2^15)
    :parameter board: batch_size* 4 * 4, tensor
    :return one_hot:
    """
    return nn.functional.one_hot(board, 16).permute(dims=(0, 3, 1, 2))  # 0~2^15 one-hot


class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, nrows=128000):
        self.data = pd.read_csv(csv_path, nrows=nrows, header=None, sep=',', engine='python')
        self.labels = np.asarray(self.data.iloc[:, 0])

    def __getitem__(self, index):
        """
        :parameter index: index of data
        :return data_as_tensor: data, (4,4), torch.tensor(long?)
        :return single_data_label: label, (1,), int?
        """
        single_data_label = self.labels[index]
        # 读取所有值，并将 1D array ([16]) reshape 成为 2D array ([4,4])
        data_as_np = np.asarray(self.data.iloc[index][1:]).reshape(4, 4).astype(np.long)
        data_as_tensor = torch.tensor(data_as_np)
        return data_as_tensor, single_data_label

    def __len__(self):
        return len(self.data.index)


def delete_first_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[count:])
        fout.write(b)


def cycle_first_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[count:] + a[:count])
        fout.write(b)


def cycle_last_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[-count:] + a[:-count])
        fout.write(b)


if __name__ == '__main__':
    data = Data(display=None)
    # while True:
    for _ in range(50):
        data.generator(epoches_n=1, p_idiot=0., s_idiot=0)
