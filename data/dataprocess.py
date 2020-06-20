import sys
sys.path.append("..")  # 把上级目录加入到变量中
from game2048.game import Game
from game2048.displays import Display
from game2048.agents import TSAgent
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import time


class Data(TSAgent):
    def __init__(self, display=None, model_path='../model_related/checkpoint.ckpt'):
        game = Game()
        # default model path
        super().__init__(game, display, model_path=model_path)

    @staticmethod
    def board_rot(board, direction, rot=0):
        return np.rot90(board, -rot).copy(), (direction + rot) % 4

    @staticmethod
    def board_transpostion(board, direction):
        return board.T.copy(), direction ^ 3

    def generator(self, max_score=999999, filepath='', is_delete=True):
        data = []

        # csv_saver
        def csv_saver(path):
            with open(path, 'a') as f:
                np.savetxt(f, data, fmt='%d', delimiter=',')
            if is_delete:
                cycle_last_lines_with_delete(path, len(data))

        # save board&step
        def saver(direction, board_s):
            def mini_saver(log_board, dire):
                log_board = log_board.flatten()
                log_board = np.concatenate((np.array([dire]), log_board), axis=0)
                data.append(log_board)

            for i in range(4):
                b, d = Data.board_rot(board_s, direction, i)
                b[b == 0] = 1
                log_board_s = np.log2(b).astype(int)

                mini_saver(log_board_s, d)
                # # TODO test do no use data agumentation
                # break
                mini_saver(*Data.board_transpostion(log_board_s, d))

        def run():
            if self.game.end:  # win or lose
                return

            cnt_step = 0
            # end(self):0: continue, 1: lose, 2: win
            while not self.game.end and self.game.score <= max_score:
                step = self.step()
                # save best_step & current board
                saver(self.best_step(), self.game.board)
                # use ai to move
                self.game.move(step)
                cnt_step += 1
            return np.sum(self.game.board)

        self.game = Game(score_to_win=2 ** 15, random=False, enable_rewrite_board=True)
        score = run()

        path = "{}data_{}.csv".format(filepath, 'train')
        csv_saver(path)
        return score


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


def cycle_last_lines_with_delete(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[-count:] + a[:-2*count])
        fout.write(b)


if __name__ == '__main__':
    # 你睡觉觉它也觉觉
    sleep_time = 0
    while True:
        data = Data(display=None)
        data.generator()
        time.sleep(sleep_time)

