import sys
sys.path.append("..")  # 把上级目录加入到变量中
import torch
from model_related.model import ConvNet
from data.dataprocess import one_hot
import data.dataprocess
from game2048.agents import MyAgent
from game2048.game import Game
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 4

if __name__ == '__main__':
    # load dataset
    dataset_path = '../data/'
    dataset_name = dataset_path + 'data_train.csv'
    model = ConvNet(num_classes).to(device)

    # create model
    model_name = 'checkpoint_test.ckpt'
    torch.save(model.state_dict(), model_name)
    print('save model to {}.'.format(model_name))

    # random initialize data set
    print('data initalizing ...')
    datagen = data.dataprocess.Data(display=None, model_path=model_name)
    path = dataset_name

    score = 0
    # test net structure close to 180(random) is better.
    for _ in range(100):
        game = Game(size=4, score_to_win=2048)
        agent = MyAgent(game, display=None, model_path=model_name)
        agent.play()
        score += np.sum(game.board)
    print("Average total score @50 times is :{:.1f}".format(score/50))

