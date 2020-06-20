import sys
sys.path.append("..")  # 把上级目录加入到变量中
import torch
from model_related.model import ConvNet as Model
from data.dataprocess import one_hot
from data.dataprocess import DatasetFromCSV
from tqdm import tqdm


def eval(eval_model=None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    batch_size = 1024

    # load dataset
    path = '../data/data_eval.csv'
    if eval_model is None:
        path = '../data/data_eval_full.csv'

    # type is numpy.ndarray
    test_dataset = DatasetFromCSV(path)

    # Data loader
    # data type is torch.long
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    if eval_model is None:
        model = Model().to(device)
        eval_filename = 'checkpoint.ckpt'
        try:
            model.load_state_dict(torch.load(eval_filename, map_location=device))
        except FileNotFoundError:
            assert False, "FileNotFoundError,{} not found.".format(eval_filename)
    else:
        model = eval_model

    # Test the model
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0

        for i, (boards, labels) in tqdm(enumerate(test_loader)):
            boards = one_hot(boards).float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(boards)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            eval_accuracy = correct / total * 100

        if eval_model is None:
            print('eval_accuracy: {:.4f}'.format(eval_accuracy))

    return eval_accuracy


if __name__ == '__main__':
    eval()
