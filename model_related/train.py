import sys
sys.path.append("..")  # 把上级目录加入到变量中
import torch
import torch.nn as nn
from model_related.model import ConvNet
from model_related.eval import eval as evaluate
from model_related.logger import Logger
from data.dataprocess import one_hot
from data.dataprocess import DatasetFromCSV
import data.dataprocess
import time
from tqdm import tqdm


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 4
batch_size = 1024
num_epochs = 3000
learning_rate = 0.009
total_step = 300
sleep_time = 0
limit = batch_size * total_step * 3


def update_times(total_score):
    # update about 10% data
    # data augmentation 8x
    # int(limit * 0.10 / 8 / (total_score/3))
    return int(limit / total_score * 0.375 * 0.10 + 1)


if __name__ == '__main__':
    eval_accuracy_best = 25.
    eval_accuracy = 25.
    train_accuracy = 25.
    total_score_best = 1856.2
    total_score = 1200
    times = update_times(total_score)

    # load dataset
    dataset_path = '../data/'
    dataset_name = dataset_path + 'data_train.csv'
    model = ConvNet(num_classes).to(device)
    
    model_name = 'model_score_best.ckpt'
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
        print('load model from {}.'.format(model_name))
    except FileNotFoundError:
        model_name = 'checkpoint.ckpt'
        torch.save(model.state_dict(), model_name)
        print('save model to {}.'.format(model_name))

    # random initialize data set
    print('data initalizing ...')
    datagen = data.dataprocess.Data(display=None, model_path=model_name)
    # empty dataset
    with open(dataset_name, 'w'):
        pass
    # create new dataset
    while True:
        datagen.generator(filepath=dataset_path, is_delete=False)
        # check if there is enough data
        with open(dataset_name, 'r') as f:
            a = f.readlines()
            print("data collecting... {}/{}".format(len(a), limit))
            if len(a) >= limit:
                break

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    # Train the model
    for epoch in range(0, num_epochs):
        train_dataset = DatasetFromCSV(dataset_name, nrows=limit)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        logger = Logger('./logs')
        total_accuracy = 0
        for step, (boards, labels) in tqdm(enumerate(train_loader)):
            if step >= total_step:
                break

            one_hot_boards = one_hot(boards).float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(one_hot_boards)
            loss = criterion(outputs, labels)

            # Predict
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (labels == predicted.squeeze()).float().mean().detach()
            total_accuracy += accuracy

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item(), accuracy.item()*100))

        if (epoch + 1) % 1 == 0:
            filename = 'checkpoint.ckpt'
            torch.save(model.state_dict(), filename)
            print("saved to {}.".format(filename))

        train_accuracy = total_accuracy/total_step*100
        print('total_accuracy: {:.2f}'.format(train_accuracy))

        eval_accuracy = evaluate(eval_model=model)
        if eval_accuracy > eval_accuracy_best:
            eval_accuracy_best = eval_accuracy
            torch.save(model.state_dict(), 'model_eval_best.ckpt')
            print("saved to model_eval_best.ckpt")
        print('eval_accuracy/best:{:.2f}/{:.2f}.'
              .format(eval_accuracy, eval_accuracy_best))

        print("wait for new data ...")
        score = 0
        datagen = data.dataprocess.Data(display=None, model_path='checkpoint.ckpt')
        for _ in tqdm(range(times)):
            score += datagen.generator(filepath='../data/')
        total_score = score/times
        print("Average total scores @{} times is {:.1f}.".format(times, total_score))
        times = update_times(total_score)
        
        if total_score > total_score_best:
            total_score_best = total_score
            torch.save(model.state_dict(), 'model_score_best.ckpt')
            print("saved to model_score_best.ckpt")

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'train_accuracy': train_accuracy,
                'eval_accuracy': eval_accuracy,
                'average_total_score:': score/times}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, (epoch + 1) * total_step)

        time.sleep(sleep_time)

    torch.save(model.state_dict(), 'model_last.ckpt')
    print("saved to model_last.ckpt")

