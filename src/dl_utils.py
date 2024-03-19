import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience:int=5, min_delta:float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.prev_loss = None
        self.counter = 0

    def __call__(self, val_loss : float):

        if self.prev_loss is None:
            self.prev_loss = val_loss
            return False

        if np.abs(self.prev_loss - val_loss) < self.min_delta:
            self.counter += 1
        else:
            self.counter = 0

        self.prev_loss = val_loss

        if self.counter >= self.patience:
            return True

        return False
    

def train(model : torch.nn.Module, optimizer : torch.optim,
          train_dataloader : DataLoader, val_dataloader : DataLoader,
          lr_scheduler : torch.optim.lr_scheduler = None,
          early_stopper : EarlyStopping = None,
          nepochs_max:int=100,
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    class_weight = torch.tensor([1.]*6+[.4]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    train_loss, val_loss = [], []

    for epoch in range(nepochs_max):
        bs_train_loss = 0
        bs_val_loss = 0
        train_bs, val_bs = 0, 0
        model.train()

        for (inp, target) in train_dataloader:
            inp = inp.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            bs_train_loss += loss.item()
            train_bs += 1

        train_loss.append(bs_train_loss / train_bs)

        with torch.no_grad():
            model.eval()
            for (inp, target) in val_dataloader:
                inp = inp.to(device)
                target = target.to(device)
                output = model(inp)
                loss = criterion(output, target)
                bs_val_loss += loss.item()
                val_bs += 1

            val_loss.append(bs_val_loss / val_bs)
        if (epoch+1) % 1 == 0:
            print(f'Epoch {epoch} : train loss {train_loss[-1]:.3f}, val loss {val_loss[-1]:.3f}')

        if lr_scheduler is not None:
            lr_scheduler.step()

        if early_stopper is not None:
            print('stopper count', early_stopper.counter)
            if early_stopper(val_loss[-1]):
                print(f'Early stopping at epoch {epoch}')
                break

    return train_loss, val_loss

def compute_test_loss(model, test_dataloader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    test_bs = 0
    model.eval()
    with torch.no_grad():
        for (inp, target) in test_dataloader:
            inp = inp.to(device)
            target = target.to(device)
            output = model(inp)
            loss = criterion(output, target)
            test_loss += loss.item()
            test_bs += 1

    return test_loss / test_bs

def plot_learning_curves(train_loss, val_loss):
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
