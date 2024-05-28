import torch
from torch.utils.data import ConcatDataset, DataLoader
from MLP import PolyMod
from dataset import PeptideDataset
from matplotlib import pyplot as plt


def get_sampler(npos, nneg):

    weights = 0.5 / torch.concat([torch.tensor([npos] * npos), torch.tensor([nneg] * nneg)]).double()
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


batch_size = 64

# Create datasets
data_pos = PeptideDataset("data", True)
data_neg = PeptideDataset("data", False)
data_pos_train, data_pos_test = data_pos.split(0.9)
data_neg_train, data_neg_test = data_neg.split(0.9)
data_pos_train_size, data_neg_train_size = len(data_pos_train), len(data_neg_train)
data_train = ConcatDataset([data_pos_train, data_neg_train])
data_test = ConcatDataset([data_pos_test, data_neg_test])

# Create data loaders
sampler_train = get_sampler(len(data_pos_train), len(data_neg_train))
sampler_test = get_sampler(len(data_pos_test), len(data_neg_test))
train_dataloader = DataLoader(data_train, sampler=sampler_train, batch_size=batch_size)
test_dataloader = DataLoader(data_test, sampler=sampler_test)

# Create model
model = PolyMod()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=15)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = []
    train_loss_ma = []
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(loss.item())
        if len(train_loss) >= 100:
            train_loss_ma.append(sum(train_loss[-100:]) / 100)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    return train_loss_ma


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            cur_loss = loss_fn(pred, y).item()
            test_loss += cur_loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss


epochs = 15
train_loss = []
test_loss = []
test_loss_xs = []
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
    train_loss += train(train_dataloader, model, loss_fn, optimizer)
    test_loss.append(test(test_dataloader, model, loss_fn))
    test_loss_xs.append(len(train_loss))
    scheduler.step()
print("Done!")

# plotting code
plt.plot(train_loss, label="Train")
plt.plot(test_loss_xs, test_loss, label="Test")
plt.scatter(test_loss_xs, torch.full((len(test_loss_xs),), fill_value=min(train_loss)), marker='|', label="Epochs")
plt.xlabel(f"Batches seen (batch size = {batch_size})")
plt.ylabel("Cross-Entropy loss")
plt.title("Train and Test loss")
plt.legend()
plt.show()
