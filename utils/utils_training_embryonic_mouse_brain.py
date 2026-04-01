from utils.utils_training_graph import evaluate, train_epoch


def train_regression(model, criterion, optimizer, trainloader, device=None):
    return train_epoch(model, criterion, optimizer, trainloader, split="train", device=device)


def train_binary(model, criterion, optimizer, trainloader, device=None):
    return train_epoch(model, criterion, optimizer, trainloader, split="train", device=device)


def test_regression(model, testloader, if_sigmoid=False, device=None, criterion=None):
    del if_sigmoid
    metrics = evaluate(model, testloader, criterion=criterion, split="test", task_type="regression", device=device)
    return metrics.get("pearson_mean")


def test_binary(model, testloader, device=None, criterion=None):
    metrics = evaluate(model, testloader, criterion=criterion, split="test", task_type="binary", device=device)
    return metrics
