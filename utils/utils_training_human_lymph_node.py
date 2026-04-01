from utils.utils_training_graph import evaluate, train_epoch


def train(model, criterion, optimizer, trainloader, device=None):
    return train_epoch(model, criterion, optimizer, trainloader, split="train", device=device)


def test(model, testloader, device=None, criterion=None):
    metrics = evaluate(model, testloader, criterion=criterion, split="test", task_type="regression", device=device)
    return metrics.get("pearson_mean")
