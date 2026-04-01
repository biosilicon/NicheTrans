from utils.utils_training_graph import evaluate, train_epoch


def train(model, criterion, optimizer, trainloader, use_img=True, device=None):
    del use_img
    return train_epoch(model, criterion, optimizer, trainloader, split="train", device=device)


def test(model, testloader, use_img=True, device=None, criterion=None):
    del use_img
    metrics = evaluate(model, testloader, criterion=criterion, split="test", task_type="regression", device=device)
    return metrics.get("pearson_mean")
