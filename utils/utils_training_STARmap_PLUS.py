from utils.utils_training_graph import evaluate, train_epoch


def train(model, criterion, optimizer, trainloader, ct_information=False, device=None):
    del ct_information
    return train_epoch(model, criterion, optimizer, trainloader, split="train", device=device)


def test(model, testloader, ct_information=False, device=None, criterion=None):
    del ct_information
    return evaluate(model, testloader, criterion=criterion, split="test", task_type="binary", device=device)
