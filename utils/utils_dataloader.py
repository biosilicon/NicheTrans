from datasets.data_loader import create_graph_loader


def sma_dataloader(args, dataset):
    trainloader = create_graph_loader(dataset.training, batch_size=args.train_batch, shuffle=True)
    testloader = create_graph_loader(dataset.testing, batch_size=args.test_batch, shuffle=False)
    return trainloader, testloader


def human_node_dataloader(args, dataset):
    trainloader = create_graph_loader(dataset.training, batch_size=args.train_batch, shuffle=True)
    testloader = create_graph_loader(dataset.testing, batch_size=args.test_batch, shuffle=False)
    return trainloader, testloader

def embryonic_mouse_brain(args, dataset):
    trainloader = create_graph_loader(dataset.training, batch_size=args.train_batch, shuffle=True)
    testloader = create_graph_loader(dataset.testing, batch_size=args.test_batch, shuffle=False)
    return trainloader, testloader

def breast_cancer_dataloader(args, dataset):
    trainloader = create_graph_loader(dataset.training, batch_size=args.train_batch, shuffle=True)
    testloader = create_graph_loader(dataset.testing, batch_size=args.test_batch, shuffle=False)
    return trainloader, testloader

def ad_mouse_dataloader(args, dataset, testing_control=False):
    del testing_control
    trainloader = create_graph_loader(dataset.training, batch_size=args.train_batch, shuffle=True)
    testloader = create_graph_loader(dataset.testing, batch_size=args.test_batch, shuffle=False)
    valloader = create_graph_loader(dataset.val, batch_size=args.test_batch, shuffle=False)

    return trainloader, testloader, valloader
