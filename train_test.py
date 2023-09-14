import torch
import time


def local_training(model, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency):
    i = 1
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        correct = 0
        running_time = 0.0
        model.train()
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            init_time = time.time()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute running accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            running_time += (time.time() - init_time)
            if i % log_frequency == 0:  # print every X mini-batches
                print(f' step: {i}, loss: {running_loss / log_frequency:.3f}, '
                      f'accuracy: {100* correct / total:.3f}%, time: {running_time / log_frequency:.3f}')
                running_loss = 0.0
                running_time = 0.0
                total = 0
                correct = 0

            i += 1

        test(model, testloader, device)


def federated_training(model, communicator, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency,
                       local_steps=3):
    i = 1
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        correct = 0
        running_time = 0.0
        model.train()
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            init_time = time.time()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute running accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            running_time += (time.time() - init_time)
            if i % log_frequency == 0:  # print every X mini-batches
                print(f' step: {i}, loss: {running_loss / log_frequency:.3f}, '
                      f'accuracy: {100 * correct / total:.3f}%, time: {running_time / log_frequency:.3f}')
                running_loss = 0.0
                total = 0
                correct = 0
                running_time = 0.0

            # perform FedAvg/D-SGD every K steps
            if i % local_steps == 0:
                comm_time = communicator.communicate(model)

            i += 1

        test(model, testloader, device)


def test(model, test_dl, device, test_batches=30, epoch=False):
    correct = 0
    total = 0
    i = 1
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # only sample a few batches if doing random sample test
            if i == test_batches and not epoch:
                break

            i += 1

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total: .3f}%')
