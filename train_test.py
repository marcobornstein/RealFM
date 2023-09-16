from mpi4py import MPI
import torch
import time


def local_training(model, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency, recorder):
    i = 1
    final_accuracy = None
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
            num_examples = labels.size(0)
            num_correct = (predicted == labels).sum().item()
            total += num_examples
            correct += num_correct

            # print statistics
            loss_val = loss.item()
            running_loss += loss_val
            comp_time = time.time() - init_time
            running_time += comp_time
            recorder.add_new(comp_time, 0, num_correct/num_examples, loss_val)

            if i % log_frequency == 0:  # print every X mini-batches
                print(f' [rank {recorder.rank}] step: {i}, loss: {running_loss / log_frequency:.3f}, '
                      f'accuracy: {100* correct / total:.3f}%, time: {running_time / log_frequency:.3f}')
                running_loss = 0.0
                running_time = 0.0
                total = 0
                correct = 0
                recorder.save_to_file()

            i += 1

        # spit out the final accuracy after training
        if epoch == epochs:
            final_accuracy = test(model, testloader, device, recorder, return_acc=True)
        else:
            test(model, testloader, device, recorder)

        MPI.COMM_WORLD.Barrier()

    return final_accuracy


def federated_training(model, communicator, trainloader, testloader, device, loss_fn, optimizer, epochs, log_frequency,
                       recorder, local_steps=3):
    i = 1
    final_accuracy = None
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
            num_examples = labels.size(0)
            num_correct = (predicted == labels).sum().item()
            total += num_examples
            correct += num_correct

            # print statistics
            loss_val = loss.item()
            running_loss += loss_val
            comp_time = time.time() - init_time
            running_time += comp_time

            # perform FedAvg/D-SGD every K steps
            if i % local_steps == 0:
                comm_time = communicator.communicate(model)
            else:
                comm_time = 0

            recorder.add_new(comp_time, comm_time, num_correct / num_examples, loss_val, local=False)

            if i % log_frequency == 0:  # print every X mini-batches
                print(f' [rank {recorder.rank}] step: {i}, loss: {running_loss / log_frequency:.3f}, '
                      f'accuracy: {100 * correct / total:.3f}%, time: {running_time / log_frequency:.3f}')
                running_loss = 0.0
                total = 0
                correct = 0
                running_time = 0.0
                recorder.save_to_file(local=False)

            i += 1

        # spit out the final accuracy after training
        communicator.sync_models(model)
        if epoch == epochs:
            # ensure models are synced so that final test accuracies are all equivalent
            final_accuracy = test(model, testloader, device, recorder, return_acc=True, local=False)
        else:
            test(model, testloader, device, recorder, local=False)

        MPI.COMM_WORLD.Barrier()

    return final_accuracy


def federated_training_nonuniform(model, communicator, trainloader, testloader, device, loss_fn, optimizer, steps_per_e,
                                  epochs, log_frequency, recorder, local_steps=6):
    i = 1
    total_steps = steps_per_e * epochs
    while True:
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
            num_examples = labels.size(0)
            num_correct = (predicted == labels).sum().item()
            total += num_examples
            correct += num_correct

            # print statistics
            loss_val = loss.item()
            running_loss += loss_val
            comp_time = time.time() - init_time
            running_time += comp_time

            # perform FedAvg/D-SGD every K steps
            if i % local_steps == 0:
                comm_time = communicator.communicate(model)
            else:
                comm_time = 0

            recorder.add_new(comp_time, comm_time, num_correct / num_examples, loss_val, local=False)

            if i % log_frequency == 0:  # print every X mini-batches
                print(f' [rank {recorder.rank}] step: {i}, loss: {running_loss / log_frequency:.3f}, '
                      f'accuracy: {100 * correct / total:.3f}%, time: {running_time / log_frequency:.3f}')
                running_loss = 0.0
                total = 0
                correct = 0
                running_time = 0.0
                recorder.save_to_file(local=False)

            if i % steps_per_e == 0:
                communicator.sync_models(model)
                if i % total_steps == 0:
                    # spit out the final accuracy after training
                    final_accuracy = test(model, testloader, device, recorder, return_acc=True, local=False)
                    return final_accuracy
                else:
                    test(model, testloader, device, recorder, local=False)

            i += 1


def test(model, test_dl, device, recorder, test_batches=30, epoch=True, return_acc=False, local=True):
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

    test_acc = correct / total
    recorder.add_test_accuracy(test_acc, epoch=epoch, local=local)
    print(f'[rank {recorder.rank}] test accuracy on {total} images: {100 * test_acc: .3f}%')
    if return_acc:
        return test_acc
