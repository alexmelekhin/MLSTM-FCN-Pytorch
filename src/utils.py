import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    for inputs, labels, seq_lens in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs, seq_lens)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, validloader, criterion, optimizer, 
          epochs=10, print_every=10, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))

    valid_loss_min = np.Inf # track change in validation loss
    steps = 0
    
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels, seq_lens in trainloader:
            steps += 1

            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss/print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss/len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy/len(validloader)*100))
                
                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                    torch.save(model.state_dict(), 'weights/'+run_name+'.pt')
                    valid_loss_min = valid_loss

                train_loss = 0

                model.train()


def load_datasets(dataset_name='ISLD'):
    data_path = './datasets/'+dataset_name+'/'

    X_train = torch.load(data_path+'X_train_tensor.pt')
    X_val = torch.load(data_path+'X_val_tensor.pt')
    X_test = torch.load(data_path+'X_test_tensor.pt')

    y_train = torch.load(data_path+'y_train_tensor.pt')
    y_val = torch.load(data_path+'y_val_tensor.pt')
    y_test = torch.load(data_path+'y_test_tensor.pt')

    seq_lens_train = torch.load(data_path+'seq_lens_train_tensor.pt')
    seq_lens_val = torch.load(data_path+'seq_lens_val_tensor.pt')
    seq_lens_test = torch.load(data_path+'seq_lens_test_tensor.pt')

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_lens_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_lens_test)

    return train_dataset, val_dataset, test_dataset

