import os
import time
import torch
from tqdm import tqdm

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(model, data_loader, verbose=False):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
#     print('----- Model Evaluation -----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(data_loader, total=len(data_loader.batch_sampler), desc="Testing"):
            # Forward pass.
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    if verbose:
        print(f'Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent


def train(model, criterion, data_loader, test_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model.""" 
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    
    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

#     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(enumerate(data_loader), total=len(data_loader.batch_sampler), desc="training"):
            # Forward pass.
            output = model(features.to(device))
            loss = criterion(output.to(device), target.to(device))
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01
            tock = time.time()
        acc = test(model, test_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print('Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds'.format(epoch+1, ema_loss, tock-tick),)
        torch.save(model.state_dict(), f'model_{epoch}.ckpt')
        print("Model Saved!")
        if os.path.isfile(f'model_{epoch-1}.ckpt'):
            os.remove(f'model_{epoch-1}.ckpt')
    return accs

