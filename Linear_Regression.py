import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(42)

# Data loading and preprocessing. It could be any format, text, video, audio, etc.

weight = 0.4
bias = 0.9
start = 0
end = 1
step = 0.04
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias
# print(X[:10], Y[:10])
# print(len(X), len(Y))

### Splitting data into training and test sets (one of the most important concept)

Train_split = int(0.7 * len(X))
X_train, Y_train = X[:Train_split], Y[:Train_split]
X_test, Y_test = X[Train_split:], Y[Train_split:]
print(len(X_train), len(Y_train), len(X_test), len(Y_test))


# Let´s visualize the split between training and test data.
def plot_data(X_train, Y_train, X_test, Y_test, Prediction=None):
    plt.figure(figsize=(10, 7))
    # convert tensors to numpy for safe plotting
    plt.scatter(X_train.detach().cpu().numpy(), Y_train.detach().cpu().numpy(), c="b", s=4, label="Training data")
    plt.scatter(X_test.detach().cpu().numpy(), Y_test.detach().cpu().numpy(), c="g", s=4, label="Testing data")
    if Prediction is not None:
        plt.scatter(X_test.detach().cpu().numpy(), Prediction.detach().cpu().numpy(), c="r", s=4, label="Prediction data")
    plt.legend()
    plt.show()

# plot_data(X_train, Y_train, X_test, Y_test, Prediction=None)

# Create a linear regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    # use correct type hint and the input 'x' (not the global X)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

model = LinearRegressionModel()
print(list(model.parameters()))

with torch.inference_mode():
    y_pred = model(X_test)
    print(y_pred)

# call plot_data with the computed prediction (was previously plot_data(y_pred))
# plot_data(X_train, Y_train, X_test, Y_test, Prediction=y_pred)

# Building training and testing loop
loss = nn.L1Loss() # MAE
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

epoch_count = []
loss_train_values = []  
loss_test_values = []

epoch = 335
for epoch in range(epoch):
    model.train() # put the model into training mode
    y_pred = model(X_train) 
    loss_train = loss(y_pred, Y_train) # compute the loss
    optimizer.zero_grad() # zero the gradients
    loss_train.backward() # backpropagation
    optimizer.step() # update the parameters

    #Testing loop
    model.eval() # put the model into evaluation mode
    with torch.inference_mode():
        test_pred = model(X_test)
        loss_test = loss(test_pred, Y_test)

    if epoch % 10 == 0:
     epoch_count.append(epoch)
     loss_train_values.append(loss_train)
     loss_test_values.append(loss_test)
     print(f"Epochs: {epoch}| Loss:  {loss_train}| Test Loss: {loss_test}")

  #  Print out model state_dict
    print(model.state_dict())
# Plotting loss curves
# plt.plot(epoch_count, loss_train_values, label="Train Loss")
y_preds = model(X_test)
plot_data(X_train, Y_train, X_test, Y_test, Prediction=y_preds)