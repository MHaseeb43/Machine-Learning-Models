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
plot_data(X_train, Y_train, X_test, Y_test, Prediction=y_pred)