import torch
import torch.nn as nn
import torch.optim as optim

# Toy dataset
x_values = [i for i in range(11)]
x_train = torch.tensor(x_values, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor([2*i + 1 for i in x_values], dtype=torch.float)

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(1, 1)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    inputs = x_train
    labels = y_train

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished')
