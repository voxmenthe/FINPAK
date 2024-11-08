# Training loop
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()

  outputs = model(inputs)  # Shape: (batch_size, num_bins)
  # Assuming we are predicting for one target period
  targets = targets.squeeze()  # Shape: (batch_size,)
  loss = loss_fn(outputs, targets)
  loss.backward()
  optimizer.step()