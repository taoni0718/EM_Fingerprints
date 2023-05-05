# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_epochs = 1000
learning_rate = 0.001
noise_std = 0.1

# num_train_samples = 5000
# num_val_samples = 1000
# num_test_samples = 500

# Generate random dataset
# train_data = torch.rand(num_train_samples, 1, 64, 64).to(device)
# val_data = torch.rand(num_val_samples, 1, 64, 64).to(device)
# test_data = torch.rand(num_test_samples, 1, 64, 64).to(device)

train_data = torch.tensor(train_data_seg, dtype=torch.float32).unsqueeze(1).to(device)
val_data = torch.tensor(val_data_seg, dtype=torch.float32).unsqueeze(1).to(device)
test_data = torch.tensor(test_data_seg, dtype=torch.float32).unsqueeze(1).to(device)


# Initialize the model, loss function, and optimizer
model = DiffusionModel(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     for i in range(0, num_train_samples, batch_size):
#         data = train_data[i:i + batch_size]
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, data)
#         loss.backward()
#         optimizer.step()

#     # Validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for i in range(0, num_val_samples, batch_size):
#             data = val_data[i:i + batch_size]
#             outputs = model(data)
#             loss = criterion(outputs, data)
#             val_loss += loss.item()

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / (num_val_samples // batch_size):.4f}")

# train_losses = []
# val_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     epoch_train_loss = 0
#     for i in range(0, len(train_data), batch_size):
#         optimizer.zero_grad()
#         batch_data = train_data[i:i + batch_size]
#         denoised_batch_data = model.denoise(batch_data, noise_std)

#         loss = criterion(denoised_batch_data, batch_data)
#         loss.backward()
#         optimizer.step()
#         epoch_train_loss += loss.item()

#     train_losses.append(epoch_train_loss / len(train_data))

#     model.eval()
#     epoch_val_loss = 0
#     with torch.no_grad():
#         for i in range(0, len(val_data), batch_size):
#             batch_data = val_data[i:i + batch_size]
#             denoised_batch_data = model.denoise(batch_data, noise_std)

#             loss = criterion(denoised_batch_data, batch_data)
#             epoch_val_loss += loss.item()

#     val_losses.append(epoch_val_loss / len(val_data))

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

noisy_data = train_noisy_images.to(device)
clean_data = train_clean_images.to(device)
val_noisy_data = val_noisy_images.to(device)
val_clean_data = val_clean_images.to(device)
test_noisy_data = test_noisy_images.to(device)
test_clean_data = test_clean_images.to(device)
    
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for i in range(0, len(clean_data), batch_size):
        optimizer.zero_grad()
        clean_batch_data = clean_data[i:i + batch_size]
        noisy_batch_data = noisy_data[i:i + batch_size]
        denoised_batch_data = model.reverse(noisy_batch_data)

        loss = criterion(denoised_batch_data, clean_batch_data)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(clean_data))

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_clean_data), batch_size):
            clean_batch_data = val_clean_data[i:i + batch_size]
            noisy_batch_data = val_noisy_data[i:i + batch_size]
            denoised_batch_data = model.reverse(noisy_batch_data)

            loss = criterion(denoised_batch_data, clean_batch_data)
            epoch_val_loss += loss.item()

    val_losses.append(epoch_val_loss / len(val_clean_data))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')