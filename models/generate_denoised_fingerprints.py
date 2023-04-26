import matplotlib.pyplot as plt

# Implement the testing function
def test_model(model, clean_images, noisy_images):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(20, 3, figsize=(20, 40))

        for i in range(20):
            original_image = clean_images[i].squeeze().cpu().numpy()
            noisy_image = noisy_images[i].squeeze().cpu().numpy()

            # Reverse process
            denoised_image = model.reverse(noisy_images[i:i + 1].to(device))

            # Convert tensors to numpy arrays and move them to CPU for plotting
            denoised_image = denoised_image.squeeze().cpu().numpy()

            # Plot the original, noisy, and denoised images
            axes[i, 0].imshow(original_image, cmap='gray')
            axes[i, 0].set_title(f'Original Image {i + 1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(noisy_image, cmap='gray')
            axes[i, 1].set_title(f'Noisy Image {i + 1}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(denoised_image, cmap='gray')
            axes[i, 2].set_title(f'Denoised Image {i + 1}')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

# Test the trained model
test_model(model, test_clean_images, test_noisy_images)