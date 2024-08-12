import matplotlib.pyplot as plt
import torch

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def display_image_pairs(dataset, num_pairs=5):
    """Display pairs of images, both from the same class and different classes."""
    same_class_pairs = []
    diff_class_pairs = []   
    for i in range(len(dataset)):
        img1, img2, label = dataset[i]  # unpacking three values
        if label == 1 and len(same_class_pairs) < num_pairs:
            same_class_pairs.append((img1, img2))
        elif label == 0 and len(diff_class_pairs) < num_pairs:
            diff_class_pairs.append((img1, img2))
        
        if len(same_class_pairs) >= num_pairs and len(diff_class_pairs) >= num_pairs:
            break

    # Plot same-class pairs
    print("Same Class Pairs:")
    plt.figure(figsize=(10, 2 * num_pairs))
    for i, (img1, img2) in enumerate(same_class_pairs):
        img1 = denormalize(img1, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        img2 = denormalize(img2, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(img1.permute(1, 2, 0).numpy())
        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(img2.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.show()

    # Plot different-class pairs
    print("Different Class Pairs:")
    plt.figure(figsize=(10, 2 * num_pairs))
    for i, (img1, img2) in enumerate(diff_class_pairs):
        img1 = denormalize(img1, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        img2 = denormalize(img2, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(img1.permute(1, 2, 0).numpy())
        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(img2.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.show()