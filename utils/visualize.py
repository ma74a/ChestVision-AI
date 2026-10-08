import torch
import random
import matplotlib.pyplot as plt

def visualize_data(dataset):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax in axes.flat:

        idx = random.randint(0, len(dataset) - 1)

        image, labels = dataset[idx]
        row = dataset.df.iloc[idx]

        # CHW -> HWC
        image_np = image.permute(1, 2, 0).numpy()

        diseases = [
            disease
            for disease, value in zip(dataset.classes, labels)
            if value == 1
        ]

        ax.imshow(image_np)
        ax.set_title(
            f"{row['Image Index']}\n"
            + ", ".join(diseases)
            if diseases
            else f"{row['Image Index']}\nNo Finding"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """(C, H, W) normalized tensor -> (C, H, W) in [0, 1]."""
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def show_batch(loader, classes, n_cols=4, max_images=16):
    imgs, labels = next(iter(loader))
    n = min(len(imgs), max_images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))
    axes = axes.flatten()

    for i in range(n):
        img = denormalize(imgs[i]).permute(1, 2, 0).numpy()  # (H, W, C)
        active = [classes[j] for j in torch.where(labels[i] == 1)[0].tolist()]
        title = "\n".join(active) if active else "No Finding"

        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    for ax in axes[n:]:  # hide unused cells
        ax.axis("off")

    plt.tight_layout()
    plt.show()

