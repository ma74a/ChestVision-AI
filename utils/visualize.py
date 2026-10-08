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