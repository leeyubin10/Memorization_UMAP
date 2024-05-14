import matplotlib.pyplot as plt

def visualize_dataset(train_dataset, num_classes, random_label_counts):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(train_dataset.targets, bins=num_classes, rwidth=0.8)
    plt.title('Overall Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(range(num_classes))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x_values = range(num_classes)
    y_values = [random_label_counts[label] for label in range(num_classes)]  # Calculate the total count of randomly labeled data for each class
    plt.bar(x_values, y_values)
    plt.title('Randomly Labeled Data Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(range(num_classes))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('label_distributions.png')
    plt.show()