import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

weight_mat_epoch = torch.load('weight_mat_epoch.pth')

# Extract the weight matrix from the loaded dictionary
weight_mat = weight_mat_epoch['epoch_last']

class_wise_cosin_sim = np.zeros((10, 10))
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

for i in range(len(weight_mat)):
    for x in range(len(weight_mat)):
        class_wise_cosin_sim[i][x] = cos(weight_mat[i], weight_mat[x])

plt.figure(figsize=(10, 8))
sns.heatmap(class_wise_cosin_sim, annot=True, fmt=".2f", cmap="viridis")

plt.title("Class-wise Cosin_sim Heatmap_epoch_last")

# Save the plot as an image file
save_dir = 'heatmap/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, f'Class-wise Cosin_sim Heatmap_epoch_last.png'))
plt.close()  # Close the plot to release memory

print("heatmap visualizations saved.")