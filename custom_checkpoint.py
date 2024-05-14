import torch
from tsne_model import CustomResNet

# List of checkpoint files to remove unexpected keys from
checkpoint_files = ['resnet18_epoch_0.pth', 'resnet18_best.pth', 'resnet18_last.pth']

# Iterate over each checkpoint file
for checkpoint_file in checkpoint_files:
    # Define the model
    model = CustomResNet()

    # Load the state dictionary of the model
    state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    # Check for unexpected keys
    unexpected_keys = [key for key in state_dict.keys() if 'resnet.fc' in key]
    print("Unexpected keys in {}: {}".format(checkpoint_file, unexpected_keys))

    # Remove unexpected keys
    for key in unexpected_keys:
        del state_dict[key]

    # Update the model's state dictionary
    model.load_state_dict(state_dict)

    # Generate a new checkpoint file name
    new_checkpoint_file = 'new_' + checkpoint_file

    # Save the new checkpoint file
    torch.save(model.state_dict(), new_checkpoint_file)

    print("New checkpoint saved as:", new_checkpoint_file)