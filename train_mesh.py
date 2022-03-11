
"""
Skeleton code for a pytorch model training function
"""
from audio2mesh import Audio2mesh
import torch
def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, num_epochs=None):
    l2loss = torch.nn.MSELoss().to(device)
    l1loss = torch.nn.L1Loss().to(device)
    while global_epoch < num_epochs:
    running_loss = 0.
    progress_bar = tqdm(enumerate(train_data_loader))

    for step, (audio_spectogram, label) in progress_bar:

        geometricloss = l2loss(vertices_mod, vertices_real)
        model.train()
        optimizer.zero_grad()
        geometricloss.backward()
        optimizer.step()


        '''
        TODO:
        1. Code the forward pass of the model,
        2. Code for the predicted vertex deformation to reference the vertices, Vˆt = Vr + δt,
        3. Code for the loss function mentioned in the paper
        Note: For debugging, you can use a random tensor of the correct shape for reference vertices, spectograms and other data variables
        '''



if __name__ == "__main__":

    device = torch.device("cuda" if use_cuda else "cpu")
    
