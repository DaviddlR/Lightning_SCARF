


import torch
from torch import nn
from torch.nn import functional as F


# (de la implementación de SimCLR)

'''
z_i and z_j Batch of augmented views from same images [batch_size, latent_dimensions]
temperature is a hyperparameter that controls the scale of the similarities. A lower temperature makes the model focus more on hard negatives, while a higher temperature makes it focus more on easy negatives.
'''
class nt_xent_loss(nn.Module):

    def __init__(self, temperature = 1.0):
        super().__init__()

        self.temperature = temperature


    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]

        # Concatenate views
        z = torch.cat([z_i, z_j], dim=0)  # Batch dimension

        # Normalization
        z = F.normalize(z, dim=1)  # Latent space dimension

        # Similarity matrix
        sim_matrix = torch.matmul(z, z.T) # When normalizing, dot product is equivalent to cosine similarity, so we can use matrix multiplication to compute the similarity between all pairs of embeddings in the batch. The resulting sim_matrix will have shape [2 * batch_size, 2 * batch_size], where sim_matrix[i, j] is the similarity between the i-th and j-th embeddings in the concatenated batch.

        # Temperature
        sim_matrix = sim_matrix / self.temperature

        # Mask to avoid self comparisons
        mask = torch.eye(2 * current_batch_size, dtype=torch.bool, device=z.device)

        # Extract positive pairs indexes
        positive_zi = torch.arange(current_batch_size, 2*current_batch_size)
        positive_zj = torch.arange(0, current_batch_size)
        positive_index = torch.cat([positive_zi, positive_zj]).to(z.device)


        # Positive pairs
        pos_pairs = torch.exp(sim_matrix[torch.arange(2*current_batch_size), positive_index])  # [2 * batch_size]
        
        # Similarities between each embedding and others (except himself)
        sim_matrix = torch.exp(sim_matrix) * (~mask)  # Remove diagonal = self similarity (which is always 1)
        sum_similarity = sim_matrix.sum(dim=1)  


        loss = -torch.log(pos_pairs / sum_similarity)
        loss = loss.mean()


        return loss





