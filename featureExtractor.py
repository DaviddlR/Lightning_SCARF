import torch


class FeatureExtractor:

    def __init__(self, model):

        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()  # Set model to evaluation mode

    
    def extractFeatures(self, dataloader, labels=False):
        
        embeddings = list()
        
        # Extract embeddings and labels
        if labels:
            labels = list()

            with torch.no_grad():

                for x, y in dataloader:
                    x = x.to(self.device)
                    z = self.model(x)
                    embeddings.append(z.cpu())
                    labels.append(y)

        
            # Concatenate embeddings and labels
            embeddings = torch.cat(embeddings, dim=0)  # [num_embeddings, dimensions, 1, 1]
            embeddings = embeddings.flatten(start_dim=1)  # [num_embeddings, dimensions]
            labels = torch.cat(labels, dim=0)  # [num_labels]

            return embeddings, labels
    
        # Extract only embeddings
        else:

            with torch.no_grad():
                for x, y in dataloader:
                    z = self.model(x)
                    embeddings.append(z)

            embeddings = torch.cat(embeddings, dim=0)

            return embeddings