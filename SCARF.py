import torch
from torch import nn
import lightning as L

from lightning.pytorch.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report




from loss import nt_xent_loss
from SCARFDataset import SCARFDataset
from featureExtractor import FeatureExtractor
from supervisedClassifier import ClassificationHead




from preprocessing import readData


seed = 25
L.seed_everything(seed, workers=True)

class Encoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_hidden, dropout = 0.0):

        super().__init__()

        layers = []

        for _ in range(num_hidden - 1):

            # Add linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Add batch norm
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Add relu
            layers.append(nn.ReLU(inplace=True))
            #layers.append(nn.GELU())

            # Add dropout
            layers.append(nn.Dropout(dropout))

            # Update in_dim after first layer
            in_dim = hidden_dim


        layers.append(nn.Linear(in_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
    

    def forward(self, x):

        x = self.model(x)

        return x


class SCARFLightning(L.LightningModule):


    def __init__(self, in_dim, hidden_dim, num_hidden, head_hidden_dim, head_num_hidden, dropout = 0.0, corruption_rate = 0.6):
        super().__init__()

        # Load model
        self.main_encoder = Encoder(in_dim, hidden_dim, num_hidden, dropout)
        self.pretraining_head = Encoder(hidden_dim, head_hidden_dim, head_num_hidden, dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

        print(self.main_encoder)
        print(self.pretraining_head)
        print(self.decoder)


        # Set corruption rate, learning rate, and loss
        self.corruption_rate = corruption_rate
        #self.learning_rate = 1e-3
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5
        self.loss = nt_xent_loss(temperature=0.4)

        self.ir_loss = nn.MSELoss()

        # Uniform distribution over marginal distribution
        #self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))



    # Forward during inference
    def forward(self, x):
        
        x = self.main_encoder(x)
        #x = self.pretraining_head(x)

        return x

        


    def training_step(self, batch, batch_index):

        # Get batch size to create a mask of size batch_size


        x, y = batch  # x is the original embedding, y is the label (not used in pretraining)

        batch_size = x.size(0)
        
        mask = torch.rand_like(x) < self.corruption_rate 


        random_indices = torch.randint(0, batch_size, (batch_size, x.size(1)), device=x.device)  # Random indices for each feature and each sample in the batch
        x_random = torch.gather(x, dim=0, index=random_indices)  # Get values from the original batch at the random indices for each feature and each sample in the batch
        #x_random = self.marginals.sample(torch.Size((batch_size, ))).to(x.device) # Sample random values from the marginal distribution for each feature and each sample in the batch
        x_corrupted = torch.where(condition=mask, input=x_random, other=x)  # If mask is True, use random value, else use original value


        # Get latent representation of both the original embedding and the corrupted one
        embeddings = self.main_encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.main_encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        # Compute contrastive loss
        loss_contrastive = self.loss(embeddings, embeddings_corrupted)

        # Reconstruction branch
        x_reconstructed = self.decoder(embeddings_corrupted)
        loss_reconstruction = self.ir_loss(x_reconstructed, x)

        total_loss = loss_contrastive + 10*loss_reconstruction

        # Log
        self.log('CL_loss', loss_contrastive, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('IR_loss', loss_reconstruction, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return loss
        return total_loss
    

    def validation_step(self, batch, batch_index):
        x, y = batch  # x is the original embedding, y is the label (not used in pretraining)

        batch_size = x.size(0)
        
        
        mask = torch.rand_like(x) < self.corruption_rate 


        random_indices = torch.randint(0, batch_size, (batch_size, x.size(1)), device=x.device) 
        x_random = torch.gather(x, dim=0, index=random_indices) 
        x_corrupted = torch.where(condition=mask, input=x_random, other=x)

        
        embeddings = self.main_encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.main_encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        # Compute contrastive loss
        loss_contrastive = self.loss(embeddings, embeddings_corrupted)

        # Reconstruction branch
        x_reconstructed = self.decoder(embeddings_corrupted)
        loss_reconstruction = self.ir_loss(x_reconstructed, x)

        total_loss = loss_contrastive + 10*loss_reconstruction

        # Log
        self.log('validation_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return loss
        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)

        return optimizer

if __name__ == "__main__":
    print("This is the SCARF module. It contains the implementation of the SCARF algorithm for feature selection.")

    trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
    x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)

    train_dataset = SCARFDataset(x_train_processed, y_train)
    validation_dataset = SCARFDataset(x_val_processed, y_val)
    test_dataset = SCARFDataset(x_test_processed, y_test)


    # TRAINING
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create lightning model
    model = SCARFLightning(in_dim = train_dataset.shape[1], 
                        hidden_dim = 256, 
                        num_hidden = 4, 
                        head_hidden_dim = 256,  # 256 
                        head_num_hidden = 2, 
                        dropout = 0, 
                        corruption_rate = 0.2)


    print(model)


    # Create trainer and fit
    
    #early_stopping_callback = EarlyStopping(monitor="validation_loss", patience=3, mode="min")
    trainer = L.Trainer(max_epochs=100, accelerator='gpu', logger=True, enable_progress_bar=True) #callbacks=[early_stopping_callback])
    trainer.fit(model, train_loader, validation_loader)




    




        




