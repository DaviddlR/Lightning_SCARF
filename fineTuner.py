import torch
from torch import nn
import lightning as L

from sklearn.model_selection import train_test_split


from SCARFDataset import SCARFDataset
from SCARF import SCARFLightning
from preprocessing import readData


seed = 20
L.seed_everything(seed, workers=True)



class FineTuner(L.LightningModule):


    def __init__(self, scarf_model):
        
        super().__init__()

        self.encoder = scarf_model.main_encoder


        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_FT_train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_FT_validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("cl_FT_test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_FT_test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-4}
        ])





# Load data
trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)

doitsmall = True

if doitsmall:
    label_proportion = 0.01
    train_embeddings,_ , train_labels, _ = train_test_split(x_train_processed, y_train, test_size=1-label_proportion, random_state=seed) 

    label_proportion = 0.01
    validation_embeddings,_ , validation_labels, _ = train_test_split(x_val_processed, y_val, test_size=1-label_proportion, random_state=seed) 


print("Training set: ", train_embeddings.shape, train_labels.shape)
print("Validation set: ", validation_embeddings.shape, validation_labels.shape)

train_dataset = SCARFDataset(train_embeddings, train_labels)
validation_dataset = SCARFDataset(validation_embeddings, validation_labels)
test_dataset = SCARFDataset(x_test_processed, y_test)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)






# Load model
saved_checkpoint = SCARFLightning.load_from_checkpoint("lightning_logs/version_29/checkpoints/epoch=99-step=61700.ckpt", in_dim = train_dataset.shape[1], hidden_dim = 256, num_hidden = 4, head_hidden_dim = 256, head_num_hidden = 2, dropout = 0, corruption_rate = 0.2)
#saved_checkpoint.freeze()


fine_tuner = FineTuner(saved_checkpoint)

trainer = L.Trainer(max_epochs=100, accelerator='gpu', logger=True, enable_progress_bar=True)
trainer.fit(fine_tuner, train_loader, validation_loader)

print("\n\nTEST CLASSIFICATION HEAD")
trainer.test(fine_tuner, test_loader)
print("\n\n")








