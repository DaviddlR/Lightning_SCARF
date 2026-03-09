

from torch import nn
import lightning as L
import torch

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split


from preprocessing import readData
    



class ClassificationModel(nn.Module):

    def __init__(self, dropout=0.0):

        self.dropout = dropout
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),

            nn.Linear(256, 256),
        )


    def forward(self, x):

        x = self.model(x)

        return x


class ClassificationHead(L.LightningModule):

    def __init__(self, dropout=0.0):
        
        super().__init__()

        self.model = ClassificationModel(dropout=dropout)

        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        
        self.loss = nn.CrossEntropyLoss()

    
    def forward(self, x):

        x = self.model(x)

        return x
    

    def training_step(self, batch, batch_index):
        
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_index):

        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_validation_loss", loss)

        return loss
    
    def test_step(self, batch, batch_index):
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("cl_test_loss", loss)
        self.log("cl_test_ACC", acc)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)

        return optimizer



def trainXGB(x_train_processed, y_train, x_test_processed, y_test):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=1492)
    xgb.fit(x_train_processed, y_train)

    y_pred_xgb = xgb.predict(x_test_processed)
    print("\n\n---------- XGBoost Classification Report ----------")
    print(classification_report(y_test, y_pred_xgb))




def trainRF(x_train_processed, y_train, x_test_processed, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=1492)
    rf.fit(x_train_processed, y_train)

    y_pred_rf = rf.predict(x_test_processed)
    print("\n\n---------- Random Forest Classification Report ----------")
    print(classification_report(y_test, y_pred_rf))





if __name__ == "__main__":
    
    trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
    x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)




    label_proportion = 0.01
    x_train_small,_ , y_train_small, _ = train_test_split(x_train_processed, y_train, test_size=1-label_proportion, random_state=1492) 
    print("Small training set: ", x_train_small.shape, y_train_small.shape)

    classifierType = "XGB"


    if classifierType == "MLP":

        train_dataset_embeddings = torch.utils.data.TensorDataset(x_train_processed, y_train)
        validation_dataset_embeddings = torch.utils.data.TensorDataset(x_val_processed, y_val)
        test_dataset_embeddings = torch.utils.data.TensorDataset(x_test_processed, y_test)

        train_dataloader_embeddings = torch.utils.data.DataLoader(
            train_dataset_embeddings,
            batch_size=128,
            shuffle=True,
            num_workers=0
        )

        validation_dataloader_embeddings = torch.utils.data.DataLoader(
            validation_dataset_embeddings,
            batch_size=128,
            shuffle=True,
            num_workers=0
        )

        test_dataloader_embeddings = torch.utils.data.DataLoader(
            test_dataset_embeddings,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )


        model = ClassificationHead(dropout=0.0)

        early_stopping_callback = EarlyStopping(monitor="cl_validation_loss", patience=3)
        trainer = L.Trainer(max_epochs=200, accelerator='gpu', logger=True, enable_progress_bar=True, callbacks=[early_stopping_callback])
        trainer.fit(model, train_dataloader_embeddings, validation_dataloader_embeddings)

        print("\n\n MLP")
        trainer.test(model, test_dataloader_embeddings)
        print("\n\n")
    
    elif classifierType == "XGB":
        print("\n\n XGB")
        model = trainXGB(x_train_small, y_train_small, x_test_processed, y_test)
        print("\n\n")
    elif classifierType == "RF":
        print("\n\n RF")
        model = trainRF(x_train_small, y_train_small, x_test_processed, y_test)
        print("\n\n")
    else:
        print("Invalid classifier type. Please choose from MLP, XGB, or RF.")







