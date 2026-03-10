

from torch import nn
import lightning as L
import torch

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split


from preprocessing import readData



seed = 24
L.seed_everything(seed, workers=True)



class ClassificationModel(nn.Module):

    def __init__(self, input_dim=256, dropout=0.0):

        self.dropout = dropout
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),

            nn.Linear(256, 10),
        )


    def forward(self, x):

        x = self.model(x)

        return x


class ClassificationHead(L.LightningModule):

    def __init__(self, input_dim=256, dropout=0.0):
        
        super().__init__()

        self.model = ClassificationModel(input_dim=input_dim, dropout=dropout)

        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        self.loss = nn.CrossEntropyLoss()

    
    def forward(self, x):

        x = self.model(x)

        return x
    

    def training_step(self, batch, batch_index):
        
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_index):

        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("cl_validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_index):
        x, y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("cl_test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)

        return optimizer



def trainXGB(x_train_processed, y_train, x_test_processed, y_test):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed)
    xgb.fit(x_train_processed, y_train)

    y_pred_xgb = xgb.predict(x_test_processed)
    print("\n\n---------- XGBoost Classification Report ----------")
    print(classification_report(y_test, y_pred_xgb))




def trainRF(x_train_processed, y_train, x_test_processed, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(x_train_processed, y_train)

    y_pred_rf = rf.predict(x_test_processed)
    print("\n\n---------- Random Forest Classification Report ----------")
    print(classification_report(y_test, y_pred_rf))





if __name__ == "__main__":
    
    trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
    x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)




    label_proportion = 0.01
    x_train_small,_ , y_train_small, _ = train_test_split(x_train_processed, y_train, test_size=1-label_proportion, random_state=seed) 
    print("Small training set: ", x_train_small.shape, y_train_small.shape)

    label_proportion = 0.01
    x_val_small,_ , y_val_small, _ = train_test_split(x_val_processed, y_val, test_size=1-label_proportion, random_state=seed) 
    print("Small validation set: ", x_val_small.shape, y_val_small.shape)

    classifierType = "RF"



    x_train_small = torch.tensor(x_train_small, dtype=torch.float32)
    y_train_small = torch.tensor(y_train_small, dtype=torch.long)

    x_val_small = torch.tensor(x_val_small, dtype=torch.float32)
    y_val_small = torch.tensor(y_val_small, dtype=torch.long)

    x_test_processed = torch.tensor(x_test_processed, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset_embeddings = torch.utils.data.TensorDataset(x_train_small, y_train_small)
    validation_dataset_embeddings = torch.utils.data.TensorDataset(x_val_small, y_val_small)
    test_dataset_embeddings = torch.utils.data.TensorDataset(x_test_processed, y_test)

    train_dataloader_embeddings = torch.utils.data.DataLoader(
        train_dataset_embeddings,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    validation_dataloader_embeddings = torch.utils.data.DataLoader(
        validation_dataset_embeddings,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    test_dataloader_embeddings = torch.utils.data.DataLoader(
        test_dataset_embeddings,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    model = ClassificationHead(input_dim=186, dropout=0.0)

    #early_stopping_callback = EarlyStopping(monitor="cl_validation_loss", patience=3)
    trainer = L.Trainer(max_epochs=30, accelerator='gpu', logger=True, enable_progress_bar=True) #, callbacks=[early_stopping_callback])
    trainer.fit(model, train_dataloader_embeddings, validation_dataloader_embeddings)

    print("\n\n MLP")
    trainer.test(model, test_dataloader_embeddings)

    model = trainXGB(x_train_small, y_train_small, x_test_processed, y_test)

    model = trainRF(x_train_small, y_train_small, x_test_processed, y_test)



'''
MLP entrenada con un 1%: 0.65 - 0.67
RF y XGB entrenados con un 1%: 0.73


Habría que hacer test con varias ejecuciones y sacar métricas más realistas para la MLP (F1, métricas por clase,...)
Pero en general (ACC):

MLP 1% top of embeddings: 0.71-0.74 
XGB 1% top of embeddings: 0.72-0.73
RF 1% top of embeddings: 0.72-0.73

Las clases minoritarias siguen siendo un problema



SCARF-epoch=99-step=61700.ckpt: 
- batch size 256
- in_dim = train_dataset.shape[1], hidden_dim = 256, num_hidden = 4, head_hidden_dim = 256, head_num_hidden = 2, dropout = 0, corruption_rate = 0.2)
- 100 epoch

MLP 1% top of embeddings: 0.741 (verificar + posiblemente se pueda optimizar más los hiperparámetros)
XGB 1% top of embeddings: 0.72
RF 1% top of embeddings: 0.72


epoch=149-step=46350.ckpt
- GELU
- batch size 512
- in_dim = train_dataset.shape[1], hidden_dim = 256, num_hidden = 4, head_hidden_dim = 128, head_num_hidden = 2, dropout = 0, corruption_rate = 0.2)
- 150 epoch


MLP 1% top of embeddings: 0.729 (verificar + posiblemente se pueda optimizar más los hiperparámetros)
XGB 1% top of embeddings: 0.72-0.73
RF 1% top of embeddings: 0.72-0.73



Con fine-tuning 1% los resultados son peores, se queda normalmente entorno a 0.67 - 0.71









'''





