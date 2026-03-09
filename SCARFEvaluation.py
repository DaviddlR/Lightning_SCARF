import lightning as L

from featureExtractor import FeatureExtractor
from supervisedClassifier import ClassificationHead

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import torch

from SCARF import SCARFLightning
from SCARFDataset import SCARFDataset
from preprocessing import readData



# Load data
trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)

train_dataset = SCARFDataset(x_train_processed, y_train)
validation_dataset = SCARFDataset(x_val_processed, y_val)
test_dataset = SCARFDataset(x_test_processed, y_test)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
saved_checkpoint = SCARFLightning.load_from_checkpoint("checkpoints/SCARF-epoch=74-step=92475.ckpt", in_dim = train_dataset.shape[1], hidden_dim = 256, num_hidden = 4, head_hidden_dim = 256, head_num_hidden = 2, dropout = 0, corruption_rate = 0.3)
saved_checkpoint.freeze()

# Classification on top of the embeddings
extractor = FeatureExtractor(saved_checkpoint)
train_embeddings, train_labels = extractor.extractFeatures(train_loader, labels=True)
validation_embeddings, validation_labels = extractor.extractFeatures(validation_loader, labels=True)
test_embeddings, test_labels = extractor.extractFeatures(test_loader, labels=True)

print(train_embeddings.shape)
print(validation_embeddings.shape)
print(test_embeddings.shape)
print(train_labels.shape)
print(validation_labels.shape)
print(test_labels.shape)


doitsmall = True

if doitsmall:
    label_proportion = 0.01
    train_embeddings,_ , train_labels, _ = train_test_split(train_embeddings, train_labels, test_size=1-label_proportion, random_state=1492) 

    label_proportion = 0.01
    validation_embeddings,_ , validation_labels, _ = train_test_split(validation_embeddings, validation_labels, test_size=1-label_proportion, random_state=1492) 


print("Training set: ", train_embeddings.shape, train_labels.shape)
print("Validation set: ", validation_embeddings.shape, validation_labels.shape)


train_dataset_embeddings = torch.utils.data.TensorDataset(train_embeddings, train_labels)
validation_dataset_embeddings = torch.utils.data.TensorDataset(validation_embeddings, validation_labels)
test_dataset_embeddings = torch.utils.data.TensorDataset(test_embeddings, test_labels)

train_dataloader_embeddings = torch.utils.data.DataLoader(
    train_dataset_embeddings,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

validation_dataloader_embeddings = torch.utils.data.DataLoader(
    validation_dataset_embeddings,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_dataloader_embeddings = torch.utils.data.DataLoader(
    test_dataset_embeddings,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# MLP CLASSIFICATION ON TOP OF EMBEDDINGS
classification_head = ClassificationHead(dropout=0.0)

#early_stopping_callback = EarlyStopping(monitor="cl_validation_loss", patience=3)
#trainer = L.Trainer(max_epochs=200, accelerator='gpu', logger=True, enable_progress_bar=True, callbacks=[early_stopping_callback])
trainer = L.Trainer(max_epochs=50, accelerator='gpu', logger=True, enable_progress_bar=True)
trainer.fit(classification_head, train_dataloader_embeddings, validation_dataloader_embeddings)

print("\n\nTEST CLASSIFICATION HEAD")
trainer.test(classification_head, test_dataloader_embeddings)
print("\n\n")

# XGBOOST CLASSIFICATION ON TOP OF EMBEDDINGS
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=1492)
xgb.fit(train_embeddings, train_labels)

y_pred_xgb = xgb.predict(test_embeddings)
print("\n\n---------- XGBoost Classification Report ----------")
print(classification_report(test_labels, y_pred_xgb))

# Random Forest CLASSIFICATION ON TOP OF EMBEDDINGS
rf = RandomForestClassifier(n_estimators=100, random_state=1492)
rf.fit(train_embeddings, train_labels)

y_pred_rf = rf.predict(test_embeddings)
print("\n\n---------- Random Forest Classification Report ----------")
print(classification_report(test_labels, y_pred_rf))




