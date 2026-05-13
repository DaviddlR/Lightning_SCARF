

from torch import nn
import lightning as L
import torch
from torch.nn import functional as F

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


from preprocessing import readData
from loss import FocalLoss

import numpy as np



# seed = 26
# L.seed_everything(seed, workers=True)



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
        weights = torch.tensor([15.0, 15.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.5, 5.0, 15.0])
        self.loss = FocalLoss(alpha=weights.to(self.device), gamma=2.0, reduction='mean')  # Focal loss
        #self.loss = nn.CrossEntropyLoss(weight=weights.to(self.device))  # Cross-entropy loss with class weights
        #self.loss = nn.CrossEntropyLoss()  # Cross-entropy 

    
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
    
    def predict_step(self, batch, batch_index):
        x, y = batch

        y_hat = self(x)

        predictions = y_hat.argmax(dim=1)
        probs = F.softmax(y_hat, dim=1)

        return {
            "preds": predictions,
            "probs": probs
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)

        return optimizer



def printAUC(y_true, y_probs, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc



def trainXGB(x_train_processed, y_train, x_test_processed, y_test):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed)
    xgb.fit(x_train_processed, y_train)

    y_pred_xgb = xgb.predict(x_test_processed)
    probs = xgb.predict_proba(x_test_processed)

    report = classification_report(y_test, y_pred_xgb, output_dict=True)
    #print(report)
    auc = printAUC(y_test, probs, num_classes=10)

    return report, auc




def trainRF(x_train_processed, y_train, x_test_processed, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(x_train_processed, y_train)

    y_pred_rf = rf.predict(x_test_processed)
    probs = rf.predict_proba(x_test_processed)

    report = classification_report(y_test, y_pred_rf, output_dict=True)
    auc = printAUC(y_test, probs, num_classes=10)

    return report, auc

def trainSVM(x_train_processed, y_train, x_test_processed, y_test):
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    svm.fit(x_train_processed, y_train)

    y_pred_svm = svm.predict(x_test_processed)
    probs = svm.predict_proba(x_test_processed)

    report = classification_report(y_test, y_pred_svm, output_dict=True)
    auc = printAUC(y_test, probs, num_classes=10)

    return report, auc

def trainKNN(x_train_processed, y_train, x_test_processed, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_processed, y_train)

    y_pred_knn = knn.predict(x_test_processed)
    probs = knn.predict_proba(x_test_processed)

    report = classification_report(y_test, y_pred_knn, output_dict=True)
    auc = printAUC(y_test, probs, num_classes=10)

    return report, auc

def trainC45(x_train_processed, y_train, x_test_processed, y_test):
    c45 = DecisionTreeClassifier(random_state=seed)
    c45.fit(x_train_processed, y_train)

    y_pred_c45 = c45.predict(x_test_processed)
    probs = c45.predict_proba(x_test_processed)

    report = classification_report(y_test, y_pred_c45, output_dict=True)
    auc = printAUC(y_test, probs, num_classes=10)

    return report, auc




if __name__ == "__main__":

    # MLP scores
    precisions = dict()
    recalls = dict()
    f1scores = dict()
    accuracies = list()
    macroavgs = list()
    weightedavgs = list()
    aucscores = dict()

    for i in range(10):
        precisions[i] = list()
        recalls[i] = list()
        f1scores[i] = list()
        aucscores[i] = list()

    # RF scores
    RF_precisions = dict()
    RF_recalls = dict()
    RF_f1scores = dict()
    RF_accuracies = list()
    RF_macroavgs = list()
    RF_weightedavgs = list()
    RF_aucscores = dict()

    for i in range (10):
        RF_precisions[i] = list()
        RF_recalls[i] = list()
        RF_f1scores[i] = list()
        RF_aucscores[i] = list()

    # XGB scores
    XGB_precisions = dict()
    XGB_recalls = dict()
    XGB_f1scores = dict()
    XGB_accuracies = list()
    XGB_macroavgs = list()
    XGB_weightedavgs = list()
    XGB_aucscores = dict()

    for i in range (10):
        XGB_precisions[i] = list()
        XGB_recalls[i] = list()
        XGB_f1scores[i] = list()
        XGB_aucscores[i] = list()

    # SVM scores
    SVM_precisions = dict()
    SVM_recalls = dict()
    SVM_f1scores = dict()
    SVM_accuracies = list()
    SVM_macroavgs = list()
    SVM_weightedavgs = list()
    SVM_aucscores = dict()

    for i in range (10):
        SVM_precisions[i] = list()
        SVM_recalls[i] = list()
        SVM_f1scores[i] = list()
        SVM_aucscores[i] = list()

    # KNN scores
    KNN_precisions = dict()
    KNN_recalls = dict()
    KNN_f1scores = dict()
    KNN_accuracies = list()
    KNN_macroavgs = list()
    KNN_weightedavgs = list()
    KNN_aucscores = dict()

    for i in range (10):
        KNN_precisions[i] = list()
        KNN_recalls[i] = list()
        KNN_f1scores[i] = list()
        KNN_aucscores[i] = list()

    # C45 scores
    C45_precisions = dict()
    C45_recalls = dict()
    C45_f1scores = dict()
    C45_accuracies = list()
    C45_macroavgs = list()
    C45_weightedavgs = list()
    C45_aucscores = dict()

    for i in range (10):
        C45_precisions[i] = list()
        C45_recalls[i] = list()
        C45_f1scores[i] = list()
        C45_aucscores[i] = list()

    


    trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
    x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet)

    # 30 different seeds
    #seeds = [1492]
    #seeds = [1,2,3,4,5]
    #seeds = [26, 42, 123, 2024, 999, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63]
    seeds = [26, 42, 123, 2024, 999, 2023, 2022, 2021, 2020, 2019]
    
    for index, seed in enumerate(seeds):
        L.seed_everything(seed, workers=True)
        
        




        label_proportion = 0.01
        x_train_small,_ , y_train_small, _ = train_test_split(x_train_processed, y_train, test_size=1-label_proportion, random_state=seed, stratify=y_train) 
        print("Small training set: ", x_train_small.shape, y_train_small.shape)

        label_proportion = 0.01
        x_val_small,_ , y_val_small, _ = train_test_split(x_val_processed, y_val, test_size=1-label_proportion, random_state=seed, stratify=y_val) 
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
        batch_size = 32

        train_dataloader_embeddings = torch.utils.data.DataLoader(
            train_dataset_embeddings,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        validation_dataloader_embeddings = torch.utils.data.DataLoader(
            validation_dataset_embeddings,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        test_dataloader_embeddings = torch.utils.data.DataLoader(
            test_dataset_embeddings,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )


        model = ClassificationHead(input_dim=186, dropout=0.0)
        n_epochs = 50

        #early_stopping_callback = EarlyStopping(monitor="cl_validation_loss", patience=3)
        trainer = L.Trainer(max_epochs=n_epochs, accelerator='gpu', logger=True, enable_progress_bar=True) #, callbacks=[early_stopping_callback])
        trainer.fit(model, train_dataloader_embeddings, validation_dataloader_embeddings)


        classes = [0,1,2,3,4,5,6,7,8,9]



        #print("\n\n MLP")

        outputs = trainer.predict(model, test_dataloader_embeddings)
        predictions = torch.cat([o["preds"] for o in outputs])
        probs = torch.cat([o["probs"] for o in outputs])

        y_pred_CLHead = predictions.cpu().numpy()
        MLPReport = classification_report(y_test, y_pred_CLHead, output_dict=True)
        # Print f1 score of class 1 from the calssification report

        MLPauc = printAUC(y_test, torch.tensor(probs).cpu().numpy(), num_classes=10)

        #print(MLPReport)
        #print(classification_report(y_test, y_pred_CLHead))
        #print(MLPauc)

        # Store MLP results
        for i in range(10):
            precisions[i].append(MLPReport[str(i)]['precision'])
            recalls[i].append(MLPReport[str(i)]['recall'])
            f1scores[i].append(MLPReport[str(i)]['f1-score'])
            aucscores[i].append(MLPauc[i])

        accuracies.append(MLPReport['accuracy'])
        macroavgs.append(MLPReport['macro avg']['f1-score'])
        weightedavgs.append(MLPReport['weighted avg']['f1-score'])

        # print(precisions)
        # print(recalls)
        # print(f1scores)
        # print(aucscores)


        

        XGBReport, XGBauc = trainXGB(x_train_small, y_train_small, x_test_processed, y_test)

        #print(XGBReport, XGBauc)

        # Store XGB results
        for i in range(10):
            XGB_precisions[i].append(XGBReport[str(i)]['precision'])
            XGB_recalls[i].append(XGBReport[str(i)]['recall'])
            XGB_f1scores[i].append(XGBReport[str(i)]['f1-score'])
            XGB_aucscores[i].append(XGBauc[i])

        XGB_accuracies.append(XGBReport['accuracy'])
        XGB_macroavgs.append(XGBReport['macro avg']['f1-score'])
        XGB_weightedavgs.append(XGBReport['weighted avg']['f1-score'])




        RFReport, RFauc = trainRF(x_train_small, y_train_small, x_test_processed, y_test)

        #print(RFReport, RFauc)

        for i in range(10):
            RF_precisions[i].append(RFReport[str(i)]['precision'])
            RF_recalls[i].append(RFReport[str(i)]['recall'])
            RF_f1scores[i].append(RFReport[str(i)]['f1-score'])
            RF_aucscores[i].append(RFauc[i])

        RF_accuracies.append(RFReport['accuracy'])
        RF_macroavgs.append(RFReport['macro avg']['f1-score'])
        RF_weightedavgs.append(RFReport['weighted avg']['f1-score'])





    # Obtenemos intervalos de confianza

    f = open("More_supervised_all_seeds.txt", "w")

    f.write("Parameters:\n")
    f.write("Number of epochs: " + str(n_epochs) + "\n")
    f.write("Batch size: " + str(batch_size) + "\n")
    f.write("Number of seeds: " + str(len(seeds)) + "\n")
    f.write("XGB: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed)\n")
    f.write("RF: RandomForestClassifier(n_estimators=100, random_state=seed)\n")
    f.write("SVM: SVC(kernel='rbf', probability=True, random_state=seed)\n")
    f.write("KNN: KNeighborsClassifier(n_neighbors=5)\n")
    f.write("C45: DecisionTreeClassifier(random_state=seed)\n")

    # MLP IC
    print("\n\n MLP Confidence Intervals \n\n")
    f.write("\n\n MLP Confidence Intervals \n\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")
        ci_precision = np.percentile(precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(aucscores[i], [2.5, 97.5])

        mean_precision = np.mean(precisions[i])
        mean_recall = np.mean(recalls[i])
        mean_f1score = np.mean(f1scores[i])
        mean_auc = np.mean(aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(accuracies)
    mean_macroavg = np.mean(macroavgs)
    mean_weightedavg = np.mean(weightedavgs)


    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")


    # RF IC

    print("\n\nRF results:")
    f.write("\n\nRF results:\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")

        ci_precision = np.percentile(RF_precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(RF_recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(RF_f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(RF_aucscores[i], [2.5, 97.5])
        mean_precision = np.mean(RF_precisions[i])
        mean_recall = np.mean(RF_recalls[i])
        mean_f1score = np.mean(RF_f1scores[i])
        mean_auc = np.mean(RF_aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(RF_accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(RF_macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(RF_weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(RF_accuracies)
    mean_macroavg = np.mean(RF_macroavgs)
    mean_weightedavg = np.mean(RF_weightedavgs)

    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")


    # XGB IC

    print("\n\nXGB results:")
    f.write("\n\nXGB results:\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")
        ci_precision = np.percentile(XGB_precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(XGB_recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(XGB_f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(XGB_aucscores[i], [2.5, 97.5])
        mean_precision = np.mean(XGB_precisions[i])
        mean_recall = np.mean(XGB_recalls[i])
        mean_f1score = np.mean(XGB_f1scores[i])
        mean_auc = np.mean(XGB_aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(XGB_accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(XGB_macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(XGB_weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(XGB_accuracies)
    mean_macroavg = np.mean(XGB_macroavgs)
    mean_weightedavg = np.mean(XGB_weightedavgs)

    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")




    # SVM IC
    print("\n\nSVM results:")
    f.write("\n\nSVM results:\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")
        ci_precision = np.percentile(SVM_precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(SVM_recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(SVM_f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(SVM_aucscores[i], [2.5, 97.5])
        mean_precision = np.mean(SVM_precisions[i])
        mean_recall = np.mean(SVM_recalls[i])
        mean_f1score = np.mean(SVM_f1scores[i])
        mean_auc = np.mean(SVM_aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(SVM_accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(SVM_macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(SVM_weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(SVM_accuracies)
    mean_macroavg = np.mean(SVM_macroavgs)
    mean_weightedavg = np.mean(SVM_weightedavgs)

    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")


    # KNN IC
    print("\n\nKNN results:")
    f.write("\n\nKNN results:\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")
        ci_precision = np.percentile(KNN_precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(KNN_recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(KNN_f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(KNN_aucscores[i], [2.5, 97.5])
        mean_precision = np.mean(KNN_precisions[i])
        mean_recall = np.mean(KNN_recalls[i])
        mean_f1score = np.mean(KNN_f1scores[i])
        mean_auc = np.mean(KNN_aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(KNN_accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(KNN_macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(KNN_weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(KNN_accuracies)
    mean_macroavg = np.mean(KNN_macroavgs)
    mean_weightedavg = np.mean(KNN_weightedavgs)


    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")



    # C45 IC
    print("\n\nC45 results:")
    f.write("\n\nC45 results:\n")

    for i in range(10):
        print("Class ", i)
        f.write("Class " + str(i) + "\n")
        ci_precision = np.percentile(C45_precisions[i], [2.5, 97.5])
        ci_recall = np.percentile(C45_recalls[i], [2.5, 97.5])
        ci_f1score = np.percentile(C45_f1scores[i], [2.5, 97.5])
        ci_auc = np.percentile(C45_aucscores[i], [2.5, 97.5])

        mean_precision = np.mean(C45_precisions[i])
        mean_recall = np.mean(C45_recalls[i])
        mean_f1score = np.mean(C45_f1scores[i])
        mean_auc = np.mean(C45_aucscores[i])

        print("Precision: ", mean_precision, " CI: ", ci_precision)
        f.write("Precision: " + str(mean_precision) + " CI: " + str(ci_precision) + "\n")
        print("Recall: ", mean_recall, " CI: ", ci_recall)
        f.write("Recall: " + str(mean_recall) + " CI: " + str(ci_recall) + "\n")
        print("F1-score: ", mean_f1score, " CI: ", ci_f1score)
        f.write("F1-score: " + str(mean_f1score) + " CI: " + str(ci_f1score) + "\n")
        print("AUC: ", mean_auc, " CI: ", ci_auc)
        f.write("AUC: " + str(mean_auc) + " CI: " + str(ci_auc) + "\n")

    ci_accuracy = np.percentile(C45_accuracies, [2.5, 97.5])
    ci_macroavg = np.percentile(C45_macroavgs, [2.5, 97.5])
    ci_weightedavg = np.percentile(C45_weightedavgs, [2.5, 97.5])
    mean_accuracy = np.mean(C45_accuracies)
    mean_macroavg = np.mean(C45_macroavgs)
    mean_weightedavg = np.mean(C45_weightedavgs)

    print("\n")
    f.write("\n")
    print("Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Macro avg F1-score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Macro avg F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Weighted avg F1-score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Weighted avg F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")




    


        # # Store values in the corresponding lists and dictionaries for each model
        # for i in range(10):
        #     precisions[i].append(MLPReport[str(i)]['precision'])
        #     recalls[i].append(MLPReport[str(i)]['recall'])
        #     f1scores[i].append(MLPReport[str(i)]['f1-score'])

        #     RF_precisions[i].append(RFReport[str(i)]['precision'])
        #     RF_recalls[i].append(RFReport[str(i)]['recall'])
        #     RF_f1scores[i].append(RFReport[str(i)]['f1-score'])

        #     XGB_precisions[i].append(XGBReport[str(i)]['precision'])
        #     XGB_recalls[i].append(XGBReport[str(i)]['recall'])
        #     XGB_f1scores[i].append(XGBReport[str(i)]['f1-score'])

        # accuracies.append(MLPReport['accuracy'])
        # RF_accuracies.append(RFReport['accuracy'])
        # XGB_accuracies.append(XGBReport['accuracy'])

        # macroavgs.append(MLPReport['macro avg']['f1-score'])
        # RF_macroavgs.append(RFReport['macro avg']['f1-score'])
        # XGB_macroavgs.append(XGBReport['macro avg']['f1-score'])

        # weightedavgs.append(MLPReport['weighted avg']['f1-score'])
        # RF_weightedavgs.append(RFReport['weighted avg']['f1-score'])
        # XGB_weightedavgs.append(XGBReport['weighted avg']['f1-score'])

        # aucscores.append(MLPauc)
        # RF_aucscores.append(RFauc)
        # XGB_aucscores.append(XGBauc)

        



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





