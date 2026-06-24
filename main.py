import os

import numpy as np

import torch
import lightning as L

from SCARF import SCARFLightning
from SCARFDataset import SCARFDataset
from featureExtractor import FeatureExtractor
from supervisedClassifier import ClassificationHead

from preprocessing import readData


from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import time


def printAUC(y_true, y_probs, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc



if __name__ == "__main__":

    print("\n\n FULL TRAINING AND EVALUATION \n\n")
    print("We use several seeds to create confidence intervals. It may take a while to run all seeds :D\n")



    
    # EEDD to store results for all seeds and models. We will compute confidence intervals with these results.
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



    # 30 seeds
    seeds = [1]
    #seeds = [1492, 67]
    #seeds = [1,2,3,4,5]
    #seeds = [26, 42, 123, 2024, 999, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63]
    #seeds = [26, 42, 123, 2024, 999, 2023, 2022, 2021, 2020, 2019]
    for seed in seeds:
        
        print(f"\n\n SEED: {seed} \n\n")

        L.seed_everything(seed, workers=True)

        print("\n Pretraining SCARF model...")

        trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
        testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
        x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = readData(trainSet, testSet, seed=seed)

        train_dataset = SCARFDataset(x_train_processed, y_train)
        validation_dataset = SCARFDataset(x_val_processed, y_val)
        test_dataset = SCARFDataset(x_test_processed, y_test)


        # TRAINING
        batch_size = 256

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create lightning model
        pretrained_model = SCARFLightning(in_dim = train_dataset.shape[1], 
                            hidden_dim = 256, 
                            num_hidden = 4, 
                            head_hidden_dim = 256,  
                            head_num_hidden = 2, 
                            dropout = 0, 
                            corruption_rate = 0.6)
        


        # Create trainer and fit

        #early_stopping_callback = EarlyStopping(monitor="validation_loss", patience=3, mode="min")
        n_epochs = 150  
        trainer = L.Trainer(max_epochs=n_epochs, accelerator='gpu', logger=True, enable_progress_bar=True) #callbacks=[early_stopping_callback])

        start_time = time.time()

        trainer.fit(pretrained_model, train_loader, validation_loader)

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n\n\n ---- Training time: {training_time} seconds")



        print("\n Extracting features and training classification head...")
        # Model already loaded...

        pretrained_model.freeze()  # Freeze the SCARF model

        extractor = FeatureExtractor(pretrained_model)

        doitsmall = True
        label_proportion = 0.01

        if doitsmall:
            x_train_small, _, y_train_small, _ = train_test_split(x_train_processed, y_train, test_size=1-label_proportion, random_state=seed, stratify=y_train)
            train_dataset_small = SCARFDataset(x_train_small, y_train_small)
            train_loader = torch.utils.data.DataLoader(train_dataset_small, batch_size=batch_size, shuffle=True)

            # Hacemos pequeño solo el conjunto de entrenamiento. Validación y test los dejamos completos.
            # x_val_small, _, y_val_small, _ = train_test_split(x_val_processed, y_val, test_size=1-label_proportion, random_state=seed, stratify=y_val)
            # validation_dataset_small = SCARFDataset(x_val_small, y_val_small)
            # validation_loader = torch.utils.data.DataLoader(validation_dataset_small, batch_size=batch_size, shuffle=False)

            

        # Classification on top of the embeddings
        start_time = time.time()
        
        train_embeddings, train_labels = extractor.extractFeatures(train_loader, labels=True)

        end_time = time.time()
        extraction_time = end_time - start_time
        print(f"\n\n\n ---- Feature extraction time: {extraction_time} seconds")

        validation_embeddings, validation_labels = extractor.extractFeatures(validation_loader, labels=True)
        test_embeddings, test_labels = extractor.extractFeatures(test_loader, labels=True)

        # print(train_embeddings.shape)
        # print(validation_embeddings.shape)
        # print(test_embeddings.shape)
        # print(train_labels.shape)
        # print(validation_labels.shape)
        # print(test_labels.shape)

        print("Training set: ", train_embeddings.shape, train_labels.shape)
        print("Validation set: ", validation_embeddings.shape, validation_labels.shape)
        print("Test set: ", test_embeddings.shape, test_labels.shape)


        train_dataset_embeddings = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        validation_dataset_embeddings = torch.utils.data.TensorDataset(validation_embeddings, validation_labels)
        test_dataset_embeddings = torch.utils.data.TensorDataset(test_embeddings, test_labels)

        train_dataloader_embeddings = torch.utils.data.DataLoader(
            train_dataset_embeddings,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        validation_dataloader_embeddings = torch.utils.data.DataLoader(
            validation_dataset_embeddings,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        test_dataloader_embeddings = torch.utils.data.DataLoader(
            test_dataset_embeddings,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        # MLP CLASSIFICATION ON TOP OF EMBEDDINGS
        classification_head = ClassificationHead(dropout=0.2)

        #early_stopping_callback = EarlyStopping(monitor="cl_validation_loss", patience=3)
        #trainer = L.Trainer(max_epochs=200, accelerator='gpu', logger=True, enable_progress_bar=True, callbacks=[early_stopping_callback])
        trainer = L.Trainer(max_epochs=50, accelerator='gpu', logger=True, enable_progress_bar=True)

        start_time = time.time()

        trainer.fit(classification_head, train_dataloader_embeddings, validation_dataloader_embeddings)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"\n\n\n ---- MLP_Training time: {training_time} seconds")
        




        #-------------- TEST --------------#
        #trainer.test(classification_head, test_dataloader_embeddings)
        start_time = time.time()
        
        outputs = trainer.predict(classification_head, test_dataloader_embeddings)
        
        end_time = time.time()
        test_time = end_time - start_time
        print(f"\n\n\n ---- MLP_classification time: {test_time} seconds")

        predictions = torch.cat([o["preds"] for o in outputs])
        probs = torch.cat([o["probs"] for o in outputs])

        y_pred_CLHead = predictions.cpu().numpy()

        # Save MLP results
        MLP_report = classification_report(y_test, y_pred_CLHead, output_dict=True)
        MLP_auc = printAUC(y_test, torch.tensor(probs).cpu().numpy(), num_classes=10)

        for i in range(10):
            precisions[i].append(MLP_report[str(i)]['precision'])
            recalls[i].append(MLP_report[str(i)]['recall'])
            f1scores[i].append(MLP_report[str(i)]['f1-score'])
            aucscores[i].append(MLP_auc[i])

        accuracies.append(MLP_report['accuracy'])
        macroavgs.append(MLP_report['macro avg']['f1-score'])
        weightedavgs.append(MLP_report['weighted avg']['f1-score'])










        # XGBOOST CLASSIFICATION ON TOP OF EMBEDDINGS
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed)

        start_time = time.time()
        xgb.fit(train_embeddings, train_labels)
        end_time = time.time()
        xgb_training_time = end_time - start_time
        print(f"\n\n\n ---- XGBoost Training time: {xgb_training_time} seconds")


        start_time = time.time()
        y_pred_xgb = xgb.predict(test_embeddings)
        end_time = time.time()
        xgb_test_time = end_time - start_time
        print(f"\n\n\n ---- XGBoost Testing time: {xgb_test_time} seconds")

        xgb_probs = xgb.predict_proba(test_embeddings)


        print("\n\n---------- XGBoost Classification Report ----------")

        # Save XGBoost results
        XGBOOST_report = classification_report(test_labels, y_pred_xgb, output_dict=True)
        XGBOOST_auc = printAUC(test_labels, xgb_probs, num_classes=10)

        for i in range(10):
            XGB_precisions[i].append(XGBOOST_report[str(i)]['precision'])
            XGB_recalls[i].append(XGBOOST_report[str(i)]['recall'])
            XGB_f1scores[i].append(XGBOOST_report[str(i)]['f1-score'])
            XGB_aucscores[i].append(XGBOOST_auc[i])

        XGB_accuracies.append(XGBOOST_report['accuracy'])
        XGB_macroavgs.append(XGBOOST_report['macro avg']['f1-score'])
        XGB_weightedavgs.append(XGBOOST_report['weighted avg']['f1-score'])





        # Random Forest CLASSIFICATION ON TOP OF EMBEDDINGS
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)

        start_time = time.time()
        rf.fit(train_embeddings, train_labels)
        end_time = time.time()
        rf_training_time = end_time - start_time
        print(f"\n\n\n ---- Random Forest Training time: {rf_training_time} seconds")

        start_time = time.time()
        y_pred_rf = rf.predict(test_embeddings)
        end_time = time.time()
        rf_test_time = end_time - start_time
        print(f"\n\n\n ---- Random Forest Testing time: {rf_test_time} seconds")

        
        rf_probs = rf.predict_proba(test_embeddings)
        print("\n\n---------- Random Forest Classification Report ----------")

        # Save RF results
        RF_report = classification_report(test_labels, y_pred_rf, output_dict=True)
        RF_auc = printAUC(test_labels, rf_probs, num_classes=10)

        for i in range(10):
            RF_precisions[i].append(RF_report[str(i)]['precision'])
            RF_recalls[i].append(RF_report[str(i)]['recall'])
            RF_f1scores[i].append(RF_report[str(i)]['f1-score'])
            RF_aucscores[i].append(RF_auc[i])
        
        RF_accuracies.append(RF_report['accuracy'])
        RF_macroavgs.append(RF_report['macro avg']['f1-score'])
        RF_weightedavgs.append(RF_report['weighted avg']['f1-score'])




        # SVM CLASSIFICATION ON TOP OF EMBEDDINGS
        # from sklearn.svm import SVC
        svm = SVC(probability=True, random_state=seed)
        start_time = time.time()
        svm.fit(train_embeddings, train_labels)
        end_time = time.time()
        svm_training_time = end_time - start_time
        print(f"\n\n\n ---- SVM Training time: {svm_training_time} seconds")

        start_time = time.time()
        y_pred_svm = svm.predict(test_embeddings)
        end_time = time.time()
        svm_test_time = end_time - start_time
        print(f"\n\n\n ---- SVM Testing time: {svm_test_time} seconds")

        svm_probs = svm.predict_proba(test_embeddings)
        print("\n\n---------- SVM Classification Report ----------")

        # Save SVM results
        SVM_report = classification_report(test_labels, y_pred_svm, output_dict=True)
        SVM_auc = printAUC(test_labels, svm_probs, num_classes=10)
        for i in range(10):
            SVM_precisions[i].append(SVM_report[str(i)]['precision'])
            SVM_recalls[i].append(SVM_report[str(i)]['recall'])
            SVM_f1scores[i].append(SVM_report[str(i)]['f1-score'])
            SVM_aucscores[i].append(SVM_auc[i])

        SVM_accuracies.append(SVM_report['accuracy'])
        SVM_macroavgs.append(SVM_report['macro avg']['f1-score'])
        SVM_weightedavgs.append(SVM_report['weighted avg']['f1-score'])



        # KNN CLASSIFICATION ON TOP OF EMBEDDINGS
        knn = KNeighborsClassifier()
        start_time = time.time()
        knn.fit(train_embeddings, train_labels)
        end_time = time.time()
        knn_training_time = end_time - start_time
        print(f"\n\n\n ---- KNN Training time: {knn_training_time} seconds")

        start_time = time.time()
        y_pred_knn = knn.predict(test_embeddings)
        end_time = time.time()
        knn_test_time = end_time - start_time
        print(f"\n\n\n ---- KNN Testing time: {knn_test_time} seconds")


        knn_probs = knn.predict_proba(test_embeddings)
        print("\n\n---------- KNN Classification Report ----------")

        # Save KNN results
        KNN_report = classification_report(test_labels, y_pred_knn, output_dict=True)
        KNN_auc = printAUC(test_labels, knn_probs, num_classes=10)
        for i in range(10):
            KNN_precisions[i].append(KNN_report[str(i)]['precision'])
            KNN_recalls[i].append(KNN_report[str(i)]['recall'])
            KNN_f1scores[i].append(KNN_report[str(i)]['f1-score'])
            KNN_aucscores[i].append(KNN_auc[i])

        KNN_accuracies.append(KNN_report['accuracy'])
        KNN_macroavgs.append(KNN_report['macro avg']['f1-score'])
        KNN_weightedavgs.append(KNN_report['weighted avg']['f1-score'])


        # C45 CLASSIFICATION ON TOP OF EMBEDDINGS
        c45 = DecisionTreeClassifier(random_state=seed)
        start_time = time.time()
        c45.fit(train_embeddings, train_labels)
        end_time = time.time()
        c45_training_time = end_time - start_time
        print(f"\n\n\n ---- C45 Training time: {c45_training_time} seconds")


        start_time = time.time()
        y_pred_c45 = c45.predict(test_embeddings)
        end_time = time.time()
        c45_test_time = end_time - start_time
        print(f"\n\n\n ---- C45 Testing time: {c45_test_time} seconds")
        
        
        c45_probs = c45.predict_proba(test_embeddings)
        print("\n\n---------- C45 Classification Report ----------")

        # Save C45 results
        C45_report = classification_report(test_labels, y_pred_c45, output_dict=True)
        C45_auc = printAUC(test_labels, c45_probs, num_classes=10)
        for i in range(10):
            C45_precisions[i].append(C45_report[str(i)]['precision'])
            C45_recalls[i].append(C45_report[str(i)]['recall'])
            C45_f1scores[i].append(C45_report[str(i)]['f1-score'])
            C45_aucscores[i].append(C45_auc[i])

        C45_accuracies.append(C45_report['accuracy'])
        C45_macroavgs.append(C45_report['macro avg']['f1-score'])
        C45_weightedavgs.append(C45_report['weighted avg']['f1-score'])


    # After all seeds are done, we can compute confidence intervals and print results

    # Also print the results in a TXT file

    f = open("results.txt", "w")


    # Save parameters
    f.write("Parameters:\n")
    f.write("Number of epochs: " + str(n_epochs) + "\n")
    f.write("Batch size: " + str(batch_size) + "\n")
    f.write("Number of seeds: " + str(len(seeds)) + "\n")
    if doitsmall:
        f.write("Label proportion: " + str(label_proportion) + "\n")
    # f.write("MLP: L.Trainer(max_epochs=50, accelerator='gpu', logger=True, enable_progress_bar=True)\n")
    # f.write("XGB: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed)\n")
    # f.write("RF: RandomForestClassifier(n_estimators=100, random_state=seed)\n")

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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")


    # XGBoost IC
    print("\n\n XGBoost Confidence Intervals \n\n")
    f.write("\n\n XGBoost Confidence Intervals \n\n")
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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")



    # Random Forest IC
    print("\n\n Random Forest Confidence Intervals \n\n")
    f.write("\n\n Random Forest Confidence Intervals \n\n")
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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")



    # SVM IC
    print("\n\n SVM Confidence Intervals \n\n")
    f.write("\n\n SVM Confidence Intervals \n\n")
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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")

    # KNN IC
    print("\n\n KNN Confidence Intervals \n\n")
    f.write("\n\n KNN Confidence Intervals \n\n")
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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")




    # C45 IC
    print("\n\n C45 Confidence Intervals \n\n")
    f.write("\n\n C45 Confidence Intervals \n\n")
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
    print("Overall Accuracy: ", mean_accuracy, " CI: ", ci_accuracy)
    f.write("Overall Accuracy: " + str(mean_accuracy) + " CI: " + str(ci_accuracy) + "\n")
    print("Overall Macro F1-Score: ", mean_macroavg, " CI: ", ci_macroavg)
    f.write("Overall Macro F1-Score: " + str(mean_macroavg) + " CI: " + str(ci_macroavg) + "\n")
    print("Overall Weighted F1-Score: ", mean_weightedavg, " CI: ", ci_weightedavg)
    f.write("Overall Weighted F1-Score: " + str(mean_weightedavg) + " CI: " + str(ci_weightedavg) + "\n")

    f.close()



