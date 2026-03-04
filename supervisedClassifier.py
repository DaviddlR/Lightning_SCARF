import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def preprocessing(filenameTrain, filenameTest):

    df = pd.read_parquet(filenameTrain)
    dfTest = pd.read_parquet(filenameTest)


    # Separate labels and features
    x_train = df.drop(columns=['label', 'attack_cat'])
    y_train = df['attack_cat']

    x_test = dfTest.drop(columns=['label', 'attack_cat'])
    y_test = dfTest['attack_cat']

    # Label encoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Create validation set (10% of the training data)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=24)  

    

    #  One-hot encoding for categorical features
    cat_columns = x_train.select_dtypes(include=['category', 'object']).columns
    num_columns = x_train.select_dtypes(exclude=['category', 'object']).columns

    num_transformer = StandardScaler()
    col_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_columns),
            ('cat', col_transformer, cat_columns)
        ]
    )
    
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    x_val_processed = preprocessor.transform(x_val)


    # Check if one-hot encoder was successfull
    if len(cat_columns) > 0:
        print(f"Original number of features: {len(x_train.columns)}")
        print(f"Number of features after one-hot encoding: {x_train_processed.shape[1]}")

    print("Training set: ", x_train_processed.shape)
    print("Validation set: ", x_val_processed.shape)
    print("Test set: ", x_test_processed.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    return x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test

    





def trainMLP():
    pass



def trainXGB(x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=1492)
    xgb.fit(x_train_processed, y_train)

    y_pred_xgb = xgb.predict(x_test_processed)
    print("\n\n---------- XGBoost Classification Report ----------")
    print(classification_report(y_test, y_pred_xgb))




def trainRF(x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=1492)
    rf.fit(x_train_processed, y_train)

    y_pred_rf = rf.predict(x_test_processed)
    print("\n\n---------- Random Forest Classification Report ----------")
    print(classification_report(y_test, y_pred_rf))





if __name__ == "__main__":
    
    trainSet = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    testSet = "UNSW_NB15/UNSW_NB15_testing-set.parquet"
    x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test = preprocessing(trainSet, testSet)

    classifierType = "XGB"


    if classifierType == "MLP":
        model = trainMLP()
    elif classifierType == "XGB":
        model = trainXGB(x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test)
    elif classifierType == "RF":
        model = trainRF(x_train_processed, y_train, x_val_processed, y_val, x_test_processed, y_test)
    else:
        print("Invalid classifier type. Please choose from MLP, XGB, or RF.")







