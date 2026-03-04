import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing(filename):

    df = pd.read_parquet(filename)

    # Separate labels and features
    X = df.drop(columns=['label', 'attack_cat'])
    Y = df['attack_cat']

    # Label encoder
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(Y)
    

    





def trainMLP():
    pass



def trainXGB():
    pass



def trainRF():
    pass





if __name__ == "__main__":
    
    filename = "UNSW_NB15/UNSW_NB15_training-set.parquet"
    data = preprocessing(filename)

    classifierType = "MLP"


    if classifierType == "MLP":
        model = trainMLP()
    elif classifierType == "XGB":
        model = trainXGB()
    elif classifierType == "RF":
        model = trainRF()
    else:
        print("Invalid classifier type. Please choose from MLP, XGB, or RF.")







