#Label Encoder
from sklearn.preprocessing import LabelEncoder

def label_encode(data, columns):
    encoders = {}
    for col in columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
    return data, encoders

encoded_df, encoders = label_encode(df, columns_to_encode)



#One hot Encoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def onehot_encode(data, columns):
    encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid multicollinearity
    encoded = encoder.fit_transform(data[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=data.index)
    
    # Drop original columns and concatenate encoded ones
    data = data.drop(columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    
    return data, encoder

encoded_df, encoder = onehot_encode(df, columns_to_encode)



#Correlation Matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_matrix(data, method='pearson', figsize=(10, 8), cmap='coolwarm', annot=True):
    corr_matrix = data.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap=cmap, annot=annot, fmt=".2f", square=True)
    plt.title(f'Correlation Matrix ({method.capitalize()} method)', fontsize=16)
    plt.show()


#Check For Duplicates
def check_duplicates(data):
    duplicates = data[data.duplicated()]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows.")
    else:
        print("No duplicate rows found.")
    return duplicates

duplicates = check_duplicates(df)



