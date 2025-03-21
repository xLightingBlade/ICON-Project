from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import seaborn as sns

TEST_SIZE = .2

def dataset_load(path, explore=False):
    og_dataset = pd.read_csv(path)
    if explore:
        print(og_dataset.head())
        print(og_dataset[pd.isnull(og_dataset).any(axis=1)])  # nessuna riga con valori NaN
        print(og_dataset.dtypes)
        for column in og_dataset.columns:
            print(f'Column : {column} \n Variance: {og_dataset[column].var()}\n')
        print(og_dataset.duplicated().sum())
        og_dataset.drop_duplicates()
        get_pie_chart_of_classes(og_dataset)

    return og_dataset


def split_x_and_y(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    return x_train, x_test, y_train, y_test

def get_pie_chart_of_classes(dataframe:DataFrame):
    class_counts = dataframe['Class'].value_counts()
    labels = ['Real', 'Fraud']
    plt.figure(figsize=(6, 6))
    class_counts.plot.pie(autopct='%1.2f%%', startangle=90, cmap='coolwarm', labels = labels)
    plt.title("Class distribution, original dataset")
    plt.savefig('dataset pie chart.png')
    plt.show()

def dataset_oversample(data:DataFrame):
    oversampler = SMOTE(sampling_strategy=0.5, random_state=42)
    x = data.drop('Class', axis=1)
    y = data['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train_oversampled, y_train_oversampled = oversampler.fit_resample(x_train, y_train)
    x_train_res_df = pd.DataFrame(x_train_oversampled, columns=x.columns)
    x_train_res_df['Class'] = y_train_oversampled
    x_test = pd.DataFrame(x_test, columns = x.columns)
    x_test['Class'] = y_test
    data = pd.concat([x_train_res_df, x_test], ignore_index=True)
    return data

def dataset_scale(data:DataFrame, to_oversample = False):
    print('original data')
    print(data.describe())
    x = data.drop('Class', axis=1)
    y = data['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns, index=x_test.index)
    train_data = pd.concat([x_train_scaled, y_train], axis=1)
    test_data = pd.concat([x_test_scaled, y_test], axis=1)
    data = pd.concat([train_data, test_data], ignore_index=True)
    if to_oversample:
        data = dataset_oversample(data)
        print('dataframe oversampled')
        print(data.describe())
        return data
    print("Dataframe 'scalato'")
    print(data.describe())
    return data

def correlation_matrix(dataframe:DataFrame, filename):
    corr_matrix = dataframe.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation Matrix {filename.split('data')[0]}")
    plt.savefig(filename)
    plt.show()
