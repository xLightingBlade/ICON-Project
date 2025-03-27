from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer
import seaborn as sns
from pandas.plotting import table
import numpy as np
import six

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
        get_pie_chart_of_classes(og_dataset, "dataset pie chart.png")

    return og_dataset


def split_x_and_y(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    return x_train, x_test, y_train, y_test

def get_pie_chart_of_classes(dataframe:DataFrame, filename):
    class_counts = dataframe['Class'].value_counts()
    labels = ['Real', 'Fraud']
    plt.figure(figsize=(6, 6))
    class_counts.plot.pie(autopct='%1.2f%%', startangle=90, cmap='coolwarm', labels = labels)
    plt.title("Class distribution, original dataset")
    plt.savefig(filename)
    plt.show()

def dataframe_choose_cols_and_sample(dataframe:DataFrame, n_samples, bins, strategy):
    df_sample = dataframe_get_sample(dataframe, n_samples, bins, strategy)
    return df_sample
    """
    #prima uno shuffle del dataframe
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    x = dataframe[['Time', 'V1', 'V2', 'V3', 'Amount']]
    y = dataframe['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    sm = SMOTE(random_state=42, sampling_strategy=.5)
    x_train_oversampled, y_train_oversampled = sm.fit_resample(x_train, y_train)
    x_train = pd.DataFrame(x_train_oversampled, columns=x_train.columns)
    y_train = pd.Series(y_train_oversampled)
    x_train['Class'] = y_train
    fraud_df = x_train.loc[x_train['Class'] == 1 ].sample(n_samples // 2)
    real_df = x_train.loc[x_train['Class'] == 0].sample(n_samples // 2)
    df_sample = pd.concat([fraud_df, real_df], ignore_index=True)
    discrete_sample = discretize_df(df_sample, bins, strategy)
    print("discrete sample df")
    print(discrete_sample.head())
    return discrete_sample
    """


def discretize_df(dataframe:DataFrame, bins, strategy):
    print('prima')
    print(dataframe.head())
    kbins = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    cols = [col for col in dataframe.columns if col != 'Class']
    x_discrete = kbins.fit_transform(dataframe[cols])
    discrete_df = pd.DataFrame(x_discrete, columns=cols, index=dataframe.index)
    discrete_df['Class'] = dataframe['Class']
    discrete_df = discrete_df[dataframe.columns]
    print(discrete_df.head())
    return discrete_df

def dataset_oversample(data:DataFrame):
    oversampler = SMOTE(sampling_strategy='auto', random_state=42)
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

#provvede a restituire un dataframe di dimensione n_samples, discretizzato (se to_discretize) con kbinsdiscretizer, proiettato sulle columns indicate
def dataframe_get_sample(dataframe:DataFrame, n_samples, bins=1, strategy= '', to_discretize=True, columns = ['Amount', 'Time', 'V1', 'V2', 'V3']):
    if to_discretize:
        dataframe = discretize_df(dataframe, bins, strategy)

    def stratified_sample(df, n, class_col='class', bin_cols=columns):
        sampled_list = []

        # Process each class separately
        for cls in [0, 1]:
            cls_df = df[df[class_col] == cls].copy()
            # DataFrame to hold mandatory rows for each bin in each specified column
            mandatory = pd.DataFrame()
            # For each discretized column, select one row per unique bin value
            for col in bin_cols:
                unique_bins = cls_df[col].unique()
                for bin_val in unique_bins:
                    # Candidate rows that fall into the bin
                    candidates = cls_df[cls_df[col] == bin_val]
                    if not candidates.empty:
                        mandatory = pd.concat([mandatory, candidates.sample(n=1)])

            # Remove any duplicate rows (in case a row was selected for more than one column)
            mandatory = mandatory.drop_duplicates()
            # Count how many rows we already have
            m = len(mandatory)
            # If mandatory rows exceed half_n, randomly select half_n among them
            if m > n//2:
                cls_sample = mandatory.sample(n=n//2)
            else:
                # Otherwise, sample additional rows (avoiding duplicates) to reach half_n
                remaining = n//2 - m
                remaining_df = cls_df.drop(mandatory.index, errors='ignore')
                additional = remaining_df.sample(n=remaining)
                cls_sample = pd.concat([mandatory, additional])
            sampled_list.append(cls_sample)

        # Combine class samples and shuffle the final result
        print("sto finendo")
        final_sample = pd.concat(sampled_list).sample(frac=1).reset_index(drop=True)
        return final_sample
    print("inizio stratified sample")
    sampled_df = stratified_sample(dataframe, n=n_samples, class_col='Class', bin_cols=columns)
    print("samples" , sampled_df)
    return sampled_df[['Amount', 'Time', 'V1', 'V2', 'V3', 'Class']]

    """
    fraud_df = dataframe.loc[dataframe['Class'] == 1].sample(n_samples // 2)
    real_df = dataframe.loc[dataframe['Class'] == 0].sample(n_samples // 2)
    samples = pd.concat([fraud_df, real_df], ignore_index=True)
    if to_discretize:
        discrete_sample = discretize_df(samples, 20, 'kmeans')
        return discrete_sample
    return samples
    """

def visualize_and_save_dataframe(dataframe, path):
    ax, fig = render_mpl_table(dataframe)
    savepath = path.split('.')[0]
    fig.savefig(f"{savepath}.png")

#fonte: https://stackoverflow.com/questions/26678467/export-a-pandas-dataframe-as-a-table-image
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax, fig
