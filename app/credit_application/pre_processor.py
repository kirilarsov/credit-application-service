# standard library imports
import pandas as pd
import numpy as np

# third-party imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# application-specific imports
from app.credit_application import model as cca_model

# Initializations
scaler = StandardScaler()


def transform_columns_to_float(applications, columns):
    """
    Transforms specified columns in a pandas DataFrame to float data type.

    :param applications: The pandas DataFrame containing the data.
    :type applications: pandas.DataFrame
    :param columns: A list of column names to transform to float data type.
    :type columns: list
    :return: The updated pandas DataFrame with specified columns transformed to float data type.
    :rtype: pandas.DataFrame
    """
    print(f"Pre-processor: transform columns to float: {columns}")
    for column in columns:
        applications[column] = applications[column].astype(float)
    return applications


def impute_with_knn(applications, data_types):
    """
    Imputes missing numeric values using K-Nearest Neighbors algorithm.

    :param applications: A pandas DataFrame representing the dataset.
    :param data_types: A list of data types to be considered for imputation.
    :return: A pandas DataFrame with imputed numerical values.

    """
    print("Pre-processor: impute missing numeric values")
    numerical_data = applications.select_dtypes(include=data_types)
    imputer = KNNImputer(n_neighbors=2)
    imputed_numerical_data = imputer.fit_transform(numerical_data)
    return pd.DataFrame(imputed_numerical_data)


def impute_numeric_features(applications):
    """
    Imputes missing values in numeric features of the given applications dataset.

    :param applications: A pandas DataFrame representing the applications dataset.
    :return: A new pandas DataFrame with missing values in numeric features imputed.
    """
    columns_to_transform = [1, 13]
    data_types_to_impute = ['float64', 'int64']
    applications = transform_columns_to_float(applications, columns_to_transform)
    imputed_numerical_applications = impute_with_knn(applications, data_types_to_impute)
    return imputed_numerical_applications


def feature_selection(applications_ready):
    """
    Perform feature selection on the given dataset.

    :param applications_ready: A pandas DataFrame containing the dataset.
    :return: Selected features as a pandas Index.
    """
    print("Pre-processor: recommended features using feature selection ")
    applications_ready_copy = applications_ready.copy()
    applications_ready_copy.columns = applications_ready_copy.columns.astype(str)
    applications_train, applications_test = train_test_split(applications_ready_copy, test_size=0.33,
                                                             random_state=42)
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    x_train, y_train = (
        applications_train.drop(['15'], axis=1).values,
        applications_train['15'].values
    )

    rf.fit(x_train, y_train)
    model = SelectFromModel(rf, prefit=True)
    features_bool = model.get_support()
    features = applications_ready_copy.drop(['15'], axis=1).columns[features_bool]
    print("Pre-processor: feature selection: ")
    print(features)
    return features


def normalization(x_train, x_test):
    """
    Normalize the input data for pre-processing.

    :param x_train: The training data to be normalized.
    :param x_test: The test data to be normalized.
    :return: The normalized training data and test data.

    Example:
        >>> x_train = [[1, 2, 3], [4, 5, 6]]
        >>> x_test = [[7, 8, 9], [10, 11, 12]]
        >>> normalized_x_train, normalized_x_test = normalization(x_train, x_test)
        >>> print(normalized_x_train)
        [[0.26726124 0.53452248 0.80178373]
         [0.45584231 0.56980288 0.68376344]]
        >>> print(normalized_x_test)
        [[0.58925616 0.78434101 0.78434101]
         [0.71713717 0.78348653 0.8498359]]

    """
    print("Pre-processor: normalization ")
    norm = Normalizer()
    return norm.fit_transform(x_train), norm.transform(x_test)


def standardization(x_train, x_test):
    """
    Apply standardization to the given training and testing data.

    :param x_train: A numpy array representing the training data.
    :param x_test: A numpy array representing the testing data.

    :return: A tuple containing the standardized training data and standardized testing data.
    """
    print("Pre-processor: standardization ")
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def train_and_evaluate(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    score = logreg.score(x_train, y_train)
    matrix = confusion_matrix(y_test, y_pred)
    print(score)
    print(matrix)


def fill_na_with_most_frequent(data, column):
    """
    Fill null values in the specified column with the most frequent value.

    :param data: DataFrame containing the data.
    :type data: pandas.DataFrame
    :param column: Name of the column to fill null values in.
    :type column: str
    :return: DataFrame with null values in the specified column filled with the most frequent value.
    :rtype: pandas.DataFrame
    """
    most_frequent = data[column].value_counts().index[0]
    return data.fillna(most_frequent)


def fill_categorical_na_with_most_frequent(applications):
    """
    Fill_Categorical_NA_With_Most_Frequent

    This method takes in a DataFrame (applications) and fills the missing values in categorical features with the most frequent values.

    :param applications: The input DataFrame containing the applications data
    :return: A new DataFrame with the missing values filled with the most frequent values

    Example Usage:
        >>> applications = pd.DataFrame({'col1': ['A', np.nan, 'B'], 'col2': ['X', np.nan, 'Y'], 'col3': [1, 2, 3]})
        >>> filled_data = fill_categorical_na_with_most_frequent(applications)
    """
    print("Pre-processor: impute categorical features with most frequent values")
    categorical_data = applications.select_dtypes(include=['object'])
    for column in categorical_data.columns:
        categorical_data = fill_na_with_most_frequent(categorical_data, column)
    return pd.get_dummies(categorical_data, columns=[0, 3, 4, 5, 6, 8, 9, 11, 12])


def remove_outliers(applications_ready):
    print("Pre-processor: remove the outliers")
    applications_ready.drop(applications_ready[applications_ready[3] > 50].index, inplace=True)
    applications_ready.drop(applications_ready[applications_ready[4] > 1400].index, inplace=True)
    applications_ready.drop(applications_ready[applications_ready[5] > 60000].index, inplace=True)
    return applications_ready


def train_test_data_split(df, test_size):
    """
    Split the given dataframe into train and test data.

    :param df: The input dataframe.
    :param test_size: The proportion of data to allocate for the test set (0-1).
    :return: A tuple containing x_train, y_train, x_test, y_test, and applications_train.

    """
    print("Pre-processor: data train & test split ")
    applications_train, applications_test = train_test_split(df, test_size=test_size,
                                                             random_state=42)

    x_train, y_train = (
        applications_train.drop([15], axis=1).values,
        applications_train[15].values
    )
    x_test, y_test = (
        applications_test.drop([15], axis=1).values,
        applications_test[15].values
    )
    return x_train, y_train, x_test, y_test, applications_train


def mark_missing(applications_original):
    print("Pre-processor: mark NaN")
    applications = applications_original.replace("?", np.NaN)
    applications[15] = applications[15].replace("-", 0)
    applications[15] = applications[15].replace("+", 1)
    return applications


def request_rescale(x):
    """
    Rescales the input data using a predefined scaler.

    :param x: The input data to be rescaled.
    :return: The rescaled input data.
    """
    return scaler.transform(x)


def request_encoded(request_json):
    r = cca_model.CreditSchema().load(request_json)
    request_df = pd.DataFrame(
        [[r.p0, r.p1, r.p2, r.p3, r.p4, r.p5, r.p6, r.p7, r.p8, r.p9, r.p10, r.p11, r.p12, r.p13, r.p14]])

    request_df[1] = request_df[1].astype(float)
    request_df[13] = request_df[13].astype(float)

    return pd.get_dummies(request_df, columns=[0, 3, 4, 5, 6, 8, 9, 11, 12])