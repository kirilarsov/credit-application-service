# standard library imports
import pandas as pd
import numpy as np

# third-party imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from pkg_resources import resource_filename
from sklearn.model_selection import GridSearchCV

# application-specific imports
from app.credit_application import domain as cca_model
# Initializations
scaler = StandardScaler()
enc = OneHotEncoder(handle_unknown='ignore')
norm = Normalizer()

def transform_columns_to_float(applications, columns):
    """
    Convert specified columns in a DataFrame to float type.

    This function takes a DataFrame and a list of column indices, converting the data type of the specified columns to float.
    :param applications: The DataFrame containing the data.
    :type applications: pandas.DataFrame
    :param columns: A list of column indices to convert to float type.
    :type columns: list
    :return: The DataFrame with specified columns converted to float type.
    :rtype: pandas.DataFrame
    """
    print(f"Pre-processor: transform columns to float: {columns}")
    for column in columns:
        # Convert the data type of each specified column to float
        applications[column] = applications[column].astype(float)
    return applications


def impute_with_knn(applications, data_types):
    """
    Impute missing numeric values using the K-Nearest Neighbors algorithm.

    This function selects numeric columns based on the specified data types and applies KNN imputation to fill in missing values.

    :param applications: A pandas DataFrame representing the dataset.
    :type applications: pandas.DataFrame
    :param data_types: A list of data types to be considered for imputation.
    :type data_types: list
    :return: A pandas DataFrame with imputed numerical values.
    :rtype: pandas.DataFrame
    """
    # Log the start of the imputation process
    print("Pre-processor: impute missing numeric values")

    # Select numeric data types for imputation
    numerical_data = applications.select_dtypes(include=data_types)

    # Initialize the KNN imputer with 2 neighbors
    imputer = KNNImputer(n_neighbors=2)

    # Perform the imputation and transform the data
    imputed_numerical_data = imputer.fit_transform(numerical_data)

    # Return the imputed data as a DataFrame
    return pd.DataFrame(imputed_numerical_data, columns=numerical_data.columns)


def impute_numeric_features(applications):
    """
    Impute missing values in numeric features of the given applications dataset.

    This function first converts specified columns to float type and then applies KNN imputation
    to fill in missing values for numeric features.

    :param applications: A pandas DataFrame representing the applications dataset.
    :type applications: pandas.DataFrame
    :return: A new pandas DataFrame with missing values in numeric features imputed.
    :rtype: pandas.DataFrame
    """
    # Specify the columns that need to be converted to float
    columns_to_transform = [1]

    # Define data types to be considered for imputation
    data_types_to_impute = ['float64', 'int64']

    # Convert specified columns to float type
    applications = transform_columns_to_float(applications, columns_to_transform)

    # Impute missing values using KNN imputation
    imputed_numerical_applications = impute_with_knn(applications, data_types_to_impute)

    return imputed_numerical_applications


def feature_selection(applications_ready, feature):
    """
    Perform feature selection on the given dataset using a RandomForestClassifier.

    :param applications_ready: A pandas DataFrame containing the dataset.
    :param feature: A pandas DataFrame containing the target feature.
    :return: Selected features as a pandas Index.
    """
    print("Pre-processor: recommended features using feature selection")

    # Create a copy of the applications DataFrame to avoid modifying the original
    applications_ready_copy = applications_ready.copy()

    # Ensure all column names are integers for consistent indexing
    applications_ready_copy.columns = applications_ready_copy.columns.astype(int)

    # Impute missing numeric features
    applications_ready_copy = impute_numeric_features(applications_ready_copy)

    # Concatenate the feature DataFrame with the applications DataFrame
    applications_ready_copy = pd.concat([applications_ready_copy, feature], axis=1)

    # Replace '-' with 0 and '+' with 1 in the target column
    applications_ready_copy[15] = applications_ready_copy[15].replace("-", 0)
    applications_ready_copy[15] = applications_ready_copy[15].replace("+", 1)

    # Split the data into training and testing sets
    x_train, y_train = train_test_data_split(applications_ready_copy, 0.31, 15)

    # Initialize the RandomForestClassifier with specified parameters
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5,random_state=42)

    # Fit the model to the training data
    rf.fit(x_train, y_train)

    # Select features based on importance weights from the trained model
    model = SelectFromModel(rf, prefit=True)
    features_bool = model.get_support()

    # Extract the selected feature names
    features = x_train.columns[features_bool]

    # Output the selected features
    print("Pre-processor: feature selection:")
    print(features)

    return features


def normalization(x_train, x_test):
    """
    Normalize the input data for pre-processing using L2 normalization.

    This function applies normalization to the training and testing datasets to ensure that each feature contributes equally to the distance calculations in machine learning algorithms.

    :param x_train: The training data to be normalized.
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param x_test: The test data to be normalized.
    :type x_test: numpy.ndarray or pandas.DataFrame
    :return: A tuple containing the normalized training data and test data.
    :rtype: tuple
    """
    # Log the start of the normalization process
    print("Pre-processor: normalization")

    # Normalize the training and testing datasets
    normalized_x_train = norm.fit_transform(x_train)
    normalized_x_test = norm.transform(x_test)

    # Return the normalized datasets
    return normalized_x_train, normalized_x_test


def standardization(x_train, x_test):
    """
    Standardize the given training and testing data.

    This function applies standardization to ensure that the data has a mean of 0 and a standard deviation of 1,
    which can improve the performance of many machine learning algorithms.
    :param x_train: A numpy array or pandas DataFrame representing the training data.
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param x_test: A numpy array or pandas DataFrame representing the testing data.
    :type x_test: numpy.ndarray or pandas.DataFrame
    :return: A tuple containing the standardized training data and standardized testing data.
    :rtype: tuple
    """
    # Log the start of the standardization process
    print("Pre-processor: standardization")

    # Standardize the training data and fit the scaler
    standardized_x_train = scaler.fit_transform(x_train)

    # Standardize the testing data using the fitted scaler
    standardized_x_test = scaler.transform(x_test)

    # Return the standardized training and testing data
    return standardized_x_train, standardized_x_test

def train_and_evaluate(x_train, x_test, y_train, y_test, hyperparameters):
    """
    Train and evaluate a Logistic Regression model.

    This function trains a Logistic Regression model using the provided training data and hyperparameters,
    then evaluates its performance on the test data.

    :param x_train: Training feature data.
    :param x_test: Testing feature data.
    :param y_train: Training target data.
    :param y_test: Testing target data.
    :param hyperparameters: Dictionary containing hyperparameters for the Logistic Regression model.
    """
    # Initialize the Logistic Regression model with the provided hyperparameters
    logreg = LogisticRegression(tol=hyperparameters['tol'],
                                max_iter=hyperparameters['max_iter'],
                                solver=hyperparameters['solver'],
                                random_state=42)

    # Fit the model to the training data
    logreg.fit(x_train, y_train)
    # Predict the target values for the test data
    y_pred = logreg.predict(x_test)

    # Calculate the accuracy score and confusion matrix
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Print the evaluation metrics
    print("Accuracy Score:", score)
    print("Confusion Matrix: \n", matrix)

    # Calculate and print training and testing accuracy
    train_accuracy = logreg.score(x_train, y_train)
    test_accuracy = logreg.score(x_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


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
    # Identify the most frequent value in the specified column
    most_frequent = data[column].value_counts().index[0]

    # Fill null values with the most frequent value
    return data.fillna(most_frequent)


def fill_categorical_na_with_most_frequent(applications):
    """
    Fill missing values in categorical features with the most frequent values.

    This method takes in a DataFrame (applications) and fills the missing values in categorical features
    with the most frequent values.

    :param applications: The input DataFrame containing the applications data
    :return: A new DataFrame with the missing values filled with the most frequent values
    """
    print("Pre-processor: impute categorical features with most frequent values")

    # Select categorical data
    categorical_data = applications.select_dtypes(include=['object'])

    # Fill missing values in each categorical column
    for column in categorical_data.columns:
        categorical_data = fill_na_with_most_frequent(categorical_data, column)

    return categorical_data


def encode_categorical_fit(categorical_data):
    """
    Fit the OneHotEncoder to the categorical data.

    :param categorical_data: The categorical data to fit the encoder.
    :return: The fitted encoder.
    """
    return enc.fit(categorical_data)


def encode_categorical_values(categorical_data):
    """
    Encode categorical data using the fitted OneHotEncoder.

    :param categorical_data: The categorical data to encode.
    :return: A DataFrame containing the encoded categorical data.
    """
    return pd.DataFrame(enc.transform(categorical_data).toarray())


def remove_outliers(applications_ready):
    """
    Remove outliers from the specified columns in the dataset.

    :param applications_ready: The DataFrame containing the applications data.
    :return: The DataFrame with outliers removed.
    """
    print("Pre-processor: remove the outliers")

    # Remove outliers based on specified conditions
    applications_ready.drop(applications_ready[applications_ready[3] > 50].index, inplace=True)
    applications_ready.drop(applications_ready[applications_ready[4] > 60000].index, inplace=True)

    return applications_ready


def train_test_data_split(df, test_size, target):
    """
    Split the given dataframe into train and test data.

    :param df: The input dataframe.
    :param test_size: The proportion of data to allocate for the test set (0-1).
    :param target: Target feature.
    :return: A tuple containing x_train, y_train, x_test, y_test.
    """
    print("Pre-processor: data train & test split ")

    # Separate features and target
    X = df.drop([target], axis=1)
    y = df[target]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return x_train, y_train, x_test, y_test


def mark_missing(applications_original):
    """
    Mark missing values in the dataset as NaN.

    :param applications_original: The original DataFrame containing the applications data.
    :return: The DataFrame with missing values marked as NaN.
    """
    print("Pre-processor: mark NaN")

    # Replace '?' with NaN
    applications = applications_original.replace("?", np.NaN)

    return applications


def request_rescale(x):
    """
    Rescale the input data using a predefined scaler.

    :param x: The input data to be rescaled.
    :return: The rescaled input data.
    """
    return scaler.transform(x)


def request_norm(x):
    """
    Normalize the input data using a predefined normalizer.

    :param x: The input data to be normalized.
    :return: The normalized input data.
    """
    return norm.transform(x)


def request_encoded(request_json):
    """
    Encode the request data using the predefined schema and one-hot encoding.

    :param request_json: The input request data in JSON format.
    :return: A DataFrame containing the encoded request data.
    """
    r = cca_model.CreditSchema().load(request_json)

    # Create a DataFrame from the request data
    request_df = pd.DataFrame(
        [[r.p0, r.p1, r.p2, r.p3, r.p4, r.p5, r.p6, r.p7, r.p8, r.p9, r.p10, r.p11, r.p12, r.p13, r.p14]]
    )

    # Convert specified columns to float
    request_df[1] = request_df[1].astype(float)
    request_df[13] = request_df[13].astype(float)

    # Encode categorical columns using one-hot encoding
    return pd.get_dummies(request_df, columns=[0, 3, 4, 5, 6, 8, 9, 11, 12])


def load_data():
    """
    Load data from file "crx.data" using the resource_filename method from the "app" package.

    :return: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(resource_filename("app", "data/crx.data"), header=None)


def grid_searching(x_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for a Logistic Regression model.

    :param x_train: Training feature data.
    :param y_train: Training target data.
    :return: A dictionary containing the best hyperparameters.
    """
    print("Pre-processor: Hyperparameter tuning")

    # Initialize the Logistic Regression model
    logreg = LogisticRegression(max_iter= 100, solver='saga', tol= 0.01, random_state=42)
    logreg.fit(x_train, y_train)

    # Define the grid of values for tol, max_iter, and solver
    tol = [0.01, 0.001, 0.0001]
    max_iter = [100, 150, 200]
    solver = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']

    # Create a dictionary where tol, max_iter, and solver are keys and the lists of their values are corresponding values
    param_grid = dict(tol=tol, max_iter=max_iter, solver=solver)

    # Initialize GridSearchCV with the defined parameter grid
    grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=3, scoring='accuracy')

    # Fit data to grid_model
    grid_model_result = grid_model.fit(x_train, y_train)

    # Get the best score and parameters
    best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_

    # Print the best score and parameters
    print("best score: %f with test data: %f using: %s" % (best_score, 1, best_params))

    return dict(tol=best_params['tol'], max_iter=best_params['max_iter'], solver=best_params['solver'])
