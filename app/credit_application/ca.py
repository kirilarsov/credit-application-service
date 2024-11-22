import numpy as np

from app.credit_application import domain as cca_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from app.credit_application import pre_processor
import pandas as pd
from sklearn.metrics import accuracy_score

def generate_model(applications_ready):
    """
    Generates and trains a logistic regression model for credit applications.

    Args:
        applications_ready (DataFrame): Preprocessed credit application data

    Returns:
        tuple: Contains:
            - trained LogisticRegression model
            - normalized test features (x_test_normalized)
            - test labels (y_test)
    """
    # Remove the outliers
    applications_ready = pre_processor.remove_outliers(applications_ready)

    # Split data for train and test
    x_train, y_train, x_test, y_test = pre_processor.train_test_data_split(applications_ready, 0.35,15)
    pre_processor.encode_categorical_fit(x_train)
    x_train_encoded = pre_processor.encode_categorical_values(x_train)
    x_test_encoded = pre_processor.encode_categorical_values(x_test)

    x_train_normalized, x_test_normalized =pre_processor.normalization(x_train_encoded, x_test_encoded)

    # Instantiate a LogisticRegression classifier with default parameter values
    model = LogisticRegression(random_state=42)
    # Fit logreg to the train set
    model.fit(x_train_normalized, y_train)
    return model, x_test_normalized, y_test


def pre_process():
    """
    Pre-processes the credit application data through multiple steps to prepare it for model training.

    The function performs the following steps:
    1. Loads the raw data
    2. Marks missing values with NaN
    3. Imputes missing numeric values
    4. Fills missing categorical values
    5. Merges all features into one DataFrame
    6. Converts target variable to binary (0/1)
    7. Removes outliers

    Returns:
        DataFrame: Fully pre-processed dataset ready for model training
    """
    # Load the raw credit application data from source
    applications_original = pre_processor.load_data()

    # Convert missing value indicators ('?') to NaN for proper handling
    applications = pre_processor.mark_missing(applications_original)

    # Impute missing values in numeric features using appropriate methods
    imputed_numerical_applications = pre_processor.impute_numeric_features(applications)

    # Fill missing values in categorical features with most frequent values
    canonical = pre_processor.fill_categorical_na_with_most_frequent(applications)

    # Combine the processed numerical and categorical features into a single DataFrame
    applications_ready = pd.concat([imputed_numerical_applications, canonical], axis=1)

    # Reset column names to numeric indices (0-15)
    applications_ready.columns = range(16)

    # Convert target variable from {'-', '+'} to {0, 1} for binary classification
    applications_ready[15] = applications_ready[15].replace("-", 0)
    applications_ready[15] = applications_ready[15].replace("+", 1)

    # Remove statistical outliers to improve model performance
    applications_ready = pre_processor.remove_outliers(applications_ready)

    return applications_ready




def predict(request, model) :
    """
    Predicts credit application status using a trained model.

    This function processes an incoming credit application request and uses a trained model
    to predict whether the application should be approved or declined.

    Args:
        request: Flask request object containing the application data in JSON format
        model: Trained machine learning model used for prediction

    Returns:
        CreditResponse: Object containing the predicted application status (APPROVED or DECLINED)

    The function follows these steps:
    1. Pre-processes the request data to match training data format
    2. Normalizes/rescales the processed request data
    3. Makes prediction using the trained model
    4. Converts prediction (0/1) to application status (DECLINED/APPROVED)
    5. Returns response with predicted status
    """
    preprocesed_request = pre_process_request_data(request.get_json())

    # Rescale the request test set
    rescaled_request_test = pre_processor.request_norm(preprocesed_request)

    # Use logreg to predict instances from the test set and store it
    y_pred = model.predict(rescaled_request_test)
    # If the out label is '-' set declined application
    status = cca_model.ApplicationStatus.APPROVED
    if y_pred[0] == 0:
        status = cca_model.ApplicationStatus.DECLINED

    return cca_model.CreditResponse(status=status.name)


def model_info(logreg, x_test, y_test):
    """
    Evaluates the performance of a logistic regression model using accuracy score
    and confusion matrix.

    Args:
        logreg: Trained logistic regression model
        x_test: Normalized test features
        y_test: True test labels

    Returns:
        tuple: Contains:
            - score (float): Accuracy score of the model (ratio of correct predictions)
            - matrix (numpy.ndarray): Confusion matrix showing true/false positives/negatives
                                    Shape: 2x2 for binary classification
                                    [[TN, FP],
                                     [FN, TP]]

    Example:
        score, matrix = model_info(trained_model, x_test_normalized, y_test)
        print(f"Model accuracy: {score}")
        print(f"Confusion matrix:\n{matrix}")
    """
    # Generate predictions for test data
    y_pred = logreg.predict(x_test)

    # Calculate accuracy score (ratio of correct predictions)
    score = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    return score, matrix

def pre_process_request_data(request_json):
    """
    Pre-processes credit application request data for model prediction.

    Args:
        request_json: JSON data containing credit application parameters

    Returns:
        DataFrame: Processed application data ready for prediction

    The function performs the following steps:
    1. Deserializes JSON data using CreditSchema
    2. Creates DataFrame from application parameters
    3. Handles missing values
    4. Performs type conversion for numeric columns
    5. Imputes missing values using KNN for numeric features
    6. Fills missing categorical values
    7. Encodes categorical variables
    """
    # Deserialize JSON data using schema
    r = cca_model.CreditSchema().load(request_json)

    # Create DataFrame from application parameters
    applications_original = pd.DataFrame([
        [r.p0, r.p1, r.p2, r.p3, r.p4, r.p5, r.p6, r.p7,
         r.p8, r.p9, r.p10, r.p11, r.p12, r.p13, r.p14]
    ])

    # Set column names as integers
    applications_original.columns = applications_original.columns.astype(int)

    # Convert numeric columns to appropriate data types
    applications_original[1] = applications_original[1].astype(float)
    applications_original[2] = applications_original[2].astype(float)
    applications_original[7] = applications_original[7].astype(float)
    applications_original[10] = applications_original[10].astype(int)
    applications_original[14] = applications_original[14].astype(int)

    # Replace missing value indicators ('?') with NaN
    applications = applications_original.replace("?", np.NaN)

    # Define numeric data types for imputation
    data_types_to_impute = ['float64', 'int64']

    # Impute missing values in numeric features using KNN
    imputed_numerical_applications = pre_processor.impute_with_knn(
        applications,
        data_types_to_impute
    )

    # Fill missing values in categorical features
    canonical = pre_processor.fill_categorical_na_with_most_frequent(applications)

    # Combine numerical and categorical features
    applications_ready = pd.concat([imputed_numerical_applications, canonical], axis=1)

    # Encode categorical variables
    canonical_applications = pre_processor.encode_categorical_values(applications_ready)

    return canonical_applications

