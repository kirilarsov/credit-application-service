from app.credit_application import model as cca_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from pkg_resources import resource_filename
from app.credit_application import pre_processor
import pandas as pd


def generate_model(applications_ready):
    """
    :param applications_ready: List of pre-processed applications
    :return: Tuple containing the trained LogisticRegression classifier and the applications used for training
    """

    # Split data for train and test
    x_train, y_train, x_test, y_test, applications_train = pre_processor.train_test_data_split(applications_ready, 0.33)

    # Instantiate a LogisticRegression classifier with default parameter values
    model = LogisticRegression()
    # Fit logreg to the train set
    model.fit(x_train, y_train)
    return model, applications_train


def pre_process():
    """
    Pre-processes the data by loading it, marking missing values with NaN, imputing missing
    numeric values, filling categorical missing values with the most frequent value, merging
    all features into one DataFrame, and removing outliers.

    :return: The pre-processed DataFrame.
    """
    # Load data
    applications_original = load_data()
    # Based on the above outputs missing values have '?' char, marking them with NaN
    applications = pre_processor.mark_missing(applications_original)
    # Before moving to the next step we need to do imputation of all missing values for the numeric features
    imputed_numerical_applications = pre_processor.impute_numeric_features(applications)
    # Impute canonical features
    canonical_applications = pre_processor.fill_categorical_na_with_most_frequent(applications)
    # Merge the numerical, canonical and the target into one df
    applications_ready = pd.concat([imputed_numerical_applications, canonical_applications], axis=1)
    applications_ready = pd.concat([applications_ready, applications[15]], axis=1)
    # Remove the outliers
    applications_ready = pre_processor.remove_outliers(applications_ready)
    return applications_ready


def predict(request, model, applications_train):
    """
    :param request: The request object containing the data to be predicted.
    :param model: The trained model used for prediction.
    :param applications_train: The training data used to train the model.
    :return: A CreditResponse object representing the predicted credit status.

    This method takes in a request object, a trained model, and the training data. It first encodes and serializes the request data. Then, it aligns the columns of the encoded request data
    * with the columns of the training data. After that, it drops the label column from the aligned request data and rescales it. The rescaled request data is then used to predict the credit
    * status using the trained model. Finally, it determines the status (approved or declined) based on the prediction and returns a CreditResponse object representing the predicted credit
    * status.
    """

    # Serialize request data and encode it
    request_encoded = pre_processor.request_encoded(request.get_json())

    # Reindex the columns of the test set aligning with the train set
    request_encoded = request_encoded.reindex(
        columns=applications_train.columns, fill_value=0
    )

    # Drop the label from the reindex test set
    request_test = request_encoded.drop([15], axis=1).values

    # Rescale the request test set
    rescaled_request_test = pre_processor.request_rescale(request_test)

    # Use logreg to predict instances from the test set and store it
    y_pred = model.predict(rescaled_request_test)

    # If the out label is '-' set declined application
    status = cca_model.ApplicationStatus.APPROVED
    if y_pred[0] == 0:
        status = cca_model.ApplicationStatus.DECLINED

    print("+++")
    print(model.score())


    return cca_model.CreditResponse(status=status.name)


def model_info(applications_ready):
    """
    Function: model_info

    This function calculates and returns the score and confusion matrix for a logistic regression model.

    :param applications_ready: The input data for the model.
    :return: score - The accuracy score of the model
             matrix - The confusion matrix of the model
    """
    # Split data for train and test
    x_train, y_train, x_test, y_test, applications_train = pre_processor.train_test_data_split(applications_ready, 0.02)

    # Standardization
    x_train, x_test = pre_processor.standardization(x_train, x_test)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    score = logreg.score(x_train, y_train)
    matrix = confusion_matrix(y_test, y_pred)

    return score, matrix


def load_data():
    """
    Loads data from file "crx.data" using the resource_filename method from the "app" package.

    :return: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(resource_filename("app", "data/crx.data"), header=None)
