from credit_application import util
from credit_application import model as cca_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def generate_model():
    # Load data
    applications = load_data()

    # Fill missing values
    applications_ready = util.data_imputation(applications)

    # Split data
    applications_train_imputed, applications_test_imputed = train_test_split(applications_ready, test_size=0.02,
                                                                             random_state=42)
    # Show test data
    #print(applications_test_imputed)

    # Encode string values
    applications_train_encoded = util.data_encode(applications_train_imputed)

    # Segregate features and labels into separate variables
    x_train, y_train = (
        applications_train_encoded.drop([15], axis=1).values,
        applications_train_encoded[15].values
    )

    # Instantiate MinMaxScaler and use it to rescale x_train and x_test
    rescaled_x_train = rescale_train(x_train)

    # Instantiate a LogisticRegression classifier with default parameter values
    logreg = LogisticRegression()

    # Fit logreg to the train set
    logreg.fit(rescaled_x_train, y_train)
    return logreg, applications_train_encoded


def predict(request, model, applications_train_encoded):
    # Serialize request data and encode it
    request_encoded = util.data_request_encode(request.get_json())

    # Reindex the columns of the test set aligning with the train set
    request_encoded = request_encoded.reindex(
        columns=applications_train_encoded.columns, fill_value=0
    )

    # Drop the label from the rescaled test set
    request_test = request_encoded.drop([15], axis=1).values

    # Rescale the request test set
    rescaled_request_test = rescale_test(request_test)

    # Use logreg to predict instances from the test set and store it
    y_pred = model.predict(rescaled_request_test)

    # If the out label is '-' set declined application
    status = cca_model.ApplicationStatus.APPROVED
    if y_pred[0] == '-':
        status = cca_model.ApplicationStatus.DECLINED

    return cca_model.CreditResponse(status=status.name)


def model_info():
    cc_apps = load_data()

    applications_ready = util.data_imputation(cc_apps)

    applications_train_imputed, applications_test_imputed = train_test_split(applications_ready, test_size=0.33,
                                                                             random_state=42)

    applications_train_encoded = util.data_encode(applications_train_imputed)
    applications_test_encoded = util.data_encode(applications_test_imputed)
    applications_test_encoded = applications_test_encoded.reindex(
        columns=applications_train_encoded.columns, fill_value=0
    )

    x_train, y_train = (
        applications_train_encoded.drop([15], axis=1).values,
        applications_train_encoded[15].values
    )
    x_test, y_test = (
        applications_test_encoded.drop([15], axis=1).values,
        applications_test_encoded[15].values
    )

    # Instantiate MinMaxScaler and use it to rescale x_train and x_test
    scale = MinMaxScaler(feature_range=(0, 1))
    rescaled_x_train = scale.fit_transform(x_train)
    rescaled_x_test = scale.transform(x_test)

    logreg = LogisticRegression()
    logreg.fit(rescaled_x_train, y_train)
    y_pred = logreg.predict(rescaled_x_test)
    score = logreg.score(rescaled_x_train, y_train)
    matrix = confusion_matrix(y_test, y_pred)

    return cca_model.ModelInfo(score=score, matrix=matrix)


def rescale_train(x):
    return scaler.fit_transform(x)


def rescale_test(x):
    return scaler.transform(x)


def load_data():
    return util.load_data("data/crx.data")
