from credit_application import util
from credit_application import model as cca_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def generate_model():
    applications = load_data()

    applications_ready = util.data_imputation(applications)

    applications_train_imputed, applications_test_imputed = train_test_split(applications_ready, test_size=0.02, random_state=42)

    applications_train_encoded = util.data_encode(applications_train_imputed)

    X_train, y_train = (
        applications_train_encoded.drop([15], axis=1).values,
        applications_train_encoded[15].values
    )
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg, applications_train_encoded


def predict(request, model, applications_train_encoded):
    request_encoded = util.data_request_encode(request.get_json())
    request_encoded = request_encoded.reindex(
        columns=applications_train_encoded.columns, fill_value=0
    )
    request_test = request_encoded.drop([15], axis=1).values
    y_pred = model.predict(request_test)
    status = cca_model.ApplicationStatus.APPROVED
    if y_pred[0] == '-':
        status = cca_model.ApplicationStatus.DECLINED

    return cca_model.CreditResponse(status=status.name)


def model_info():
    cc_apps = load_data()

    applications_ready = util.data_imputation(cc_apps)

    applications_train_imputed, applications_test_imputed = train_test_split(applications_ready, test_size=0.3, random_state=42)

    applications_train_encoded = util.data_encode(applications_train_imputed)
    applications_test_encoded = util.data_encode(applications_test_imputed)
    applications_test_encoded = applications_test_encoded.reindex(
        columns=applications_train_encoded.columns, fill_value=0
    )

    X_train, y_train = (
        applications_train_encoded.drop([15], axis=1).values,
        applications_train_encoded[15].values
    )
    X_test, y_test = (
        applications_test_encoded.drop([15], axis=1).values,
        applications_test_encoded[15].values
    )

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    score = logreg.score(X_train, y_train)
    matrix = confusion_matrix(y_test, y_pred)

    return cca_model.ModelInfo(score=score, matrix=matrix)


def load_data():
    return util.load_data("data/crx.data")
