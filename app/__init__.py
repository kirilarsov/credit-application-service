from flask import Flask, request
from sklearn.preprocessing import StandardScaler

from app.credit_application import model
from app.credit_application import ca
import warnings
# Initializations
scaler = StandardScaler()


warnings.filterwarnings('ignore')
app = Flask(__name__)

applications_ready = ca.pre_process()
logistic_regression, trained_data, x_train, y_train, x_test, y_test = ca.generate_model(applications_ready, scaler)
score, matrix = ca.model_info(logistic_regression, x_train, y_train, x_test, y_test)

def create_app():
    # Create the Flask application
    return app

@app.route("/info")
def model_info():
    response = model.ModelInfoSchema()
    return response.dump(model.ModelInfo(score, matrix)), 200


@app.route('/creditApplicationRequest', methods=['POST'])
def credit_application_request():
    result = ca.predict(request, logistic_regression, trained_data, scaler, x_train, y_train, x_test)
    response = model.CreditResponseSchema()
    return response.dump(result), 201
