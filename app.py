from flask import Flask, request
from credit_application import ca, model
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

logistic_regression, trained_data = ca.generate_model()
@app.route("/info")
def model_info():
    result = ca.model_info()
    response = model.ModelInfoSchema()
    return response.dump(result), 200


@app.route('/creditApplicationRequest', methods=['POST'])
def credit_application_request():
    result = ca.predict(request, logistic_regression, trained_data)
    response = model.CreditResponseSchema()
    return response.dump(result), 201

if __name__ == "__main__":
    app.run()