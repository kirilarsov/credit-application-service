from flask import Flask, request

from app.credit_application import domain
from app.credit_application import ca
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize the Flask application
app = Flask(__name__)

# Pre-process credit applications and generate model and test data
applications_ready = ca.pre_process()
model, x_test, y_test = ca.generate_model(applications_ready)

# Get model performance metrics
score, matrix = ca.model_info(model, x_test, y_test)

def create_app():
    """
    Create and return the Flask application instance.
    """
    return app

@app.route("/info")
def model_info():
    """
    Endpoint to retrieve model information including score and matrix.
    Returns a JSON response with model performance metrics.
    """
    # Create a response schema for model information
    response = domain.ModelInfoSchema()

    # Dump the model info into a JSON-compatible format
    return response.dump(domain.ModelInfo(score, matrix)), 200

@app.route('/creditApplicationRequest', methods=['POST'])
def credit_application_request():
    """
    Endpoint to handle credit application requests.
    Expects a POST request with application data.
    Returns a JSON response with the prediction result.
    """
    # Predict the outcome of the credit application using the model
    result = ca.predict(request, model)

    # Create a response schema for the credit application result
    response = domain.CreditResponseSchema()

    # Dump the prediction result into a JSON-compatible format
    return response.dump(result), 201