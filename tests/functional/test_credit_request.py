import pytest
from flask import json
from app import create_app


@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_info(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/info' page is requested (GET)
    THEN check that the response is valid
    """

    response = client.get('/info')
    assert response.status_code == 200
    assert b'{"matrix":"[[126   0]\\n [  0 101]]","score":1.0}\n' in response.data



def test_credit_request_case_approved1(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 46.00, "p2": 4.0, "p3": "u", "p4": "g", "p5": "j", "p6": "j", "p7": 0.000, "p8": "t",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00100", "p14": 960}))
    assert response.status_code == 201
    assert b"APPROVED" in response.data


def test_credit_request_case_declined2(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 46.00, "p2": 4.0, "p3": "u", "p4": "g", "p5": "j", "p6": "j", "p7": 0.000, "p8": "t",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00100", "p14": 960}))

    assert b"DECLINEDds" in response.data


def test_credit_request_case_declined2(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 20.00, "p2": 0.0, "p3": "u", "p4": "g", "p5": "d", "p6": "v", "p7": 0.5, "p8": "f", "p9": "f",
        "p10": "0", "p11": "f", "p12": "g", "p13": "00144", "p14": 0
    }))
    assert response.status_code == 201
    assert b"status\":\"APPROVED" in response.data


def test_credit_request_case_approved2(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 40.83, "p2": 10.000, "p3": "u", "p4": "g", "p5": "q", "p6": "h", "p7": 1.750, "p8": "t",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00029", "p14": 837}))
    assert response.status_code == 201
    assert b"status\":\"APPROVED" in response.data
