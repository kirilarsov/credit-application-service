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
    assert b'{"matrix":"[[9 0]\\n [0 5]]","score":1.0}\n' in response.data



def test_credit_request_case_approved1(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 36.75, "p2": 0.125, "p3": "y", "p4": "p", "p5": "c", "p6": "v", "p7": 1.5, "p8": "f",
        "p9": "f", "p10": "0", "p11": "t", "p12": "g", "p13": "00232", "p14": 113}))
    assert response.status_code == 201
    assert b"APPROVED" in response.data


def test_credit_request_case_declined1(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 0, "p2": 1.5, "p3": "u", "p4": "g", "p5": "ff", "p6": "ff", "p7": 0, "p8": "f",
        "p9": "t", "p10": "02", "p11": "t", "p12": "g", "p13": "00200", "p14": 105}))

    assert b"DECLINED" in response.data


def test_credit_request_case_declined2(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 21.08, "p2": 5, "p3": "y", "p4": "p", "p5": "ff", "p6": "ff", "p7": 0, "p8": "f", "p9": "f",
        "p10": "0", "p11": "f", "p12": "g", "p13": "00000", "p14": 0
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
        "p0": "b", "p1": 24.33, "p2": 6.625, "p3": "y", "p4": "p", "p5": "d", "p6": "v", "p7": 5.5, "p8": "t",
        "p9": "f", "p10": "0", "p11": "t", "p12": "s", "p13": "00100", "p14": 0}))
    assert response.status_code == 201
    assert b"status\":\"APPROVED" in response.data
