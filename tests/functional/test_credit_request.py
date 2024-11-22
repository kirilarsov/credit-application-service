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
    assert b'{"matrix":"[[106  28]\\n [  8  99]]","score":0.8506224066390041}\n' in response.data



def test_credit_request_case_1_declined(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 29.50, "p2": 0.58, "p3": "u", "p4": "g", "p5": "w", "p6": "v", "p7": 0.29, "p8": "f",
        "p9": "t", "p10": "01", "p11": "f", "p12": "g", "p13": "00340", "p14": 2803}))

    assert b"DECLINED" in response.data

def test_credit_request_case_2_approved(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 20.25, "p2": 9.96, "p3": "u", "p4": "g", "p5": "e", "p6": "dd", "p7": 0, "p8": "t",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00000", "p14": 0}))

    assert b"APPROVED" in response.data

def test_credit_request_case_3_declined(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 20.75, "p2": 9.54, "p3": "u", "p4": "g", "p5": "i", "p6": "v", "p7": 0.04, "p8": "f",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00200", "p14": 1000}))

    assert b"DECLINED" in response.data

def test_credit_request_case_4_declined(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "a", "p1": 33.25, "p2": 3, "p3": "y", "p4": "p", "p5": "aa", "p6": "v", "p7": 2, "p8": "f",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00180", "p14": 0}))

    assert b"DECLINED" in response.data

def test_credit_request_case_5_false_negative_declined(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 21.25, "p2": 1.5, "p3": "u", "p4": "g", "p5": "w", "p6": "v", "p7": 1.5, "p8": "f",
        "p9": "f", "p10": "0", "p11": "f", "p12": "g", "p13": "00150", "p14": 8}))

    assert b"DECLINED" in response.data

def test_credit_request_case_6_approved(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 60.08, "p2": 14.5, "p3": "u", "p4": "g", "p5": "ff", "p6": "ff", "p7": 18, "p8": "t",
        "p9": "t", "p10": "15", "p11": "t", "p12": "g", "p13": "00000", "p14": 1000}))

    assert b"APPROVED" in response.data

def test_credit_request_case_7_declined(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/creditApplicationRequest' page is requested (GET)
    THEN check that the response is valid
    """

    # Create a test client using the Flask application configured for testing

    response = client.post('/creditApplicationRequest', headers={"Content-Type": "application/json"}, data=json.dumps({
        "p0": "b", "p1": 23.75, "p2": 12, "p3": "u", "p4": "g", "p5": "c", "p6": "v", "p7": 2.085, "p8": "f",
        "p9": "f", "p10": "0", "p11": "f", "p12": "s", "p13": "00080", "p14": 0}))

    assert b"DECLINED" in response.data