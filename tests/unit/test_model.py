from app.credit_application import model as cca_model


def test_model_info():
    """
    GIVEN a ModelInfo model
    WHEN a new ModelInfo is created
    THEN check the score and matrix fields are defined correctly
    """
    model_info = cca_model.ModelInfo('1', '2')
    assert model_info.score == '1'
    assert model_info.matrix == '2'


def test_credit_response():
    """
    GIVEN a CreditResponse model
    WHEN a new CreditResponse is created
    THEN check the score and matrix fields are defined correctly
    """
    model_info = cca_model.CreditResponse('1')
    assert model_info.status == '1'
