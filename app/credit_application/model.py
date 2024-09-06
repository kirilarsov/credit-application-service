import enum
from marshmallow import Schema, fields, post_load


class CreditRequest:
    def __init__(self, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14) -> None:
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.p10 = p10
        self.p11 = p11
        self.p12 = p12
        self.p13 = p13
        self.p14 = p14


class CreditSchema(Schema):
    p0 = fields.Str()
    p1 = fields.Decimal()
    p2 = fields.Decimal()
    p3 = fields.Str()
    p4 = fields.Str()
    p5 = fields.Str()
    p6 = fields.Str()
    p7 = fields.Decimal()
    p8 = fields.Str()
    p9 = fields.Str()
    p10 = fields.Str()
    p11 = fields.Str()
    p12 = fields.Str()
    p13 = fields.Str()
    p14 = fields.Decimal()

    @post_load
    def make_request(self, data, **kwargs) -> CreditRequest:
        return CreditRequest(**data)


class CreditResponse:
    def __init__(self, status) -> None:
        self.status = status


class CreditResponseSchema(Schema):
    status = fields.Str()

    @post_load
    def make_response(self, data, **kwargs) -> CreditResponse:
        return CreditResponse(**data)


class ModelInfo:
    def __init__(self, score, matrix) -> None:
        self.score = score
        self.matrix = matrix


class ModelInfoSchema(Schema):
    score = fields.Float()
    matrix = fields.Str()

    @post_load
    def make_response(self, data, **kwargs) -> ModelInfo:
        return ModelInfo(**data)


class ApplicationStatus(enum.Enum):
    APPROVED = 'APPROVED'
    DECLINED = 'DECLINED'
