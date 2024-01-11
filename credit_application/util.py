import pandas as pd
import numpy as np
from credit_application import model as cca_model


def load_data(data_location):
    return pd.read_csv(data_location, header=None)


def data_imputation(applications):
    applications_train_nans_replaced = applications.replace("?", np.NaN)

    val0 = applications[0].value_counts().index[0]
    val1 = pd.to_numeric(applications[1], errors='coerce').mean()
    val2 = pd.to_numeric(applications[2], errors='coerce').mean()
    val3 = applications[3].value_counts().index[0]
    val4 = applications[4].value_counts().index[0]
    val5 = applications[5].value_counts().index[0]
    val6 = applications[6].value_counts().index[0]
    val7 = pd.to_numeric(applications[7], errors='coerce').mean()
    val8 = applications[8].value_counts().index[0]
    val9 = applications[9].value_counts().index[0]
    val10 = pd.to_numeric(applications[10], errors='coerce').mean()
    val11 = applications[11].value_counts().index[0]
    val12 = applications[12].value_counts().index[0]
    val13 = pd.to_numeric(applications[13], errors='coerce').mean()
    val14 = pd.to_numeric(applications[14], errors='coerce').mean()

    values = {0: val0,
              1: val1,
              2: val2,
              3: val3,
              4: val4,
              5: val5,
              6: val6,
              7: val7,
              8: val8,
              9: val9,
              10: val10,
              11: val11,
              12: val12,
              13: val13,
              14: val14}

    return applications_train_nans_replaced.fillna(value=values)


def data_encode(data):
    return pd.get_dummies(data,columns=[0,3,4,5,6,8,9,11,12,13])


def data_request_encode(request_json):
    r = cca_model.CreditSchema().load(request_json)
    request_df = pd.DataFrame(
        [[r.p0, r.p1, r.p2, r.p3, r.p4, r.p5, r.p6, r.p7, r.p8, r.p9, r.p10, r.p11, r.p12, r.p13, r.p14]])
    return pd.get_dummies(request_df,columns=[0,3,4,5,6,8,9,11,12,13])
