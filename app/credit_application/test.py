from pkg_resources import resource_filename
from app.credit_application import pre_processor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# Load data set
applications_original = pd.read_csv(resource_filename("app", "data/crx.data"), header=None)

print("EDA: head")
print(applications_original.head())

print("EDA: info")
print(applications_original.info())

print("EDA: class (im)balance")
print(applications_original[15].value_counts(normalize=False))

# Based on the above outputs missing values have '?' char, marking them with NaN
applications = pre_processor.mark_missing(applications_original)

print("EDA: missing values report")
print(applications.isnull().any())

# Before moving to the next step we need to do imputation of all missing values for the numeric features
imputed_numerical_applications = pre_processor.impute_numeric_features(applications)

print("EDA: detecting outliers")


# Impute canonical features
canonical_applications = pre_processor.fill_categorical_na_with_most_frequent(applications)

print('1')
print(canonical_applications)


canonical = pre_processor.encode_categorical_values(canonical_applications)
print('2')
print(canonical)
