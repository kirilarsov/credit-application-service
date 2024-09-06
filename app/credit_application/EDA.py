from pkg_resources import resource_filename
from app.credit_application import pre_processor
import pandas as pd
import matplotlib.pyplot as plt

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
plt.scatter(imputed_numerical_applications[3], imputed_numerical_applications[4])
plt.show()
plt.scatter(imputed_numerical_applications[4], imputed_numerical_applications[5])
plt.show()

# Impute canonical features
canonical_applications = pre_processor.fill_categorical_na_with_most_frequent(applications)

# Merge the numerical, canonical and the target into one df
applications_ready = pd.concat([imputed_numerical_applications, canonical_applications], axis=1)
applications_ready = pd.concat([applications_ready, applications[15]], axis=1)

# Remove the outliers
applications_ready = pre_processor.remove_outliers(applications_ready)

# Determine recommended features
columns = pre_processor.feature_selection(applications_ready)

# Split data for train and test
x_train, y_train, x_test, y_test, applications_train = pre_processor.train_test_data_split(applications_ready, 0.33)

# Normalization
# x_train, x_test =normalization(x_train, x_test)

# Standardization
x_train, x_test = pre_processor.standardization(x_train, x_test)

# Train and evaluate
pre_processor.train_and_evaluate(x_train, x_test, y_train, y_test)
