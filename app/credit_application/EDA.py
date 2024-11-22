from app.credit_application import pre_processor
import pandas as pd
import matplotlib.pyplot as plt

# Load data set
applications_original = pre_processor.load_data()

print("EDA: head")
print(applications_original.head())

print("EDA: info")
print(applications_original.info())

print("EDA: class (im)balance")
print(applications_original[15].value_counts(normalize=False))

# # Drop the features 11 and 13, features like DriversLicense and ZipCode are not as important as the other features in the dataset for predicting credit card approvals
applications_original = applications_original.drop(columns=[11, 13], axis=1)
target_feature = 13

# Based on the above outputs missing values have '?' char, marking them with NaN
applications = pre_processor.mark_missing(applications_original)

print("EDA: missing values report")
print(applications.isnull().any())

# Determine recommended features
recommended_features = pre_processor.feature_selection(applications, applications_original[15])

# Before moving to the next step we need to do imputation of all missing values for the numeric features
imputed_numerical_applications = pre_processor.impute_numeric_features(applications)

# Investigate for outliers
# plt.scatter(imputed_numerical_applications[2], imputed_numerical_applications[3])
# plt.show()
# plt.scatter(imputed_numerical_applications[3], imputed_numerical_applications[4])
# plt.show()
# plt.scatter(imputed_numerical_applications[4], imputed_numerical_applications[4])
# plt.show()

# Remove the outliers
# imputed_numerical_applications = pre_processor.remove_outliers(imputed_numerical_applications)

# Impute canonical features
canonical_applications = pre_processor.fill_categorical_na_with_most_frequent(applications)

# Merge the numerical, canonical into one df
applications_ready = pd.concat([imputed_numerical_applications, canonical_applications], axis=1)
print("EDA: missing values report again")
print(applications_ready.isnull().any())


applications_ready.columns = range(target_feature+1)
print(applications_ready.head())
applications_ready[target_feature] = applications_ready[target_feature].replace("-", 0)
applications_ready[target_feature] = applications_ready[target_feature].replace("+", 1)

# Split data for train and test
x_train, y_train, x_test, y_test = pre_processor.train_test_data_split(df=applications_ready, test_size=0.31,
                                                                       target=target_feature)
pre_processor.encode_categorical_fit(x_train)
x_train = pre_processor.encode_categorical_values(x_train)
x_test = pre_processor.encode_categorical_values(x_test)



# Normalization. Is more useful in regression than classification
x_train, x_test =pre_processor.normalization(x_train, x_test)

# Standardization, It is more useful in classification than regression
# x_train, x_test = pre_processor.standardization(x_train, x_test)

# Hyperparameter tuning
from sklearn.preprocessing import OneHotEncoder

enc1 = OneHotEncoder(handle_unknown='ignore')
X = applications_ready.drop([target_feature], axis=1)
y = applications_ready[target_feature]

enc1.fit(X)
X = pd.DataFrame(enc1.transform(X).toarray())
hyperparameters = pre_processor.grid_searching(X,y)

# Train and evaluate
pre_processor.train_and_evaluate(x_train, x_test, y_train, y_test, hyperparameters)
