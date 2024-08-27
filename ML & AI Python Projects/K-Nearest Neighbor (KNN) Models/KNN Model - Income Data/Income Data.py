import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn


# Placeholder for dataset URL or file path
# Replace the following with the actual dataset URL or local file path

# Example of using a direct URL
# temp_data = "https://www.kaggle.com/datasets/amirhosseinmirzaie/americancitizenincome/data"

# Example of using a local file path
# temp_data = "income.csv"

income_ds = pd.read_csv("REPLACE_THIS_WITH_ACTUAL_URL_OR_LOCAL_PATH")

# Handle missing or unknown values by replacing "?" with NaN, then dropping them
income_ds.replace('?', pd.NA, inplace=True)
income_ds.dropna(inplace=True)

# Convert income to numeric
income_ds['income'] = income_ds['income'].map({'>50K': 1, '<=50K': 0})

# Convert categorical variables to numeric using one-hot encoding
income_encoded = pd.get_dummies(income_ds, drop_first=True)

# Display the first few rows of the encoded dataset
income_encoded.head()

# To use Sckit-learn, we need to first convert the pandas dataframe to a numpy array.:
Excluded_headers = ['age', 'fnlwgt', 'education.num']
X = income_ds.drop(columns=Excluded_headers)
y = income_ds["income"].values

# # Now, we need to standardize the data. This is a good practice to do especially for KNN models.:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

print(income_ds[0:0])

# The following 3 lines are to convert categorical values to a numeric format in order to use them in a KNN model.
cat_feat = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
column_transformer = ColumnTransformer(transformers=[("onehot", OneHotEncoder(), cat_feat)], remainder = "passthrough")
X_transformed = column_transformer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=4)

print("Training set: ", X_train.shape, Y_train.shape)
print("Testing set: ", X_test.shape, Y_test.shape)

# Next, we begint to build the classifier for K-Nearest Neighbor:
from sklearn.neighbors import KNeighborsClassifier

# Training the KNN model:
k = 3
neighbor = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
neighbor

yhat = neighbor.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Training test accuracy is: ", metrics.accuracy_score(Y_train, neighbor.predict(X_train)))
print("Testing test accuracy is: ", metrics.accuracy_score(Y_test, yhat))


# Training K and finding the right K value:

ks = 10
mean_acc = np.zeros((ks-1))
std_acc = np.zeros((ks-1))

for k in range(1,ks):
    neighbor = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    yhat=neighbor.predict(X_test)
    mean_acc[k-1] = metrics.accuracy_score(Y_test, yhat)

    std_acc[k-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

# Find the best K value
best_k = np.argmax(mean_acc) + 1
best_accuracy = mean_acc[best_k-1]

print("Best K value:", best_k)
print("Best accuracy:", best_accuracy)

# After finding the best K value
best_k = np.argmax(mean_acc) + 1

# Train the final model with the best K value
final_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, Y_train)

# Make predictions using the final model
final_predictions = final_model.predict(X_test)

# Evaluate the final model
final_accuracy = metrics.accuracy_score(Y_test, final_predictions)
print("Final accuracy with best K ({}): {}".format(best_k, final_accuracy))

# Plotting the data with matplotlib:

import matplotlib.pyplot as plt
plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

# The output plot shows that out of the K values defined up to 10, value 3 is the one with the greatest accuracy. With this, I can now continue to plot the predicted data alongside the data from the dataset as well to paint the full picture and completing this project.