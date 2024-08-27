import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Placeholder for dataset URL or file path
# Replace the following with the actual dataset URL or local file path

# Example of using a direct URL
# temp_data = "https://www.kaggle.com/datasets/sevgisarac/temperature-change/data"

# Example of using a local file path
# temp_data = "data/temp.csv"

temp_data = pd.read_csv("REPLACE_THIS_WITH_ACTUAL_URL_OR_LOCAL_PATH")


# Manually create a list of all unique country names from the second column
country_names = dataset.iloc[1:, 1].unique().tolist()  # Skip the first row if it is the header

# The next section defines the initial program's variables to obtain/gather data from the user:
country_name_or_code = input("Please select the country code or the country name to begin: ")

# Validate the country name or code
if country_name_or_code not in country_names:
    print(f"Invalid country name or code. Please select from the following: {country_names}")
else:
    prediction_year = int(input("Next, please select the year you'd like to predict: "))

    # The following line simply creates a range for the years from the dataset since I could not figure out a way to do this by grabbing the header values for the year.
    years = list(range(1961, 2020))  # Exclude the first column which is country names/codes

    # Extract the specific country data for "Temperature change"
    country_data = dataset[(dataset.iloc[:, 1] == country_name_or_code) & (dataset.iloc[:, 5] == 'Temperature change')]

    # Check if the country data exists
    if country_data.empty:
        print(f"No data found for {country_name_or_code} with Temperature change")
    else:
        # Average the values for each year across all months
        annual_data = country_data.iloc[:, 7:].astype(float).mean()  # Assuming the first 7 columns are metadata
        Y = annual_data.tolist()

        # Convert X and Y into pandas DataFrame for easier manipulation
        df = pd.DataFrame({'Year': years, 'Value': Y})

        # Check if the prediction year exists in the dataset
        if prediction_year in df['Year'].values:
            actual_value = df[df['Year'] == prediction_year]['Value'].values[0]
            print(f"Actual value for {prediction_year}: {actual_value}")
        else:
            # Prepare feature and target variables for regression
            X = df[['Year']]
            y = df['Value']

            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the regression model
            regression = linear_model.LinearRegression()
            regression.fit(X_train, Y_train)

            # Print the coefficients
            print('Coefficients: ', regression.coef_)
            print('Intercept: ', regression.intercept_)

            # Predict the values for the test set
            predictions = regression.predict(X_test)

            # Calculate the mean absolute error
            mae = np.mean(np.absolute(predictions - Y_test))
            print("Mean absolute error: %.2f" % mae)

            # Predict the value for the specified prediction year
            prediction = regression.predict(pd.DataFrame({'Year': [prediction_year]}))
            print(f"Predicted value for {prediction_year}: {prediction[0]}")

# Plot the data
            plt.scatter(df['Year'], df['Value'], color='green', label='Actual Data')
            plt.plot(X_test, predictions, color='yellow', linewidth=2, label='Predicted Data')
            plt.xlabel("Year")
            plt.ylabel("Temperature Change in Celsius")

            # Plot future predictions based on the input year
            last_year = max(years)
            future_years = list(range(last_year + 1, prediction_year + 1))
            future_predictions = regression.predict(pd.DataFrame({'Year': future_years}))

            # Plot future predictions
            plt.plot(future_years, future_predictions, color='red', linestyle='--', label='Future Predictions')
            plt.legend()
            plt.title("Predicted Climate Change")
            plt.show()