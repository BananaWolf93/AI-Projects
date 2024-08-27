from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import datetime as dt
import matplotlib.dates as mdates


# Placeholder for dataset URL or file path
# Replace the following with the actual dataset URL or local file path

# Example of using a direct URL
# coffee_data = "https://www.kaggle.com/datasets/ihelon/coffee-sales"

# Example of using a local file path
# coffee_data = "data/coffee_sales.csv"

coffee_data = "REPLACE_THIS_WITH_ACTUAL_URL_OR_LOCAL_PATH"
df = pd.read_csv(coffee_data)

#def coffee_sales():
   # colors = {'Latte':'orange', 'Hot Chocolate':'brown', 'Americano': 'grey', 'Americano with Milk':'yellow', 'Cocoa':'purple', 'Cortado':'blue', 'Espresso':'black', 'Cappuccino':'pink'}
   # df['color'] = df['coffee_name'].map(colors)
    #plt.bar(df.date, df.money, color = df.color)
    #plt.xlabel("Dates")
    #plt.ylabel("Money Spent")
    #plt.title("Coffee Vending Machine Sales")
    #plt.show()

df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure 'date' is in datetime format
df['date_ordinal'] = df['date'].map(dt.datetime.toordinal)  # Drop any rows where 'date' conversion failed

# Drop rows with NaT in 'date'
df = df.dropna(subset=['date_ordinal'])

# Prepare the feature and target variables
X = df[['date_ordinal']]  # Feature variable as 2D array
y = df['money']  # Target variable

def regression_model():
    # Split the data into training and testing sets
    msk = np.random.rand(len(df)) < 0.8

    X_train = X[msk]
    X_test = X[~msk]
    y_train = y[msk]
    y_test = y[~msk]

    colors = {'Latte':'orange', 'Hot Chocolate':'brown', 'Americano': 'grey', 'Americano with Milk':'yellow', 'Cocoa':'purple', 'Cortado':'blue', 'Espresso':'black', 'Cappuccino':'pink'}
    df['color'] = df['coffee_name'].map(colors)

      # Apply colors to training data only for the scatter plot
    X_train_color = df.loc[msk, 'color']
    coffee_names = df['coffee_name'].unique()  # Get unique coffee names


    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # Print the coefficients:
    print('coefficients: ', regr.coef_)
    print('intercept: ', regr.intercept_)

        # Convert ordinal dates back to datetime for plotting
    X_train_dates = [dt.datetime.fromordinal(int(d)) for d in X_train['date_ordinal']]
    X_test_dates = [dt.datetime.fromordinal(int(d)) for d in X_test['date_ordinal']]

    plt.bar(X_train_dates, y_train, color=X_train_color, label='Training data')
    plt.plot(X_test_dates, regr.predict(X_test), color='red', linewidth=2, label='Regression Line')

    # Extend the date range for future predictions
    future_dates = pd.date_range(start=df['date'].max(), periods=180).to_pydatetime().tolist()
    future_ordinal_dates = [d.toordinal() for d in future_dates]
    future_predictions = regr.predict(pd.DataFrame(future_ordinal_dates, columns=['date_ordinal']))

    # Plot future predictions
    plt.plot(future_dates, future_predictions, color='blue', linestyle='--', label='Future Predictions')

    plt.xlabel("Dates")
    plt.ylabel("Money Spent")
    plt.title("Coffee Vending Machine Sales")

    for coffee, color in colors.items():
        plt.scatter([], [], color=color, label=coffee)

    plt.legend()

        # Format the x-axis to show dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # Rotate date labels to fit

    plt.show()

    prediction = regr.predict(X_test)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - y_test)))

regression_model()