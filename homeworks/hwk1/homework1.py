# Numerical/scientific computing packages.
import numpy as np
import scipy
import copy
import pandas as pd
import seaborn as sns
import csv


# Machine learning package.
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_error,
)


# Useful for saving our models.
import pickle

# Plotting packages.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

# Sets the matplotlib backend for the notebook.
# sets the backend of matplotlib to the 'inline' backend:
# With this backend, the output of plotting commands is
# displayed inline within the Jupyter notebook,
# directly below the code cell that produced it

print("Finished successfully loading packages")


# Reading data from a CSV file into a NumPy array
def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return np.array(data)


file_path = "housingUnits.csv"
data = read_csv(file_path)
all_columns = data[0:1, :]  # This selects all rows and all columns
print(all_columns)

# Loading the columns
housing_median_age_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(0,), skip_header=1
)

total_rooms_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(1,), skip_header=1
)
total_bedrooms_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(2,), skip_header=1
)
population_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(3,), skip_header=1
)
households_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(4,), skip_header=1
)

median_income_col = np.genfromtxt(
    file_path, delimiter=",", dtype=float, usecols=(5,), skip_header=1
)
ocean_proximity_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(6,), skip_header=1
)
median_house_value_col = np.genfromtxt(
    file_path, delimiter=",", dtype=int, usecols=(7,), skip_header=1
)

# Normalizing our columns
rooms_by_population = total_rooms_col / population_col
bedrooms_by_population = total_bedrooms_col / population_col

rooms_by_households = total_rooms_col / households_col
bedrooms_by_households = total_bedrooms_col / households_col


def plot_linear_regression(predictor, outcome, title="", xlab="", ylab=""):
    """
    Takes two different columns: predictor and the outcome, and then returns a matplotlib plot of the LinearRegression using sklearn +
    r2_score of it
    """
    # Reshape predictor and outcome columns to 2D arrays
    predictor = predictor.reshape(-1, 1)
    outcome = outcome.reshape(-1, 1)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(predictor, outcome)

    # Make predictions
    predictions = model.predict(predictor)

    # Calculate R^2 score
    r2 = r2_score(outcome, predictions)

    # Plot the data points and the regression line
    plt.scatter(predictor, outcome, color="blue")
    plt.plot(predictor, predictions, color="red")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()

    # Display R^2 score, betas, and intercept
    plt.text(
        0.95,
        0.95,
        f"R^2 = {r2:.2f}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.95,
        0.90,
        f"Beta 0 (Intercept) = {model.intercept_[0]:.2f}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.95,
        0.85,
        f"Beta 1 = {model.coef_[0][0]:.2f}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )

    # Show the plot
    plt.show()


population_to_price_unnormalized = plot_linear_regression(
    predictor=population_col,
    outcome=median_house_value_col,
    title="population in the block to median price (unnormalized)",
    xlab="population in the block",
    ylab="median house price in the block",
)


households_to_price_unnormalized = plot_linear_regression(
    predictor=households_col,
    outcome=median_house_value_col,
    title="# households in the block to median price (unnormalized)",
    xlab="number of households",
    ylab="median house price in the block",
)

rooms_to_price_unnormalized = plot_linear_regression(
    predictor=total_rooms_col,
    outcome=median_house_value_col,
    title="rooms number to price (unnormalized)",
    xlab="num_of_rooms",
    ylab="median_housing_price",
)

bedrooms_to_price_unnormalized = plot_linear_regression(
    predictor=total_bedrooms_col,
    outcome=median_house_value_col,
    title="bedrooms number to price (unnormalized)",
    xlab="num_of_rooms",
    ylab="median_housing_price",
)

rooms_to_price_normalized_by_population = plot_linear_regression(
    predictor=rooms_by_population,
    outcome=median_house_value_col,
    title="rooms number to price (normalized by the population)",
    xlab="rooms per capita",
    ylab="median_housing_price",
)

bedrooms_to_price_normalized_by_population = plot_linear_regression(
    predictor=bedrooms_by_population,
    outcome=median_house_value_col,
    title="bedrooms number to price (normalized by the population)",
    xlab="bedrooms per capita",
    ylab="median_housing_price",
)

rooms_to_price_normalized_by_households = plot_linear_regression(
    predictor=rooms_by_households,
    outcome=median_house_value_col,
    title="rooms number to price (normalized by the households)",
    xlab="rooms per household",
    ylab="median_housing_price",
)

bedrooms_to_price_normalized_by_households = plot_linear_regression(
    predictor=bedrooms_by_households,
    outcome=median_house_value_col,
    title="bedrooms number to price (normalized by the households)",
    xlab="num_of_bedrooms_per_household",
    ylab="median_housing_price",
)

print(
    "Correlation coff between 'population' and 'house_value': ",
    np.corrcoef(population_col, median_house_value_col)[0, 1],
)
print(
    "Correlation coff between 'number of households' and 'house_value': ",
    np.corrcoef(households_col, median_house_value_col)[0, 1],
)

print(
    "Correlation coff between 'number of rooms' and 'house_value': ",
    np.corrcoef(total_rooms_col, median_house_value_col)[0, 1],
)
print(
    "Correlation coff between 'number of bedrooms' and 'house_value': ",
    np.corrcoef(total_bedrooms_col, median_house_value_col)[0, 1],
)

print(
    "Correlation coff between 'number of rooms normalized by population' and 'house_value': ",
    np.corrcoef(rooms_by_population, median_house_value_col)[0, 1],
)
print(
    "Correlation coff between 'number of bedrooms normalized by population' and 'house_value': ",
    np.corrcoef(bedrooms_by_population, median_house_value_col)[0, 1],
)

print(
    "Correlation coff between 'number of rooms normalized by households' and 'house_value': ",
    np.corrcoef(rooms_by_households, median_house_value_col)[0, 1],
)
print(
    "Correlation coff between 'number of bedrooms normalized by households' and 'house_value': ",
    np.corrcoef(bedrooms_by_households, median_house_value_col)[0, 1],
)

# Data
data = [
    ["Population", np.corrcoef(population_col, median_house_value_col)[0, 1]],
    ["Number of households", np.corrcoef(households_col, median_house_value_col)[0, 1]],
    ["Number of rooms", np.corrcoef(total_rooms_col, median_house_value_col)[0, 1]],
    [
        "Number of bedrooms",
        np.corrcoef(total_bedrooms_col, median_house_value_col)[0, 1],
    ],
    [
        "Rooms normalized by population",
        np.corrcoef(rooms_by_population, median_house_value_col)[0, 1],
    ],
    [
        "Bedrooms normalized by population",
        np.corrcoef(bedrooms_by_population, median_house_value_col)[0, 1],
    ],
    [
        "Rooms normalized by households",
        np.corrcoef(rooms_by_households, median_house_value_col)[0, 1],
    ],
    [
        "Bedrooms normalized by households",
        np.corrcoef(bedrooms_by_households, median_house_value_col)[0, 1],
    ],
]

# Table header
headers = ["Variable", "Correlation coefficient"]

# Print table
print(tabulate(data, headers=headers, tablefmt="grid"))

median_income_to_price = plot_linear_regression(
    predictor=median_income_col,
    outcome=median_house_value_col,
    title="income to price",
    xlab="median income in the block",
    ylab="median house value in the block",
)

ocean_proximity_to_price = plot_linear_regression(
    predictor=ocean_proximity_col,
    outcome=median_house_value_col,
    title="ocean proximity to price",
    xlab="ocean proximity 1-4",
    ylab="median house value in the block",
)

housing_median_age_to_price = plot_linear_regression(
    predictor=housing_median_age_col,
    outcome=median_house_value_col,
    title="housing age to price",
    xlab="median housing age in the block",
    ylab="median house value in the block",
)


def visualize_3d_regression(x1, x2, y):
    # Reshape inputs for sklearn
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    y = np.array(y)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(np.hstack((x1, x2)), y)

    # Generate predictions
    y_pred = model.predict(np.hstack((x1, x2)))

    # Calculate evaluation metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Print coefficients
    print("Coefficients (betas):")
    print("Beta 1 (x1):", model.coef_[0])
    print("Beta 2 (x2):", model.coef_[1])
    print("Intercept:", model.intercept_)

    # Print evaluation metrics
    print("R^2 score:", r2)
    print("RMSE:", rmse)

    # Visualize the regression plane
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x1, x2, y, c="blue", marker="o", alpha=0.5)

    # Create meshgrid for regression plane
    x1_range = np.linspace(min(x1), max(x1), 10)
    x2_range = np.linspace(min(x2), max(x2), 10)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    y_mesh = model.intercept_ + model.coef_[0] * x1_mesh + model.coef_[1] * x2_mesh

    # Plot regression plane
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.5, color="red")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")

    plt.title("3D Visualization of Regression Model")
    plt.show()

    # Return relevant statistics
    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "r2_score": r2,
        "rmse": rmse,
    }


visualize_3d_regression(median_income_col, rooms_by_population, median_house_value_col)


# Reading data from a CSV file into a NumPy array
def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return np.array(data)


file_path = "housingUnits.csv"
data = read_csv(file_path)
all_columns = data[0:1, :]  # This selects all rows and all columns
print(all_columns)


def multiple_regression(data):
    """
    without normalization: simply taking all the predictor columns and then outcome and mapping it in the 8D
    """
    data_array = np.array(
        data[1:], dtype=float
    )  # make numeric and remove the first header row

    # I am using RAW predictors 2 and 3 (not normalized)
    x = data_array[:, :-1]  # Select all columns except the last one as predictors
    y = data_array[:, -1]  # Select the last column as outcome

    # Split data into training and testing sets + add a seed for random gen
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=21
    )

    # Let's now fit linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # generating predictions based on the testing set
    y_pred = model.predict(x_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Return relevant statistics
    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
    }


print("Using all the raw predictors: \n")

results = multiple_regression(data)
print("--------------------------------------------")
print("Intercept: ", results["intercept"])
print("--------------------------------------------")
for i, ii in enumerate(results["coefficients"]):
    print("Beta {}: ".format(i + 1), ii)
print("--------------------------------------------")
print("R^2 Score: ", results["r2_score"])
print("--------------------------------------------")
print("RMSE: ", results["rmse"])
print("--------------------------------------------")
print("Mean arithmetic error: ", results["mae"])
print("--------------------------------------------")


from sklearn.preprocessing import StandardScaler


def multiple_regression_with_normalized(data):
    """
    with normalized variables
    """
    housing_median_age_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(0,), skip_header=1
    )
    median_income_col = np.genfromtxt(
        file_path, delimiter=",", dtype=float, usecols=(5,), skip_header=1
    )
    ocean_proximity_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(6,), skip_header=1
    )
    median_house_value_col = np.genfromtxt(
        file_path, delimiter=",", dtype=float, usecols=(7,), skip_header=1
    )

    # Normalizing our columns
    total_rooms_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(1,), skip_header=1
    )
    total_bedrooms_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(2,), skip_header=1
    )
    population_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(3,), skip_header=1
    )
    households_col = np.genfromtxt(
        file_path, delimiter=",", dtype=int, usecols=(4,), skip_header=1
    )

    rooms_by_population = total_rooms_col / population_col
    bedrooms_by_population = total_bedrooms_col / population_col

    # Using selected columns as predictors
    X = np.column_stack(
        (
            housing_median_age_col,
            median_income_col,
            ocean_proximity_col,
            rooms_by_population,
            bedrooms_by_population,
            households_col,
            population_col,
        )
    )

    # Using median_house_value_col as outcome
    y = median_house_value_col

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )

    # Normalizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Return relevant statistics
    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "r2_score": r2,
        "rmse": rmse,
    }


results = multiple_regression_with_normalized(data)
print("Using normalized predictors: ")
print("--------------------------------------------")
print("Intercept: ", results["intercept"])
print("--------------------------------------------")
for i, ii in enumerate(results["coefficients"]):
    print("Beta {}: ".format(i + 1), ii)
print("--------------------------------------------")
print("R^2 Score: ", results["r2_score"])
print("--------------------------------------------")
print("RMSE: ", results["rmse"])
print("--------------------------------------------")

pop_to_households = plot_linear_regression(
    predictor=population_col,
    outcome=households_col,
    title="population to households relationship",
    xlab="population",
    ylab="number of households",
)

rooms_to_bedrooms = plot_linear_regression(
    predictor=total_rooms_col,
    outcome=total_bedrooms_col,
    title="rooms to bedrooms relationship",
    xlab="rooms",
    ylab="bedrooms",
)

rooms_to_bedrooms_normalized = plot_linear_regression(
    predictor=rooms_by_population,
    outcome=bedrooms_by_population,
    title="rooms to bedrooms relationship (normalized by population)",
    xlab="rooms per capita",
    ylab="bedrooms per capita",
)

rooms_to_bedrooms_normalized = plot_linear_regression(
    predictor=rooms_by_households,
    outcome=rooms_by_households,
    title="rooms to bedrooms relationship (normalized by bedrooms)",
    xlab="rooms per household",
    ylab="bedrooms per household",
)


def plot_freq_dist(data, color: str = "red", title: str = "N/A"):
    """
    data: 1d dataframe to plot on the graph
    """
    data_size = data.shape[0]
    # Let's use the Scott's rule for choosing the right bin width
    w = 3.49 * np.std(data) / data_size ** (1.0 / 3)

    # define number of bins with accordance to the appropriate size
    bins = np.arange(data.min().item(), data.max().item() + w, w, dtype=float)

    plt.hist(data, edgecolor=color, bins=bins)
    plt.xlabel("{} distribution".format(title))  # Set x-axis label
    plt.ylabel("frequency")  # Set y-axis label
    plt.show()  # Show the plot


rooms_normalized_plot = plot_freq_dist(
    rooms_by_population, color="red", title="total rooms normalized"
)
bedrooms_normalized_plot = plot_freq_dist(
    bedrooms_by_population, color="green", title="bedrooms normalized"
)

housing_median_age_plot = plot_freq_dist(
    housing_median_age_col, color="red", title="housing median age"
)
median_income_plot = plot_freq_dist(
    median_income_col, color="green", title="median income normalized"
)

ocean_proximity_plot = plot_freq_dist(
    ocean_proximity_col, color="red", title="ocean proximity plot"
)
median_house_value_plot = plot_freq_dist(
    median_house_value_col, color="green", title="median house value plot"
)

population_plot = plot_freq_dist(population_col, color="red", title="population plot")
households_plot = plot_freq_dist(households_col, color="green", title="households plot")


if __name__ == "__main__":

    print("Finished successfully running the homework1.py")
