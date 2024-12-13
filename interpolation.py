import numpy as np
import matplotlib.pyplot as plt
import csv

def newton_divided_difference(x, y):
    n = len(x)
    coef = np.zeros([n, n])  # Create a 2D array of zeros to store coefficients
    coef[:,0] = y  # Set the first column of coef to the y values

    for j in range(1, n):  # Loop over each column
        for i in range(n-j):  # Loop over each row in the column
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])  # Calculate the divided difference for the coefficient

    return coef[0, :]  # Return the first row, which contains the coefficients

def newton_polynomial(coef, x_data, x):
    n = len(x_data) - 1  # Get the degree of the polynomial
    p = coef[n]  # Start with the last coefficient

    for k in range(1, n+1):  # Loop over each coefficient
        p = coef[n-k] + (x - x_data[n-k]) * p  # Calculate the polynomial value at a specific x

    return p  # Return the polynomial's value at x


def simple_linear_regression(x, y):
    n = len(x)  # Get the number of data points
    x_mean = np.mean(x)  # Calculate the mean of the x values
    y_mean = np.mean(y)  # Calculate the mean of the y values
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))  # Calculate the numerator for the slope
    den = sum((x[i] - x_mean) ** 2 for i in range(n))  # Calculate the denominator for the slope
    slope = num / den  # Calculate the slope of the regression line
    intercept = y_mean - slope * x_mean  # Calculate the intercept of the regression line
    return slope, intercept  # Return the slope and intercept


def fill_missing_values(x, y):
    known_x = [x[i] for i in range(len(x)) if y[i] is not None]  # Get x values where y is known
    known_y = [y[i] for i in range(len(x)) if y[i] is not None]  # Get y values that are not None
    coef = newton_divided_difference(known_x, known_y)  # Calculate the Newton coefficients for the known data
    filled_y = [newton_polynomial(coef, known_x, xi) if yi is None else yi for xi, yi in zip(x, y)]  # Interpolate missing y values
    return filled_y  # Return the list of y values, with missing values filled


def forecast_values(x, y, k):
    slope, intercept = simple_linear_regression(x, y)  # Get the slope and intercept from linear regression
    forecast_x = [x[-1] + i for i in range(1, k+1)]  # Create future x values by extending from the last known x
    forecast_y = [slope * xi + intercept for xi in forecast_x]  # Calculate the corresponding y values using the regression line
    return forecast_x, forecast_y  # Return the forecasted x and y values


def plot_data(x, y, filled_y=None, forecast_x=None, forecast_y=None, slope=None, intercept=None):
    original_x = [xi for xi, yi in zip(x, y) if yi is not None]  # Filter out the missing y values for plotting
    original_y = [yi for yi in y if yi is not None]  # Filter out the missing y values
    plt.plot(original_x, original_y, 'o-', label='Original Data', color='blue')  # Plot the original data
    
    if filled_y:
        plt.plot(x, filled_y, 'x--', label='Interpolated Data', color='red')  # Plot the interpolated data
    if forecast_x and forecast_y:
        plt.plot(forecast_x, forecast_y, 'o--', label='Forecasted Data', color='black')  # Plot the forecasted data
    if slope is not None and intercept is not None:
        regression_x = np.linspace(min(x), max(x), 100)  # Create x values for the regression line
        regression_y = slope * regression_x + intercept  # Calculate y values for the regression line
        plt.plot(regression_x, regression_y, '-', label='Regression Line', color='orange')  # Plot the regression line
        
        extended_x = np.linspace(min(x), max(forecast_x), 100)  # Extend the x values for the regression line
        extended_y = slope * extended_x + intercept  # Calculate the extended y values
        plt.plot(extended_x, extended_y, '--', color='orange')  # Plot the extended regression line
        
    plt.legend()  # Show the legend for the plot
    plt.show()  # Display the plot


def main():
    while True:
        print("[1] – Newton Divided-and-Difference Interpolation")
        print("[2] – Linear Regression")
        print("[0] – Exit")
        choice = int(input("Enter your choice: "))
        
        if choice == 0:
            break
        
        file_path = input("Enter the path to the CSV file: ")
        x = []
        y = []
        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    x.append(float(row['x']))
                    y.append(None if row['y'] in ('', 'None') else float(row['y']))
        except KeyError as e:
            print(f"Error: Missing column in CSV file: {e}")
            continue
        except ValueError as e:
            print(f"Error: Invalid data in CSV file: {e}")
            continue
        
        if choice == 1:
            filled_y = fill_missing_values(x, y)
            print("Given values:")
            print(f"x = {[float(val) for val in x]}")
            print(f"y = {[float(val) if val is not None else None for val in y]}")
            print("Interpolated values:")
            print(f"y = {[float(val) for val in filled_y]}")
            plot_data(x, y, filled_y=filled_y)
        
        elif choice == 2:
            k = int(input("Enter the number of future points to forecast: "))
            known_x = [x[i] for i in range(len(x)) if y[i] is not None]
            known_y = [y[i] for i in range(len(x)) if y[i] is not None]
            forecast_x, forecast_y = forecast_values(known_x, known_y, k)
            slope, intercept = simple_linear_regression(known_x, known_y)
            print("Given values:")
            print(f"x = {[float(val) for val in x]}")
            print(f"y = {[float(val) if val is not None else None for val in y]}")
            print("Forecasted values:")
            print(f"x = {[float(val) for val in forecast_x]}")
            print(f"y = {[float(val) for val in forecast_y]}")
            plot_data(x, y, forecast_x=forecast_x, forecast_y=forecast_y, slope=slope, intercept=intercept)

if __name__ == "__main__":
    main()