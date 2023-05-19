import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def Cost_Function_J(m, y_true, y_predicted):
    return (1 / m) * (1 / 2) * np.sum((y_predicted - y_true) ** 2)


def normalisation(data):
    m_samples, n_features = data.shape
    normalised_data = np.zeros((m_samples, n_features))
    normalised_min_max = {}

    for i in range(n_features):
        max_value, min_value = data[:, i].max(), data[:, i].min()
        normalised_min_max["index {} maximum".format(i)] = max_value
        normalised_min_max["index {} minimum".format(i)] = min_value
        normalised_data[:, i] = (data[:, i] - min_value) / (max_value - min_value)
    return normalised_data, normalised_min_max


def model_optimisation(X, y, learning_rate, no_iters):
    m_samples, n_features = X.shape

    # initial weights and bias values
    weights = np.zeros((1, n_features))
    bias = 0

    # create empty list to store costs
    costs = []
    x_vals_costs = []

    # running linear regression
    for i in range(no_iters):
        y_hat = np.dot(X, weights) + bias

        # cost
        iteration_cost = Cost_Function_J(m_samples, y, y_hat)

        if i != 0:
            # checking for convergence
            if costs[-1] == iteration_cost:
                costs.append(iteration_cost)
                x_vals_costs.append(i + 1)
                print("converged in {} iterations...".format(i + 1))
                break
            elif i + 1 == iterations:
                costs.append(iteration_cost)
                x_vals_costs.append(i + 1)
                print("did not converge...")
                break

        # append cost to list
        if no_iters < 100:
            costs.append(iteration_cost)
            x_vals_costs.append(i + 1)
        elif i % 10 == 0:
            costs.append(iteration_cost)
            x_vals_costs.append(i + 1)

        # update rules
        dw = (1 / m_samples) * np.dot(X.T, (y_hat - y))
        db = (1 / m_samples) * np.sum(y_hat - y)

        # updating weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, costs, x_vals_costs


def predict(X, weights, bias):
    return np.dot(X, weights) + bias


def scoring_formula(X, y, y_hat):
    m_samples = X.shape[0]
    # calculating the R^2 score
    return 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - ((1 / m_samples) * np.sum(y))) ** 2))


if __name__ == "__main__":
    # learning rate
    alpha = 0.1

    # iterations
    iterations = 10000

    # importing data
    df = pd.read_csv("Salary_Data.csv")

    X_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1:].values

    # normalising the data
    X_data, X_max_min = normalisation(X_data)

    # splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=1/3, random_state=0)

    # running linear regression
    w, b, cost_list, cost_iterations = model_optimisation(X_train, y_train, alpha, iterations)

    # using test data for prediction
    prediction = predict(X_test, w, b)

    # check MSE and R2 scores
    r2 = scoring_formula(X_test, y_test, prediction)
    print("R2 score: ", r2)

    # plotting cost per iteration
    plt.figure(1, figsize=(8, 6))
    plt.plot(cost_iterations, cost_list)
    plt.title("Cost over iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")

    # plotting linear regression result
    x_vals = np.linspace(0, 1, 100)
    x_vals = x_vals.reshape(len(x_vals), 1)
    y_vals = np.dot(x_vals, w) + b

    plt.figure(2, figsize=(8, 6))
    plt.scatter(X_train, y_train, color='red', marker='x', s=10)
    plt.scatter(X_test, y_test, color='blue', marker='x', s=10)
    plt.plot(x_vals, y_vals, color='k')
    plt.title("Linear Regression")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.legend(["linear regression", "training data", "test data"])

    plt.show()
