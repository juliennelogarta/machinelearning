import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Ask for N
N = int(input("Enter the number of points (N): "))

# Initialize arrays for x and y
x = np.zeros(N)
y = np.zeros(N)

# Read N points
for i in range(N):
    x[i] = float(input(f"Enter x value for point {i+1}: "))
    y[i] = float(input(f"Enter y value for point {i+1}: "))

# Reshape x to fit the model
x = x.reshape(-1, 1)

# Ask for k
k = int(input("Enter the number of neighbors (k): "))

if k > N:
    print("Error: k cannot be greater than N.")
else:
    # Fit the model
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x, y)

    # Ask for X
    X = float(input("Enter a new x value (X): "))
    X = np.array([X]).reshape(-1, 1)

    # Predict Y
    Y = model.predict(X)

    print(f"The predicted y value (Y) for x = {X[0][0]} is {Y[0]}")

    # Calculate and print the coefficient of determination
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    print(f"The coefficient of determination is {r2}")