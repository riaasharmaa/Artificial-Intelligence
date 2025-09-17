import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg

if __name__ == '__main__':
    cfile = sys.argv[1]
    #Question 2: Visualize Data
    years = []
    days = []
    #Read csv
    with open(cfile, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            years.append(int(row['year']))
            days.append(int(row['days']))
    #Plot
    plt.plot(years, days)
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    #Save as plot.jpg
    plt.savefig('plot.jpg')

    #Question 3: Linear Regression
    #Q3a
    matrix = []
    with open(cfile, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            year = int(row['year'])
            vector = np.array([1, year], dtype=np.int64)
            vector = np.transpose(vector)
            matrix.append(vector)
    matrix = np.array(matrix)
    print("Q3a:")
    print(matrix)

    #Q3b
    y = []
    with open(cfile, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            days = int(row['days'])
            y.append(days)
    y = np.array(y, dtype=np.int64)
    print("Q3b:")
    print(y)

    #Q3c
    matrixtrans = np.transpose(matrix)
    z = np.matmul(matrixtrans, matrix)
    z = np.array(z, dtype=np.int64)
    print("Q3c:")
    print(z)

    #Q3d
    inv = np.linalg.inv(z)
    print("Q3d:")
    print(inv)

    #Q3e
    pseudoinv = np.matmul(inv, matrixtrans)
    print("Q3e:")
    print(pseudoinv)

    #Q3f
    beta = np.matmul(pseudoinv, y)
    print("Q3f:")
    print(beta)

    #Question 4: Prediction
    testyear = 2022
    prediction = beta.item(0) + (beta.item(1) * testyear)
    print("Q4: {:}".format(prediction))

    #Question 5: Model Interpretation
    if beta.item(1) > 0:
        print("Q5a: >")
    if beta.item(1) < 0:
        print("Q5a: <")
    else:
        print("Q5a: =")
    print(
        "Q5b: If the sign is negative, as the starting year increases, the number of ice days will decrease."
        " When the sign is zero, there is no clear relationship between the starting year changing."
        " and the number of ice days. When the sign is positive, as the starting year increases, the number of ice days will increase."
    )

    #Question 6: Model Limitation
    if beta.item(1) != 0:
        limit = ((-1) * beta.item(0)) / beta.item(1)
        print("Q6a: {:}".format(limit))
        print("Q6b: The predicted year seems inaccurate since the number of ice days didn't decrease that much in the time observed, so it's unlikely there will be no ice days by the time of the predicted year.")
    else:
        print("Q6a: Cannot determine when there will be no ice days.")
