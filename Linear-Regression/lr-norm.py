import sys
import numpy as np
import pandas as pd


def train_data(filename):
    datas = pd.read_csv(filename)
    X = datas['Area'].values
    X = np.column_stack((np.ones(len(X)), X))

    y = datas['Price'].values
    return X, y


def normal_equation(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def main():
    train_filename = sys.argv[1]
    X, y = train_data(train_filename)
    theta = normal_equation(X, y)
    print(theta)


if __name__ == '__main__':
    main()
