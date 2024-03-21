# analysis of the hidden state activity

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split


def ridge_wgridsearch(X, Y, alphavec=None):
    # does a grid search for ridge regression

    if alphavec is None:
        alphavec = [1e-4, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    Rstest = []
    for k in range(len(alphavec)):
        alpha = alphavec[k]

        # partition into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        reg_1 = Ridge(alpha=alpha).fit(X_train, Y_train)
        rsquared = reg_1.score(X_test, Y_test)
        Rstest.append(rsquared)

    amax = alphavec[np.argmax(Rstest)]

    # do the final regression
    reg_1 = Ridge(alpha=amax).fit(X, Y)
    rsquared = reg_1.score(X, Y)

    slope = reg_1.coef_
    b = reg_1.intercept_

    return slope, b, rsquared, amax, Rstest[np.argmax(Rstest)]


def lasso_wgridsearch(X, Y, alphavec=None):
    # does a grid search for LASSO (L1) regression

    if alphavec is None:
        alphavec = [1e-4, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    Rstest = []
    # print(len(alphavec))
    for k in range(len(alphavec)):
        alpha = alphavec[k]

        # partition into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        reg_1 = Lasso(alpha=alpha).fit(X_train, Y_train)
        rsquared = reg_1.score(X_test, Y_test)
        Rstest.append(rsquared)

    amax = alphavec[np.argmax(Rstest)]

    # do the final regression
    reg_1 = Lasso(alpha=amax).fit(X, Y)
    rsquared = reg_1.score(X, Y)

    slope = reg_1.coef_
    b = reg_1.intercept_

    return slope, b, rsquared, amax, Rstest[np.argmax(Rstest)]
