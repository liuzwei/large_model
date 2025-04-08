from sklearn import linear_model

def test_linear_model():
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], [0, 1, 4, 8, 16])
    print(reg.coef_)
    print(reg.predict([[1.5, 1.5]]))