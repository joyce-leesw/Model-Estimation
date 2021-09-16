import numpy as np
import utilities as util
import pandas
import matplotlib.pyplot as plt
import sys

#load data from file
Dx,Dy = util.load_points_from_file(sys.argv[1:][0])

#write a function for least square
def least_squares(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

#calculate sum squared errors
def square_error(y, y_hat):
    return np.sum((y - y_hat)**2)

def poly(x, y, power):
    build = np.ones(x.shape)
    for i in range (1, power+1):
        build = np.column_stack((build, x**i))
    A = np.linalg.inv(build.T.dot(build)).dot(build.T).dot(y)
    return A[:,0]

def exp(x, y):
    X = np.column_stack((np.ones(x.shape), np.exp(x)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    xs = np.linspace(x[0], x[19], 20)
    ys = A[0] + A[1] * np.exp(xs)
    yys = A[0] + A[1] * np.exp(x)
    return xs, ys, yys

def sine(x, y):
    X = np.column_stack((np.ones(x.shape), np.sin(x)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    xs = np.linspace(x[0], x[19], 20)
    ys = A[0] + A[1] * np.sin(xs)
    yys = A[0] + A[1] * np.sin(x)
    return xs, ys, yys

def linear(x, y):
    a, b = least_squares(x, y)
    y1 = a + b * x
    return y1

def cross_validation(x, y, kfold):
    # kfold = 5
    final_error = 100000
    final_order = 1

    for o in range (1, 7):
        order_total = 0
        for i in range (1, 20//kfold):
            if i == 0:
                x_test = x[:(i+1)*kfold-1]
                y_test = y[:(i+1)*kfold-1]

            x_train = x[i*kfold-1:(i+1)*kfold-1]
            x_test = list(x[:i*kfold-1])
            x_test.extend([x[(i+1)*kfold-1]])

            y_train = y[i*kfold-1:(i+1)*kfold-1]
            y_test = list(y[:i*kfold-1])
            y_test.extend([y[(i+1)*kfold-1]])

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            px = x_train.reshape((len(x_train), 1))
            py = y_train.reshape((len(y_train), 1))
            xs = np.linspace(px.min(), px.max(), 1000)
            coefficients = poly(px, py, o)
            fit = np.poly1d(np.flip(coefficients))

            poly_fitted = fit(x_test)
            order_error = square_error(y_test, poly_fitted)
            order_total = order_total + order_error

        if (order_total < final_error):
            final_error = order_total
            final_order = o

    return final_order

fig, ax = plt.subplots()
ax.scatter( Dx, Dy, s=20 )
final_error = 0

for i in range(0, len(Dx)//20):
    final = 1
    s = i*20
    e = s + 20

    error= 100000

    #coefficient = np.polyfit(Dx[s:e], Dy[s:e], d)
    #poly = np.poly1d(coefficient)

    plt.scatter(Dx, Dy)

    px = Dx[s:e].reshape((len(Dx[s:e]), 1))
    py = Dy[s:e].reshape((len(Dy[s:e]), 1))
    xs = np.linspace(px.min(), px.max(), 1000)
    coefficients = poly(px, py, 3)
    poly_fitted = coefficients[0] + coefficients[1] * xs + coefficients[2] * (xs ** 2) + coefficients[3] * (xs ** 3)  

    xxs = np.linspace(px.min(), px.max(), 20)
    poly_hat = coefficients[0] + coefficients[1] * px + coefficients[2] * (px ** 2) + coefficients[3] * (px ** 3) 

    if (square_error(py, poly_hat) < error):
        final = 3 
        error = square_error(py, poly_hat)

    sx, sy, syy = sine(Dx[s:e], Dy[s:e])
    if (square_error(Dy[s:e], syy) < error):
        final = 1 
        error = square_error(Dy[s:e], syy)
    
    ex, ey, eyy = exp(Dx[s:e], Dy[s:e])
    if (square_error(Dy[s:e], eyy) < error):
        final = 2
        error = square_error(Dy[s:e], eyy)

    linearY = linear(Dx[s:e], Dy[s:e])
    if (square_error(Dy[s:e], linearY) < error):
        final = 4
        error = square_error(Dy[s:e], linearY)

    order = cross_validation(Dx[s:e], Dy[s:e], 10)
    print("Final order is:", order)

    # to prevent overfitting
    # if order == 1:
    #     final = 4

    if final == 1:
        ax.plot(sx, sy)
        print("sine")

    if final == 2:
        ax.plot(ex, ey)

    if final == 3:
       ax.plot(xs, poly_fitted) 

    if final == 4:
        ax.plot(Dx[s:e], linearY)

final_error = final_error + error
plt.show()

# c = np.polyfit(Dx[s:e], Dy[s:e], final)
# p = np.poly1d(c)
# if s != 0:
#     s = s - 1
# ax.plot(Dx[s:e], p(Dx[s:e]))

# if s != 0:
#     s + 1

if (len(sys.argv[1:]) > 1):
    if (sys.argv[1:][1] == "--plot"):
        plt.show()

print(final_error)
       

