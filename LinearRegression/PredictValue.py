import numpy as num
'''
h(x) = theta0 + theta1x;
also for multiple features
'''
def Hypothe(theta, xi):
    if(len(theta) == len(xi)):
        sum = 0
        for i in range(len(xi)):
            sum += theta[i] * xi[i]

        return sum
    else:
        return False


'''
Cost FUcntion
'''
def RegCostFunction(theta, x, y):
    m = len(y)
    J = 0
    for i in range(x):
        J += ((Hypothe(theta, x[i]) - y[i])**2)
    J = J/2*m
    return J

'''
Gradient Decent Algo
for all values of θ­ simultaneously update,
 θj= θj - α * (Ə J(θ)/ Əθj)
}

for J(theta)
I have defined GradeTerm her
and for theta GradDecent
'''
def gradientTerm(X, y, Theta, position):
    sum = 0
    for i in range(X):
        sum = (Hypothe(Theta, X[i]) - y[i]) - X[i][position]
    return sum

def gradientDecent(X, y, Theta, alpha):
    ThetaList = []
    for i in range(X):
        ThetaList.append((Theta[i] - (alpha * gradientTerm(X, y, Theta, i))))
    return ThetaList

'''
TO perform linear regrssion,
1. We'll call linear regression and provide it with a Features and Labels.
2. We'll add feature x0 = 1 to every example in data set.
3. We'll initialize Theta=[0]*len(Features[0]). (i.e.  number of parameters should be equal to number of features )
4. We'll select a particular value of alpha and no. of  iterations, and run gradient descent algorithm to minimize the parameters θ.
Make sure to keep alpha as small as possible in the order of 0.000001 or less and the number of iterations to be around 100,000-1,000,000 for better results.
'''

def linearRegression(Xfeatures, yLabels, alpha, iteration):
    if(len(X) != len(Y)):
        print("Data Missing")
    else:
        #initializing X[0] = 1 for all features
        for i in  range(Xfeatures):
            Xfeatures[i].insert(0,1)
        #initialiizing theta make sure theta size equals to X Fetaures
        Theta = [0][len(Xfeatures)]

        for i in range(iteration):
            print("\n Iteration number")
            print("Theta value before", Theta)
            Theta = gradientDecent(Xfeatures, yLabels, Theta, alpha)
        print("Theta After GD",Theta)
        return Theta



'''
GET DATA FROM CSV FILE OR TXT FILE TO PERFORM LINEAR REGRETION
'''
Xfeatures = []
YLabels = []
alpha = 0
iteration = 10000
f1 = open("LinearRegression.txt")
z = f1.readline()
print("Fetching data ...")
while z:
    print(".", end=".")
    temp = z.split(",")
    temp1 = []
    for i in range(len(temp) - 1):
        temp1.append(float(temp[i]))
    Xfeatures.append(temp1)
    YLabels.append(float(temp[-1]))
    z = f1.readline()
f1.flush()
print("")
print(Xfeatures)