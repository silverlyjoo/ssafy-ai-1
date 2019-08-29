import numpy as np 
from matplotlib import pyplot as plt 

data = np.array([[100, 20], 
		[150, 24], 
		[300, 36], 
		[400, 47], 
		[130, 22], 
		[240, 32],
		[350, 47], 
		[200, 42], 
		[100, 21], 
		[110, 21], 
		[190, 30], 
		[120, 25], 
		[130, 18], 
		[270, 38], 
		[255, 28]])

# plt.scatter(data[:, 0], data[:, 1]) 
# plt.title("Time / Distance")
# plt.xlabel("Delivery Distance (meter)")
# plt.ylabel("Time Consumed (minute)")
# plt.axis([0, 420, 0, 50])
# plt.show() 

def qr_householder(A):
    m, n = A.shape
    Q = np.eye(m) # Orthogonal transform so far
    R = A.copy() # Transformed matrix so far

    for j in range(n):
        # Find H = I - beta*u*u' to put zeros below R[j,j]
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        u = x / u1
        u[0] = 1
        beta = -rho * u1 / normx

        R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
        Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
        
    return Q, R

m, n = data.shape
A = np.array([data[:,0], np.ones(m)]).T
b = data[:, 1] 

print('A')
print(A)
print('b')
print(b)

Q, R = qr_householder(A) 

print('Q')
print(Q)
print('R')
print(R)

b_hat = Q.T.dot(b) 


R_upper = R[:n, :]
b_upper = b_hat[:n]

# print(R_upper, b_upper) 

x = np.linalg.solve(R_upper, b_upper) 
slope, intercept = x 

print(x)