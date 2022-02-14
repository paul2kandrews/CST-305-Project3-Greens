import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def q1_result(y, x):
    func = 4 * np.exp(-x) + 2*x*np.exp(-x) + 2*x - 4
    return [y[1], func]

def q2_result(y, x):
    func = 2 * np.cos(x) + x**2 - 2
    return [y[1], func]

def q1_greens(t):
    return 1/3*(t**3)

def useOdeint(func, x0, y0, h, n): # Wrapper function for iterative method
    inputSpace = np.linspace(x0, x0+h*n, n+1) # Create input space
    y = odeint(func, y0, inputSpace) # Solve differential equation using SciPy
    
    return [inputSpace, y] # Return input and solution arrays

def useGreens(func, x0, h, n):
    inputSpace = np.linspace(x0, x0+h*n, n+1) # Create input space
    y = []
    for i in inputSpace:
        y.append(func(i))
    return [inputSpace, y]

# Define initial conditions
x0 = 1
y0 = [0, 0] # y(0) = 0, y'(0) = 0
h = 0.02
n = 1000


# Question 1: Plot results from standard methods
odeintResults = useOdeint(q1_result, x0, y0, h, n) # Get results from Odeint method
plt.plot(odeintResults[0], odeintResults[1][:,0], 'r:', linewidth=2, label=f'Undet. Coeffs/Var. Params') # Plot results
                        # should we use [:,0] or [:,1] ???
plt.xlabel('x') # Label x-axis
plt.ylabel('y') # Label y-axis
plt.title('Solution Approximation using SciPy Odeint') # Set plot title
plt.legend() # Add legend
plt.show() # Display plot

# Question 1: Plot results from Green's function
odeintResults = useGreens(q1_greens, x0, h, n) # Get results from Green's function
plt.plot(odeintResults[0], odeintResults[1], 'g:', linewidth=2, label=f"Green's Function") # Plot results
plt.xlabel('x') # Label x-axis
plt.ylabel('y') # Label y-axis
plt.title("Solution Approximation using Green's Function") # Set plot title
plt.legend() # Add legend
plt.show() # Display plot

# Question 2: Plot results from standard methods
odeintResults = useOdeint(q2_result, x0, y0, h, n) # Get results from Odeint method
plt.plot(odeintResults[0], odeintResults[1][:,0], 'r:', linewidth=2, label=f'Undet. Coeffs/Var. Params') # Plot results
plt.xlabel('x') # Label x-axis
plt.ylabel('y') # Label y-axis
plt.title('Solution Approximation using SciPy Odeint') # Set plot title
plt.legend() # Add legend
plt.show() # Display plot