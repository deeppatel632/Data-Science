"""
This script demonstrates advanced NumPy functionalities such as performance comparison with Python lists, memory efficiency, advanced indexing, broadcasting rules, mathematical functions, handling missing values, and plotting using Matplotlib.
"""

### Numpy array vs Python lists

# Comparing speed of addition operation between Python lists and NumPy arrays

# Creating two large Python lists with 10 million elements each
a = [i for i in range(10000000)]
b = [i for i in range(10000000,20000000)]

c = []
import time

# Timing the addition of elements from two lists using a for loop
start = time.time()
for i in range(len(a)):
  c.append(a[i] + b[i])
print(time.time()-start)

# Using NumPy arrays for the same operation
import numpy as np
a = np.arange(10000000)
b = np.arange(10000000,20000000)

# Timing the addition using vectorized NumPy operation (much faster)
start = time.time()
c = a + b
print(time.time()-start)

3.26/0.06

# Comparing memory usage of Python lists vs NumPy arrays

# Creating a large Python list
a = [i for i in range(10000000)]

import sys

# Checking memory size of the Python list
sys.getsizeof(a)

# Creating a NumPy array with int8 datatype (more memory efficient)
a = np.arange(10000000,dtype=np.int8)
# Checking memory size of the NumPy array
sys.getsizeof(a)

# convenience

"""### Advanced Indexing"""

# Normal Indexing and slicing
# Creating a 2D NumPy array with shape (6,4)
a = np.arange(24).reshape(6,4)
a

# Accessing element at row 1, column 2
a[1,2]

# Slicing rows 1 to 2 and columns 1 to 2 (subarray extraction)
a[1:3,1:3]

# Fancy Indexing
# Selecting specific columns (0, 2, and 3) for all rows using fancy indexing
a[:,[0,2,3]]

# Boolean Indexing
# Creating a 6x4 array with random integers between 1 and 100
a = np.random.randint(1,100,24).reshape(6,4)
a

# Find all numbers greater than 50 using boolean mask
a[a > 50]

# Find all even numbers using boolean mask
a[a % 2 == 0]

# Find all numbers greater than 50 and even
a[(a > 50) & (a % 2 == 0)]

# Find all numbers not divisible by 7
a[~(a % 7 == 0)]

"""### Broadcasting

The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations.

The smaller array is “broadcast” across the larger array so that they have compatible shapes.
"""

# Example of broadcasting with arrays of the same shape
a = np.arange(6).reshape(2,3)
b = np.arange(6,12).reshape(2,3)

print(a)
print(b)

# Element-wise addition since shapes match exactly
print(a+b)

# Example of broadcasting with arrays of different shapes
a = np.arange(6).reshape(2,3)
b = np.arange(3).reshape(1,3)

print(a)
print(b)

# b is broadcasted along the first axis to match shape of a
print(a+b)

"""#### Broadcasting Rules

**1. Make the two arrays have the same number of dimensions.**<br>
- If the numbers of dimensions of the two arrays are different, add new dimensions with size 1 to the head of the array with the smaller dimension.<br>

**2. Make each dimension of the two arrays the same size.**<br>
- If the sizes of each dimension of the two arrays do not match, dimensions with size 1 are stretched to the size of the other array.
- If there is a dimension whose size is not 1 in either of the two arrays, it cannot be broadcasted, and an error is raised.

<img src = "https://jakevdp.github.io/PythonDataScienceHandbook/figures/02.05-broadcasting.png">
"""

# More examples demonstrating broadcasting rules and behavior

a = np.arange(12).reshape(4,3)
b = np.arange(3)

print(a)
print(b)

# b is broadcasted along rows to match shape of a
print(a+b)

a = np.arange(12).reshape(3,4)
b = np.arange(3).reshape(3,1)

print(a)
print(b)

# b is broadcasted along columns to match shape of a
print(a+b)

a = np.arange(3).reshape(1,3)
b = np.arange(3).reshape(3,1)

print(a)
print(b)

# Both arrays broadcast to (3,3) shape for addition
print(a+b)

a = np.arange(3).reshape(1,3)
b = np.arange(4).reshape(4,1)

print(a)
print(b)

# Broadcasting to (4,3) shape for addition
print(a + b)

a = np.array([1])
# shape -> (1,1)
b = np.arange(4).reshape(2,2)
# shape -> (2,2)

print(a)
print(b)

# Scalar broadcasted to (2,2)
print(a+b)

a = np.arange(12).reshape(3,4)
b = np.arange(4).reshape(1,4)

print(a)
print(b)

# b broadcasted along rows to match a
print(a + b)

a = np.arange(16).reshape(4,4)
b = np.arange(4).reshape(2,2)

print(a)
print(b)

# This addition would raise an error due to incompatible shapes
# print(a+b)

"""### Working with mathematical formulas"""

# Creating an array and applying sine function element-wise
a = np.arange(10)
np.sin(a)

# sigmoid
# Defining the sigmoid function, commonly used in machine learning for activation
def sigmoid(array):
  return 1/(1 + np.exp(-(array)))


a = np.arange(100)

# Applying sigmoid function to array elements
sigmoid(a)

# mean squared error
# Defining mean squared error function to measure average squared difference between actual and predicted values
actual = np.random.randint(1,50,25)
predicted = np.random.randint(1,50,25)

def mse(actual,predicted):
  return np.mean((actual - predicted)**2)

# Calculating MSE between actual and predicted arrays
mse(actual,predicted)

# binary cross entropy
# Computing mean squared error again as an example (usually binary cross entropy is different)
np.mean((actual - predicted)**2)

actual

"""### Working with missing values"""

# Working with missing values -> np.nan
a = np.array([1,2,3,4,np.nan,6])
a

# Filtering out NaN values using boolean indexing
a[~np.isnan(a)]

"""### Plotting Graphs"""

# plotting a 2D plot
# x = y (linear function)
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = x

plt.plot(x,y)

# y = x^2 (quadratic function)
x = np.linspace(-10,10,100)
y = x**2

plt.plot(x,y)

# y = sin(x) (trigonometric function)
x = np.linspace(-10,10,100)
y = np.sin(x)

plt.plot(x,y)

# y = xlog(x) (logarithmic function scaled by x)
x = np.linspace(-10,10,100)
y = x * np.log(x)

plt.plot(x,y)

# sigmoid function plot
x = np.linspace(-10,10,100)
y = 1/(1+np.exp(-x))

plt.plot(x,y)

"""### Meshgrids"""

# Meshgrids

