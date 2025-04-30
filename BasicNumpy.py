"""
NumPy Basics
------------

What is NumPy?
NumPy (Numerical Python) is a powerful Python library used for numerical computations. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures efficiently.

Key Features:
- Efficient storage and operations on large arrays of numeric data
- Mathematical functions: linear algebra, statistics, Fourier transform
- Random number generation
- Broadcasting and vectorization for speed



"""

# np.array
import numpy as np

# Creating a 1D array using np.array
a = np.array([1,2,3])
print("1D array a:", a)

# Creating a 2D array
b = np.array([[1,2,3],[4,5,6]])
print("2D array b:\n", b)

# Creating a 3D array
c = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print("3D array c:\n", c)

# Creating an array with specified dtype
print("Array with dtype float:", np.array([1,2,3], dtype=float))

# np.arange creates an array from start to stop with a step
# Syntax: np.arange(start, stop, step)
print("np.arange from 1 to 10 with step 2:", np.arange(1, 11, 2))

# Using reshape to change the shape of the array
print("Reshaped array from np.arange(16):\n", np.arange(16).reshape(2, 2, 2, 2))

# np.ones creates an array filled with ones
# Shape is (3, 4) here
print("Array of ones with shape (3,4):\n", np.ones((3, 4)))

# np.zeros creates an array filled with zeros
print("Array of zeros with shape (3,4):\n", np.zeros((3, 4)))

# np.random.random creates an array with random floats in [0, 1)
print("Random array with shape (3,4):\n", np.random.random((3, 4)))

# np.linspace creates linearly spaced numbers between start and stop
print("Linearly spaced integers between -10 and 10 (10 points):", np.linspace(-10, 10, 10, dtype=int))

# np.identity creates an identity matrix
print("Identity matrix of size 3:\n", np.identity(3))

"""### Array Attributes"""

a1 = np.arange(10,dtype=np.int32)
a2 = np.arange(12,dtype=float).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

print("3D array a3:\n", a3)

# .ndim: Number of dimensions
print("Number of dimensions (a3.ndim):", a3.ndim)

# .shape: Shape of the array
print("Shape of a3:", a3.shape)
print("Array a3:\n", a3)

# .size: Total number of elements
print("Total number of elements in a2:", a2.size)
print("Array a2:\n", a2)

# .itemsize: Memory size of each element in bytes
print("Memory size per element in a3 (bytes):", a3.itemsize)

# .dtype: Data type of elements
print("Data type of a1:", a1.dtype)
print("Data type of a2:", a2.dtype)
print("Data type of a3:", a3.dtype)

"""### Changing Datatype"""

# astype creates a copy of array with specified dtype
print("a3 converted to int32:\n", a3.astype(np.int32))

"""### Array Operations"""

a1 = np.arange(12).reshape(3,4)
a2 = np.arange(12,24).reshape(3,4)

print("Array a2:\n", a2)

# scalar operations

# arithmetic: square each element in a1
print("a1 squared:\n", a1 ** 2)

# relational: check where a2 elements equal 15
print("Elements of a2 equal to 15:\n", a2 == 15)

# vector operations
# arithmetic: element-wise power of a1 to a2
print("a1 raised to the power of a2 element-wise:\n", a1 ** a2)

"""### Array Functions"""

a1 = np.random.random((3,3))
a1 = np.round(a1*100)
print("Random array a1 rounded:\n", a1)

# max/min/sum/prod
# 0 -> col and 1 -> row
print("Product of elements in a1 along axis 0 (columns):", np.prod(a1, axis=0))

# mean/median/std/var
print("Variance of elements in a1 along axis 1 (rows):", np.var(a1, axis=1))

# trigonometric functions
print("Sine of elements in a1:\n", np.sin(a1))

# dot product
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(12,24).reshape(4,3)

print("Dot product of a2 and a3:\n", np.dot(a2, a3))

# log and exponents
print("Exponential of elements in a1:\n", np.exp(a1))

# round/floor/ceil
print("Ceiling of random values multiplied by 100:\n", np.ceil(np.random.random((2, 3)) * 100))

"""### Indexing and Slicing"""

# Array indexing and slicing
# Syntax: array[start:stop:step] or array[row_index, column_index]

a1 = np.arange(10)
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

print("3D array a3:\n", a3)

print("1D array a1:", a1)

print("2D array a2:\n", a2)

print("Element at row 1, column 0 of a2:", a2[1, 0])

print("3D array a3:\n", a3)

print("Element at [1, 0, 1] in a3:", a3[1, 0, 1])

print("Element at [1, 1, 0] in a3:", a3[1, 1, 0])

print("1D array a1:", a1)

print("Slicing a1 from index 2 to 5 with step 2:", a1[2:5:2])

print("2D array a2:\n", a2)

print("Slicing a2 rows 0 to 2 and columns from 1 with step 2:\n", a2[0:2, 1::2])

print("Slicing a2 every 2nd row and columns from 1 with step 2:\n", a2[::2, 1::2])

print("Slicing row 1 and every 3rd column of a2:", a2[1, ::3])

print("All columns of row 0 in a2:", a2[0, :])

print("All rows of column 2 in a2:", a2[:, 2])

print("Slicing rows from 1 and columns 1 to 3 in a2:\n", a2[1:, 1:3])

a3 = np.arange(27).reshape(3,3,3)
print("3D array a3 reshaped to (3,3,3):\n", a3)

print("Slicing a3 with steps on first and last dimensions:\n", a3[::2, 0, ::2])

print("Slicing a3 at index 2, rows 1 onwards, columns 1 onwards:\n", a3[2, 1:, 1:])

print("Slicing a3 at index 0, row 1, all columns:\n", a3[0, 1, :])

"""### Iterating"""

print("Iterating over 1D array a1:")
for i in a1:
  print(i)

print("Iterating over 2D array a2:")
for i in a2:
  print(i)

print("Iterating over 3D array a3:")
for i in a3:
  print(i)

print("Iterating over all elements of 3D array a3 using nditer:")
for i in np.nditer(a3):
  print(i)

"""### Reshaping"""

# reshape

# Transpose
print("Transpose of a2 using np.transpose:\n", np.transpose(a2))
print("Transpose of a2 using .T attribute:\n", a2.T)

# ravel flattens the array
print("Flattened a3 using ravel():", a3.ravel())

"""### Stacking"""

# horizontal stacking
a4 = np.arange(12).reshape(3,4)
a5 = np.arange(12,24).reshape(3,4)
print("Array a5:\n", a5)

print("Horizontal stacking of a4 and a5:\n", np.hstack((a4,a5)))

# Vertical stacking
print("Vertical stacking of a4 and a5:\n", np.vstack((a4,a5)))

"""### Useful Missing Functions"""

# np.eye: Identity matrix
print("Identity matrix using np.eye(3):\n", np.eye(3))  # 3x3 identity matrix

# np.full: Create an array filled with a constant value
print("Array filled with 7s of shape (2,3):\n", np.full((2, 3), 7))  # 2x3 array filled with 7

# np.where: Conditional element selection
a = np.array([1, 2, 3, 4, 5])
print("Conditional selection with np.where (a > 3):", np.where(a > 3, 'Yes', 'No'))  # returns ['No' 'No' 'No' 'Yes' 'Yes']

"""### Splitting"""

print("Array a4:\n", a4)

# Horizontal splitting: splits array along columns into equal parts
print("Horizontal splitting of a4 into 2 parts:\n", np.hsplit(a4, 2))  # Split into 2 equal parts since a4 has 4 columns

print("Array a5:\n", a5)

# Vertical splitting: splits array along rows
print("Vertical splitting of a5 into 3 parts:\n", np.vsplit(a5, 3))  # Splits into 3 equal parts row-wise
