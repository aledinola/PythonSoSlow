# This program demonstrates how to sum the elements of a 2D NumPy array using row-major 
# order traversal.
import numpy as np 
matrix = np.random.randint(1, 101, size=(10, 10))

def row_major_loop(matrix):
    result = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result = result + matrix[i, j]
    return result

result = row_major_loop(matrix)
print(result)

