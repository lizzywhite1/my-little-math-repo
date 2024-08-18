import numpy as np
'''
Simiple Gauss Elimination function which uses partial pivoting to solve an augmented 
matrix. Partial pivoting involves exchanging pivot row with rows below if the pivot 
is smaller than any elements below it.

In my function the row reduction cost O(n^3) and the forward/backward substitution
cost O(n^2). 
'''
def gauss_elimination(A_matrix, b_matrix): 
    # Certain conditions must be satisfied: 1) Square A_Matrix 2) 2) b matrix is nx1
    if A_matrix.shape[0] != A_matrix.shape[1]: 
        print('ERROR: The A matrix must be square')
        return 
    if (b_matrix.shape[0] != A_matrix.shape[0]) or (b_matrix.shape[1]!=1): 
        print('ERROR: The b matrix must be a column vector with n rows')
        return
    
    # Initilize variables
    n = len(b_matrix) # Number of rows/columns-1 of aug matrix 
    m = n-1 
    i = 0 # Pointer
    x = np.zeros(n) # Solution vector (nx1 matrix)

    # Create augmented matrix 
    aug_matrix = np.concatenate((A_matrix,b_matrix),axis =1,dtype = float)
    print(f'Augmented matrix: \n{aug_matrix}')
    print('Solving for upper triangular matrix...\n')

    # Apply Gauss Elimination (w/ row exchange if necessary)
    while i < n: 
        
        # Partial Pivoting
        for p in range(i+1,n): 
            if abs(aug_matrix[i,i]) < abs(aug_matrix[p,i]): 
                aug_matrix[[p,i]] = aug_matrix[[i,p]] #Swap rows p and i 
        
        # Check for 0 in diagonals (ie. cannot find distinct solution)
        if aug_matrix[i,i] == 0: 
            print('Divide by zero error: No/Infinitely many solutions')
            return 
        # Convert all elements below pivot to zero
        for j in range(i+1,n): 
            scaling_factor = (aug_matrix[j,i]/aug_matrix[i,i])
            aug_matrix[j] = aug_matrix[j] - (scaling_factor*aug_matrix[i])
        i = i + 1 
    print(f'Upper Triangular Matrix: \n{aug_matrix}')
    # Back substitution; find x matrix 
    x[m] = aug_matrix[m,n] / aug_matrix[m,m] # Get the last element of x-matrix 
    for j in range(n-2,-1,-1): # Work backwards starting with second last row 
        x[j] = aug_matrix[j,n] # Initialize j-th elemenent of x to be b[j]
        # Subtract known variables from x[j]
        for k in range(j+1,n): # Iterate through columns j+1 to n
            x[j] = x[j] - aug_matrix[j,k] * x[k]
        x[j] = x[j] / aug_matrix[j,j] # Solve for new x[j]
    print(f'The solution to above augmented matrix is: \n{x}')

# Matrix with solution provided pivot
A = np.array([[2,4,-2,-2], [1,2,4,-3], [-3,-3,8,-2], [-1,1,6,-3]])
b = np.array ([[-4],[5],[7],[7]])
gauss_elimination(A,b)

# Matrix with infinitely many solutions 
A = np.array([
    [0, 1, -1],
    [1, 0, 2],
    [0, -3, 3]
])
b = np.array([[3],[2],[-9]])

gauss_elimination(A,b)

# Matrix with no solutions (r1 + r2 = r3)
A = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [2, 3, 2]
])
b = np.array([[1],[2],[0]])
gauss_elimination(A,b)