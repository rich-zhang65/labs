import numpy as np
from sympy import symbols, Eq, Matrix, solve, sympify

def swap_rows(m, row_index_1, row_index_2):
    m = m.copy()
    m[[row_index_1, row_index_2]] = m[[row_index_2, row_index_1]]
    return m

def get_index_first_non_zero_value_from_column(m, column, starting_row):
    column_array = m[starting_row:,column]
    for i, val in enumerate(column_array):
        if not np.isclose(val, 0, atol = 1e-5):
            index = i + starting_row
            return index
    return -1

def get_index_first_non_zero_value_from_row(m, row, augmented = False):
    m = m.copy()
    if augmented == True:
        m = m[:,:-1]
        
    row_array = m[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1

# b must be array of single value arrays, i.e. [[1], [2], ...] 
def augmented_matrix(a, b):
    augmented_m = np.hstack((a, b))
    return augmented_m

def string_to_augmented_matrix(equations):
    equation_list = equations.split('\n')
    equation_list = [x for x in equation_list if x != '']
    coefficients = []
    
    ss = ''
    for c in equations:
        if c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in ss:
                ss += c + ' '
    ss = ss[:-1]
    
    variables = symbols(ss)
    for equation in equation_list:
        sides = equation.replace(' ', '').split('=')
        left_side = sympify(sides[0])
        coefficients.append([left_side.coeff(variable) for variable in variables])
        coefficients[-1].append(int(sides[1]))

    augmented_matrix = Matrix(coefficients)
    augmented_matrix = np.array(augmented_matrix).astype("float64")

    A, B = augmented_matrix[:,:-1], augmented_matrix[:,-1].reshape(-1,1)
    
    return ss, A, B
