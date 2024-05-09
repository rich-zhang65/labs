import numpy as np
import tests
import utils

def test_utils():
    m = np.array([[1,3,6], [0,-5,2], [-4,5,8]])
    print(m)
    m_swapped = utils.swap_rows(m, 0, 2)
    print(m_swapped)

    n = np.array([[0,5,-3,6,8], [0,6,3,8,1], [0,0,0,0,0], [0,0,0,0,7], [0,2,1,0,4]])
    print(n)
    print(utils.get_index_first_non_zero_value_from_column(n, column = 0, starting_row = 0))
    print(utils.get_index_first_non_zero_value_from_column(n, column = -1, starting_row = 2))

    print(n)
    print(f'Output for row 2: {utils.get_index_first_non_zero_value_from_row(n, 2)}')
    print(f'Output for row 3: {utils.get_index_first_non_zero_value_from_row(n, 3)}')
    print(f'Output for row 3: {utils.get_index_first_non_zero_value_from_row(n, 3, augmented = True)}')

    a = np.array([[1,2,3], [3,4,5], [4,5,6]])
    b = np.array([[1], [5], [7]])
    print(utils.augmented_matrix(a, b))

    equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

    variables, a, b = utils.string_to_augmented_matrix(equations)

    sols = gaussian_elimination(a, b)

    if not isinstance(sols, str):
        for variable, solution in zip(variables.split(' '), sols):
            print(f"{variable} = {solution:.4f}")
    else:
        print(sols)

def row_echelon_form(a, b):  
    det_A = np.linalg.det(a)

    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    a = a.copy()
    b = b.copy()

    a = a.astype('float64')
    b = b.astype('float64')

    num_rows = len(a)

    m = utils.augmented_matrix(a, b)
    
    for row in range(num_rows):
        pivot_candidate = m[row, row]

        if np.isclose(pivot_candidate, 0) == True:
            first_non_zero_value_below_pivot_candidate = utils.get_index_first_non_zero_value_from_column(m, row, row)
            m = utils.swap_rows(m, row, first_non_zero_value_below_pivot_candidate)
            pivot = m[row, row]
        else:
            pivot = pivot_candidate
        
        m[row] = 1/pivot * m[row]

        for j in range(row + 1, num_rows): 
            value_below_pivot = m[j, row]
            m[j] = m[j] - value_below_pivot*m[row]

    return m

def back_substitution(m):
    m = m.copy()
    num_rows = m.shape[0]
    
    for row in reversed(range(num_rows)): 
        substitution_row = m[row]
        index = utils.get_index_first_non_zero_value_from_row(m, row, True)

        for j in range(row): 
            row_to_reduce = m[j]
            value = row_to_reduce[index]
            row_to_reduce = row_to_reduce - value * substitution_row
            m[j,:] = row_to_reduce

    solution = m[:,-1]
    
    return solution

def gaussian_elimination(a, b):
    row_echelon_m = row_echelon_form(a, b)

    if not isinstance(row_echelon_m, str): 
        solution = back_substitution(row_echelon_m)

    return solution

test_utils()

a = np.array([[1,2,3], [0,1,0], [0,0,5]])
b = np.array([[1], [2], [4]])
row_echelon = row_echelon_form(a, b)
print(row_echelon)

back_sub = back_substitution(row_echelon)
print(back_sub)

# Validate that the two results are equal
gaussian = gaussian_elimination(a, b)
print(gaussian)

print(tests.test_row_echelon_form(row_echelon_form))
print(tests.test_back_substitution(back_substitution))
print(tests.test_gaussian_elimination(gaussian_elimination))
