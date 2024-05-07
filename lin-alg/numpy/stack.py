import numpy as np

a1 = np.array([[1,1], [2,2]])
a2 = np.array([[3,3], [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

vert_stack = np.vstack((a1, a2))
print(vert_stack)

horz_stack = np.hstack((a1, a2))
print(horz_stack)

horz_split_two = np.hsplit(horz_stack, 2)
print(horz_split_two)

horz_split_four = np.hsplit(horz_stack, 4)
print(horz_split_four)

horz_split_first = np.hsplit(horz_stack, [1])
print(horz_split_first)

vert_split_two = np.vsplit(vert_stack, 2)
print(vert_split_two)

vert_split_four = np.vsplit(vert_stack, 4)
print(vert_split_four)

# Start a subarray at each of the indices listed
vert_split_first_third = np.vsplit(vert_stack, [1,3])
print(vert_split_first_third)
