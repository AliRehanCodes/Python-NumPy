import numpy as np
import matplotlib.pyplot as plt
import random

arr_1D = np.array([1,2,3,4])
# print(arr_1D)
# print(type(arr_1D))
# print(arr_1D.ndim)

arr_2D = np.array([[1,2,3,4], [5,6,7,8]])
# print(arr_2D)
# print(type(arr_2D))
print(arr_2D[:, 2:3])

# print(arr_2D.size)
# print(arr_2D.shape)
# print(arr_2D.dtype)

ones_arr = np.ones((4,6), dtype=int)
# print(ones_arr)

zero_arr = np.zeros((2,3), dtype=int)
# print(zero_arr)

empty_arr = np.empty((2,3))
# print(empty_arr)

arange_arr = np.arange(1,17, 2)
# print(my_arr)

reshape_arr = np.arange(1,17).reshape(4,4)
# print(reshape_arr)

revel_arr = np.arange(1,11).reshape(5,2).ravel()
# print(revel_arr)

linespace_arr = np.linspace(1,11,12)
# print(linespace_arr)

transpose_arr = arr_2D.transpose()
# print(transpose_arr)

arr1 = np.arange(1,10).reshape(3,3)
arr2 = np.arange(1,10).reshape(3,3)

add = arr1 + arr2
sub = arr1 - arr2
mul = arr1 * arr2
matric_multiplication = arr1 @ arr2

max_digit = arr1.argmax(axis= 1)
# print(max_digit)

mini_digit = arr2.min(axis=0)
# print(mini_digit)

sum_of_arr = np.sum(arr1)
# print(sum_of_arr)

np.mean(arr1)
np.sqrt(arr1)
np.std(arr1)
np.exp(arr1)
np.log(arr2)
np.log10(arr2)

arr_slicing = np.arange(1,101).reshape(10,10)

# print(arr_slicing[:, 0:1])
# print(arr_slicing[6,9])
# print(arr_slicing[1:4, 1:4])
# print(arr_slicing.shape)

Connection_arr = np.concatenate((arr1, arr2), axis=1)
# print(Connection_arr)

split_arr = np.array([1,2,3,4,5])
# print(np.split(split_arr, [1,3]))

x_value = np.arange(0,3*np.pi, 0.1)
y_sin = np.sin(x_value)

# plt.plot(x_value, y_sin)
# plt.show()

y_cos = np.cos(x_value)
# plt.plot(x_value, y_cos)
# plt.show()

y_tan = np.tan(x_value)
# plt.plot(x_value,y_tan)
# plt.show()

random_arr = np.random.random((3,3))
# print(random_arr)

randint_arr = np.random.randint(1,100, (5,4))
# print(randint_arr)

x = [1,2,3,4,5,6]
choice_arr = np.random.choice(x)
# print(choice_arr)

permutation_arr = np.random.permutation(x)
# print(permutation_arr)

# if we want to print same number than we'll use seed()

np.random.seed(10)
x = [1,2,3,4,5,6]
choice_arr = np.random.choice(x)
# print(choice_arr)

person_name = 'Ali Rehan Codes'
str1 = 'Hello'

# print(np.char.add(person_name, str1))
# print(np.char.center(str1, 60, fillchar="*"))

np.char.lower(str1)
np.char.upper(str1)
np.char.title(str1)
np.char.split(person_name)

str2 = 'dmy'
# print(np.char.join(':', str2))

