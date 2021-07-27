import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385]
                  ,[8.9, -1.81, 0.2]
                  ,[1.41, 1.051, 0.026]] #Relatively more correct as zero index is more greater thsn other
# layer_outputs = [4.8, 4.79, 4.25] ----> Relatively less correct as almost same

# E = math.e

# exp_values = np.exp(layer_outputs)


# print(np.sum(layer_outputs, axis=1, keepdims=True))
# # exp_values = []

# # for output in layer_outputs:
# #     exp_values.append(E**output)

# # print(exp_values)

# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims= True)

# # norm_base = sum(exp_values)
# # norm_values = []

# # for value in exp_values:
# #     norm_values.append(value/ norm_base)

# print(norm_values)
# # print(sum(norm_values))

# Doint the above can give overflow
