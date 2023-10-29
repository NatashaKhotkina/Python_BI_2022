import numpy as np


def matrix_multiplication(x1, x2):
    return np.matmul(x1, x2)


def multiplication_check(matrices):
    for idx in range(len(matrices) - 1):
        if matrices[idx].shape[1] != matrices[idx + 1].shape[0]:
            return False
    return True


def multiply_matrices(matrices):
    if not multiplication_check(matrices):
        return None
    else:
        return np.linalg.multi_dot(matrices)


def compute_2d_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)


def compute_multidimensional_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)


def compute_pair_distances(arr):
    n = arr.shape[0]
    answer = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            answer[i, j] = np.linalg.norm(arr[i] - arr[j])
    return answer


if __name__ == "__main__":
    first_array = np.full((3, 3), 1)
    second_array = np.eye(3)
    third_array = np.arange(0, 20, 2).reshape(2, 5)
