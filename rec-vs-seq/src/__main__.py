import time
import random
import numpy as np


def timer(func):
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Tiempo de ejecución [{func_name}]:\n{end - start}s")
        return result

    return wrapper


def validate_determinant_result(func):
    def wrapper(*args, **kwargs):
        matrix = args[0] or kwargs["matrix"]
        np_result = matrix.get_determinant_np()
        func_result = func(matrix)
        print(f"Valid result: {np_result == func_result}")
        return func_result

    return wrapper


class Matrix:
    def __init__(self, dimension=3, matrix: list = [], min=1, max=100) -> None:
        if dimension < 2:
            raise ValueError("Dimension debe ser mayor o igual a 2")

        self.min = min
        self.max = max
        self.dimension = dimension

        self.matrix = matrix
        if not matrix:
            self.matrix = self._build_matrix()

    def _get_randint(self) -> int:
        return random.randint(self.min, self.max)

    def _build_matrix(self) -> list:
        return [
            [self._get_randint() for _ in range(self.dimension)]
            for _ in range(self.dimension)
        ]

    def get_matrix(self) -> list:
        return self.matrix

    def get_determinant_np(self):
        return round(np.linalg.det(self.get_matrix()))

    # @timer
    def get_determinant_recursive(self):
        return RecursiveDeterminant.get(self)

    def get_determinant_memo(self):
        return MemoRecursiveDeterminant.get(self)

    # @timer
    def get_determinant_iterative(self):
        return IterativeDeterminant.get(self)

    def __str__(self) -> str:
        out = ""
        for row in self.get_matrix():
            out += f"{' '.join(map(str, row))}\n"
        return "---\n" + out + "---\n"


class RecursiveDeterminant:
    @staticmethod
    def _get_mult(pos: int, num: int):
        return (-1) ** pos * num

    @staticmethod
    def _get_sub_matrix(pos: int, matrix_list: list):
        sub_mat = []
        for row in matrix_list[1:]:
            sub_row = []
            for j, val in enumerate(row):
                if j == pos:
                    continue
                sub_row.append(val)
            sub_mat.append(sub_row)
        return sub_mat

    @staticmethod
    def recurse(matrix_list: list):
        det = 0

        if len(matrix_list) == 2:
            a, b = matrix_list[0]
            c, d = matrix_list[1]
            return a * d - b * c

        for pos, num in enumerate(matrix_list[0]):
            mult = RecursiveDeterminant._get_mult(pos, num)
            sub_matrix = RecursiveDeterminant._get_sub_matrix(pos, matrix_list)
            det += mult * RecursiveDeterminant.recurse(sub_matrix)
        return det

    @staticmethod
    # @validate_determinant_result
    def get(matrix: Matrix):
        matrix_list = matrix.get_matrix()
        result = RecursiveDeterminant.recurse(matrix_list)
        return round(result)


class MemoRecursiveDeterminant:
    @staticmethod
    def _get_mult(pos: int, num: int):
        return (-1) ** pos * num

    @staticmethod
    def _get_sub_matrix(pos: int, matrix_list: list):
        sub_mat = []
        for row in matrix_list[1:]:
            sub_row = []
            for j, val in enumerate(row):
                if j == pos:
                    continue
                sub_row.append(val)
            sub_mat.append(sub_row)
        return sub_mat

    @staticmethod
    def recurse(matrix_list: list, memo: dict):
        det = 0

        if len(matrix_list) == 2:
            a, b = matrix_list[0]
            c, d = matrix_list[1]
            return a * d - b * c

        sub_matrix_dimension = len(matrix_list) - 1

        for pos, num in enumerate(matrix_list[0]):
            mult = MemoRecursiveDeterminant._get_mult(pos, num)
            if sub_matrix_dimension in memo.keys():
                sub_matrix = memo[sub_matrix_dimension]
            else:
                sub_matrix = MemoRecursiveDeterminant._get_sub_matrix(pos, matrix_list)
                memo[sub_matrix_dimension] = sub_matrix
            det += mult * MemoRecursiveDeterminant.recurse(sub_matrix, memo)
        return det

    @staticmethod
    # @validate_determinant_result
    def get(matrix: Matrix):
        memo = {}
        matrix_list = matrix.get_matrix()
        result = MemoRecursiveDeterminant.recurse(matrix_list, memo)
        return round(result)


class IterativeDeterminant:
    @staticmethod
    def add_row(row1, row2):
        return [round(val1 + val2, 15) for val1, val2 in zip(row1, row2)]

    @staticmethod
    def multiply_row(mult, row):
        return [mult * val for val in row]

    @staticmethod
    def find_mult(pos, row_to_modify, row_to_multiply):
        mult = -row_to_modify[pos] / row_to_multiply[pos]
        new_row = IterativeDeterminant.multiply_row(mult, row_to_multiply)
        return new_row

    @staticmethod
    def zero_value(pos, row_to_modify, row_to_multiply):
        zero_row = IterativeDeterminant.find_mult(pos, row_to_modify, row_to_multiply)
        new_row = IterativeDeterminant.add_row(row_to_modify, zero_row)
        return new_row

    @staticmethod
    def get_triangular_form(matrix_list: list):
        i = 0
        while i < len(matrix_list):
            row_to_multiply = matrix_list[i]
            for row_idx in range(i + 1, len(matrix_list)):
                row_to_modify = matrix_list[row_idx]
                new_row = IterativeDeterminant.zero_value(
                    i, row_to_modify, row_to_multiply
                )
                matrix_list[row_idx] = new_row
            i += 1
        return matrix_list

    @staticmethod
    # @validate_determinant_result
    def get(matrix: Matrix):
        matrix_list = matrix.get_matrix()
        triangular_form = IterativeDeterminant.get_triangular_form(matrix_list)
        det = 1
        for i, _ in enumerate(triangular_form):
            det *= triangular_form[i][i]
        return round(det)


def compare_algs(min_dimension=3, max_dimension=10):
    for i in range(min_dimension, max_dimension + 1):
        matrix = Matrix(i)
        recursive_start = time.time()
        matrix.get_determinant_recursive()
        recursive_end = time.time()

        memo_recursive_start = time.time()
        matrix.get_determinant_recursive()
        memo_recursive_end = time.time()

        iterative_start = time.time()
        matrix.get_determinant_iterative()
        iterative_end = time.time()

        recursive_time = recursive_end - recursive_start
        memo_recursive_time = memo_recursive_end - memo_recursive_start
        iterative_time = iterative_end - iterative_start
        memo_percentage_diff = (100 * memo_recursive_time / recursive_time) - 100
        memo_percentage_diff = (
            f"+{memo_percentage_diff:5f}%"
            if memo_percentage_diff >= 0
            else f"{memo_percentage_diff:5f}%"
        )
        iterative_percentage_diff = (100 * iterative_time / recursive_time) - 100
        iterative_percentage_diff = (
            f"+{iterative_percentage_diff:5f}%"
            if iterative_percentage_diff >= 0
            else f"{iterative_percentage_diff:5f}%"
        )

        print(f"Dimension: {i}X{i}")
        print(f"Recursive Time: {recursive_time}s")
        print(
            f"Recursive Time (with memoization): {memo_recursive_time}s ({memo_percentage_diff})"
        )
        print(f"Iterative Time: {iterative_time}s ({iterative_percentage_diff})")


def test_iterative(min_dimension=3, max_dimension=10):
    for i in range(min_dimension, max_dimension + 1):
        matrix = Matrix(i)

        iterative_start = time.time()
        matrix.get_determinant_iterative()
        iterative_end = time.time()
        iterative_time = iterative_end - iterative_start

        print(f"Dimension: {i}X{i}")
        print(f"Iterative Time: {iterative_time}s")


def test_recursive(min_dimension=3, max_dimension=10):
    for i in range(min_dimension, max_dimension + 1):
        matrix = Matrix(i)

        recursive_start = time.time()
        matrix.get_determinant_recursive()
        recursive_end = time.time()
        recursive_time = recursive_end - recursive_start

        print(f"Dimension: {i}X{i}")
        print(f"Recursive Time: {recursive_time}s")


def test_memo(min_dimension=3, max_dimension=10):
    for i in range(min_dimension, max_dimension + 1):
        matrix = Matrix(i)

        memo_start = time.time()
        matrix.get_determinant_memo()
        memo_end = time.time()
        memo_time = memo_end - memo_start

        print(f"Dimension: {i}X{i}")
        print(f"Recursive Time (with memoization): {memo_time}s")


if __name__ == "__main__":
    # compare_algs(3, 15)
    test_recursive(3, 11)
    input("Press enter to continue...")
    test_memo(3, 11)
    input("Press enter to continue...")
    test_iterative(3, 100)
