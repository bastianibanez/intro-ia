import time
import random
import numpy as np

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Tiempo de ejecución: {end-start}s")
        return result
    return wrapper

class Matrix:
    def __init__(self, dimension=3, matrix:list = [], min=1, max=100) -> None:
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
        return [[self._get_randint() for _ in range(self.dimension)] for _ in range(self.dimension)]
    
    def get_matrix(self) -> list:
        return self.matrix
    
    @timer
    def get_determinant_np(self):
        return round(np.linalg.det(self.get_matrix()))

    @timer
    def get_determinant_recursive(self):
        return RecursiveDeterminant.get(self)
    
    def __str__(self) -> str:
        out = ""
        for row in self.get_matrix():
            out += f"{" ".join(map(str, row))}\n"
        return "---\n" + out + "---\n" 

class RecursiveDeterminant:
    @staticmethod
    def _get_mult(pos:int, num:int):
        return (-1)**pos*num
    
    @staticmethod
    def _get_sub_matrix(pos:int, matrix: Matrix):
        mat = matrix.get_matrix()
        sub_mat = []
        for row in mat[1:]:
            sub_row = []
            for j, val in enumerate(row):
                if j == pos:
                    continue
                sub_row.append(val)
            sub_mat.append(sub_row)
        return Matrix(matrix.dimension-1, sub_mat)

    @staticmethod
    def get(matrix: Matrix):
        m = matrix.get_matrix()
        det = 0
    
        if len(m) == 2:
            a, b = m[0]
            c, d = m[1]
            return a*d - b*c
    
        for pos, num in enumerate(m[0]):
            mult = RecursiveDeterminant._get_mult(pos, num)
            sub_matrix = RecursiveDeterminant._get_sub_matrix(pos, matrix)
            det += mult*RecursiveDeterminant.get(sub_matrix)
        return det

class IterativeDeterminant:
    @staticmethod
    def add_row(row1: list, row2: list):
        if not len(row1) == len(row2):
            raise ValueError("Rows have different len")
        final_row = []
        for val1, val2 in zip(row1, row2):
            final_row.append(val1+val2)
        return final_row

        
if __name__ == '__main__': 
    # for i in range(2, 11):
    #     matrix = Matrix(i)
    
    #     print(f"{matrix.get_determinant_np() = }", end="\n\n")
    #     print(f"{matrix.get_determinant_recursive() = }", end="\n\n\n")

    matriz = Matrix(5)
    # determinante = matriz.get_determinant_recursive()
    # determinante2 = matriz.get_determinant_np()

    m = matriz.get_matrix()
    r1 = m[0]
    r2 = m[1]
    rs = IterativeDeterminant.add_row(r1, r2)

    print(r1, r2, rs)
