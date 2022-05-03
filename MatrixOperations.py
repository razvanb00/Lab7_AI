def multiply(X, Y):
    result = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result


class Matrix:
    def transpose_matrix(self, m):
        result = [[0 for i in range(len(m))] for j in range(len(m[0]))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                result[j][i] = m[i][j]
        return result

    def get_minor(self, m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def get_determinant(self, m):
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * self.get_determinant(self.get_minor(m, 0, c))
        return determinant

    def get_inverse(self, m):
        determinant = self.get_determinant(m)
        if determinant != 0:
            # special case for 2x2 matrix:
            if len(m) == 2:
                return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                        [-1 * m[1][0] / determinant, m[0][0] / determinant]]

            cofactors = []
            for r in range(len(m)):
                cofactorRow = []
                for c in range(len(m)):
                    minor = self.get_minor(m, r, c)
                    cofactorRow.append(((-1) ** (r + c)) * self.get_determinant(minor))
                cofactors.append(cofactorRow)
            cofactors = self.transpose_matrix(cofactors)
            for r in range(len(cofactors)):
                for c in range(len(cofactors)):
                    cofactors[r][c] = cofactors[r][c] / determinant
            return cofactors