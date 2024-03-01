from tabulate import tabulate
from typing import Optional


def is_list_of_numbers(l: list):
    for item in l:
        try:
            item = float(item)
        except ValueError:
            return False
    return True

def count_left_zeros(l: list[float]):
    count = 0
    while l[count] == 0:
        count += 1
    return count

def has_trailing_zeros(l: list[float], start: Optional[int] = None, step: Optional[int] = None):
    if len(l) == 0:
        raise ValueError
    return not any(l[start::step])

class System:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.independent_terms_column = [0 for _ in rows]
        self.pivots = 0
        self.n = len(self.rows)
        self.m = len(self.rows[0])
        self.base_vectors = [[0 if i != j else 1 for i in range(self.n)] for j in range(self.m)]
        self.row_permutation = list(range(self.n))
        self.column_permutation = list(range(self.m))
        self.solved = False

    def get_row(self, i: int) -> list[float]:
        return self.rows[i]

    def get_rows(self, i: Optional[int] = None, j: Optional[int] = None):
        if i is None:
            i = 0
        s = slice(i, j)
        self.rows.__getitem__(s)

    def get_column(self, i) -> list[float]:
        return [row[i] for row in self.rows]

    def get_columns(self, i: Optional[int] = None, j: Optional[int] = None):
        if i is None:
            i = 0
        if j is None:
            j = self.m
        if j <= i:
            raise ValueError
        sliced_rows = [row[i:j] for row in self.rows]
        columns = []
        for j in range(0, j-i):
            col = [row[j] for row in sliced_rows]
            columns.append(col)
        return columns

    def order_by_pivot_columns(self):
        columns = self.get_columns()
        for col_idx, column in enumerate(columns):
            if col_idx < len(self.base_vectors) and column != self.base_vectors[col_idx] and self.base_vectors[col_idx] in columns:
                pivot_column_idx = column.index(self.base_vectors[col_idx])
                self.exchange_columns(col_idx, pivot_column_idx)
        return [columns.index(base_vector) for base_vector in self.base_vectors if base_vector in (columns := self.get_columns())]

    def get_pivot(self, row_idx: int):
        for number in self.rows[row_idx]:
            if number != 0:
                return number
        return 0

    def has_null_tail(self, col_idx: int, row_idx: int):
        column = self.get_column(col_idx)
        return has_trailing_zeros(column, row_idx)

    def move_null_rows_to_bottom(self):
        for row_idx, row in enumerate(self.rows):
            if has_trailing_zeros(row):
                self.rows.pop(row_idx)
                self.rows.append(row)

    def order_rows(self) -> None:
        self.rows.sort(reverse=True, key=lambda row: count_left_zeros(row))

    def normalize_pivot(self, row_idx: int):
        row = self.rows[row_idx]
        pivot = self.get_pivot(row_idx)
        if pivot == 0:
            raise ZeroDivisionError("Cannot normalize null row")
        self.rows[row_idx] = [value / pivot for value in row]

    def exchange_rows(self, i: int, j: int):
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
        self.row_permutation[i], self.row_permutation[j] = self.row_permutation[j], self.row_permutation[i]
        self.independent_terms_column[i], self.independent_terms_column[j] = self.independent_terms_column[j], self.independent_terms_column[i]

    def exchange_columns(self, i: int, j: int):
        column_1 , column_2 = self.get_column(i), self.get_column(j)
        for k, row in enumerate(self.rows):
            row[i], row[j] = column_2[k], column_1[k]
        self.column_permutation[i], self.column_permutation[j] = self.column_permutation[j], self.column_permutation[i]

    def rearrange_columns(self, permutation: list[int]):
        columns = [self.get_column(idx) for idx in permutation]
        for i, row in enumerate(self.rows):
            for j, _ in enumerate(row):
                row[j] = columns[j][i]

    def subtract_rows(self, i: int, j: int, multiplier: float):
        row_1, row_2 = self.rows[i], self.rows[j]
        result_row = [x - multiplier * y for x, y in zip(row_1, row_2)]
        self.rows[i] = result_row

    def run_elimination_on_column(self, col_idx: int, pivot_row_idx: int):
        self.normalize_pivot(pivot_row_idx)
        for row_idx in range(self.n):
            if row_idx == pivot_row_idx:
                continue
            multiplier = self.rows[row_idx][col_idx]
            self.subtract_rows(row_idx, pivot_row_idx, multiplier)

    def gauss_jordan(self):
        if self.solved:
            return
        pivot_count = 0
        self.order_rows()
        for col_idx in range(self.m):
            if pivot_count >= self.n:
                break
            if not self.has_null_tail(col_idx, pivot_count):
                current_column = self.get_column(col_idx)
                largest_column_entry = max(current_column[pivot_count:], key=lambda value: abs(value))
                self.exchange_rows(pivot_count, current_column.index(largest_column_entry, pivot_count))
                self.run_elimination_on_column(col_idx, pivot_count)
                pivot_count += 1
        self.move_null_rows_to_bottom()
        free_vars = [i for i in range(self.m) if i not in [columns.index(column) for column in self.base_vectors if column in (columns := self.get_columns())]]
        expressions = []
        for row in sorted(self.rows, key=lambda row: self.column_permutation[row.index(1)] if 1 in row else float("inf")):
            i = 0
            if not has_trailing_zeros(row):
                expression = []
                pivot_idx = row.index(1)
                if free_vars:
                    for col_idx in sorted(free_vars, key=lambda x: self.column_permutation[x]):
                        value = round(row[col_idx], 4)
                        if i == 0:
                            expression.append(f"{value}x_{self.column_permutation[col_idx]}")
                        elif value < 0:
                            expression.append(f"+ {abs(value)}x_{self.column_permutation[col_idx]}")
                        else:
                            expression.append(f"- {abs(value)}x_{self.column_permutation[col_idx]}")
                        i += 1
                    expression = f"x_{self.column_permutation[pivot_idx]} = " + " ".join(expression)
                    expressions.append(expression)
                else:
                    expressions.append(f"x_{self.column_permutation[pivot_idx]} = 0")
                    i += 1
        self.solved = True
        print(self, end=2*"\n")
        for exp in expressions: print(exp)
    def __str__(self) -> str:
        rows = []
        for row in self.rows:
            table_row = [rounded_value if (rounded_value := round(value, 4)) != 0 else abs(rounded_value) for value in row]
            rows.append(table_row)
        return tabulate(rows, floatfmt=".4f", tablefmt="presto")


rows = []
prev_coeffs = None
while len((cur_coeffs := input().split())) > 0:
    if not is_list_of_numbers(cur_coeffs):
        print("Invalid entry!")
        continue
    if prev_coeffs is not None and len(cur_coeffs) != len(prev_coeffs):
         print("Size mismatch!")
         continue
    for i, coeff in enumerate(cur_coeffs):
            cur_coeffs[i] = float(cur_coeffs[i])
    rows.append(cur_coeffs)
    prev_coeffs = cur_coeffs

if len(rows) > 0:
    system = System(rows)
    system.gauss_jordan()
