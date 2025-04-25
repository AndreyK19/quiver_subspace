import numpy as np
from itertools import combinations
from sympy import ImmutableSparseMatrix
import numpy as np

def generate_coeff_mat(vec_set, basis):
    row_idx = []
    col_idx = []
    val = []
    for idx_vec, v in enumerate(vec_set):
        summands = v.as_ordered_terms()
        row_idx = row_idx + [idx_vec] * len(summands)
        for s in summands:
            coeff, b = s.as_coeff_Mul()

            col_idx.append(basis[b])
            val.append(coeff)

    return row_idx, col_idx, val

def get_new_basis_idx(col_idx):
    sorted_unique = sorted(set(col_idx))
    
    # Create a dictionary that maps each element to its rank
    rank_dict = {value: index for index, value in enumerate(sorted_unique)}
    
    # Replace each element in the list by its rank
    return [rank_dict[element] for element in col_idx]

def is_linearly_indep(vec_set, basis):
    row_idx, col_idx, val = generate_coeff_mat(vec_set, basis)

    basis_vec_idx = sorted(list(set(col_idx)))
    col_idx_new = get_new_basis_idx(col_idx)

    idx = {(r, c): v for r, c, v in zip(row_idx, col_idx_new, val)}
    M = ImmutableSparseMatrix(len(vec_set), len(basis_vec_idx), idx)

    is_indep = M.rank() == M.shape[0]
    return is_indep

def calc_expected_dim(lambda_mat):
    lambda_mat = np.array(lambda_mat)
    k_mat = -np.diff(lambda_mat, axis=1)
    n = k_mat.shape[0]
    idx_rep_vec = k_mat.sum(axis=0)

    comb = list(combinations(range(1, n + 1), 2))
    num = np.prod([sum(idx_rep_vec[x[0] - 1 : x[1] - 1]) + np.abs(x[1] - x[0]) for x in comb])
    den = np.prod([np.abs(x[1] - x[0]) for x in comb])

    dim = num / den
    assert dim == int(dim)
    return int(dim)