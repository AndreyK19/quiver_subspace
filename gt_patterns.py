import numpy as np
from itertools import product, permutations
from collections import defaultdict
from sympy import Symbol, Rational
from sympy.physics.quantum import TensorProduct
import numpy as np
from ast import literal_eval

from utils import is_linearly_indep

def generate_basis_irrep(lambda_vec):
    # lambda_vec must satisfy lambda[j] >= lambda[j + 1]
    n = len(lambda_vec)
    prev_basis = [[lambda_vec]]
    for _ in range(n - 1):
        curr_basis = []
        for v in prev_basis:
            last_row = v[-1]
            for l in product(*[range(last_row[j + 1], last_row[j] + 1) for j in range(len(last_row) - 1)]):
                curr_basis.append(v + [list(l)])
        prev_basis = curr_basis
    
    curr_basis = [b[::-1] for b in curr_basis]
    symbol_dict = {str(b): Symbol(str(b), commutative=False) for b in curr_basis}
    return symbol_dict

def generate_basis_tensor_prod(lambda_mat):
    bases = [list(generate_basis_irrep(l).values()) for l in lambda_mat]
    combinations = list(product(*bases))
    basis = {TensorProduct(*c): i for i, c in enumerate(combinations)}
    return basis

def highest_weight_vector_irrep(lambda_vec):
    l = [lambda_vec[:i+1] for i in range(len(lambda_vec))]
    return Symbol(str(l), commutative=False)

def highest_weight_vector_tensor_prod(lambda_mat):
    return TensorProduct(*[highest_weight_vector_irrep(l) for l in lambda_mat])

def extremal_vector(lambda_vec, sigma):
    hw = highest_weight_vector_irrep(lambda_vec)
    pattern_list = literal_eval(hw.name)
    for idx, row in enumerate(pattern_list):
        k = len(row)
        pattern_list[idx] = sorted([lambda_vec[sigma[i]] for i in range(k)], reverse=True)
    return Symbol(str(pattern_list), commutative=False)

def generating_set(lambda_mat):
    n = len(lambda_mat)
    gen_set = []
    for sigma in list(permutations(range(n))):
        tensor_factors = [extremal_vector(lambda_vec, sigma) for lambda_vec in lambda_mat]
        gen_set.append(TensorProduct(*tensor_factors))
    gen_set = list(dict.fromkeys(gen_set))
    return gen_set

def operator(a, n, offset=1):
    row = [a + offset, a] if a != 0 else [1, n]
    op = np.tile(row, (n, 1))
    op[a, :] = 0
    return op

def weight_basis_vec_irrep(pattern):
    pattern_list = literal_eval(pattern.name)
    n = len(pattern_list)
    w = [sum(pattern_list[k]) - sum(pattern_list[k - 1]) if k > 0 else pattern_list[0][0] for k in range(n)]
    return w

def weight_basis_vec_tensor_prod(vec):
    weights = []
    for p in vec.args:
        weight = weight_basis_vec_irrep(p)
        weights.append(weight)
    weight_total = np.array(weights).sum(axis=0)
    return weight_total

def weight_vec_tensor_prod(vec):
    weights = []
    for summand in vec.as_ordered_terms():
        _, basis_vec = summand.as_coeff_Mul()
        weight = weight_basis_vec_tensor_prod(basis_vec)
        weights.append(weight)
    
    assert(all([np.all(x == weights[0]) for x in weights]))
    return weights[0]

def sort_gen_set_by_weight(gen_set):
    weights_dict = defaultdict(list)
    vecs_with_ranks = [(weight_vec_tensor_prod(v), v) for v in gen_set]
    for (weight, v) in vecs_with_ranks:
        weights_dict[str(weight.tolist())].append(v)
    return dict(weights_dict)

def verify_pattern(pattern):
    if isinstance(pattern, Symbol):
        pattern = literal_eval(pattern.name)

    is_valid = True
    n = len(pattern)
    for k in range(n - 1):
        if not pattern[k] == sorted(pattern[k], reverse=True):
            return False
        for i in range(k + 1):
            is_valid = is_valid and pattern[k][i] <= pattern[k + 1][i] and pattern[k][i] >= pattern[k + 1][i + 1]
            if not is_valid:
                return False
    return True

def add_to_row(pattern, k, i, offset):
    pattern_list = literal_eval(pattern.name)
    pattern_list[k - 1][i] = pattern_list[k - 1][i] + offset
    if verify_pattern(pattern_list):
        return Symbol(str(pattern_list), commutative=False)
    else:
        return 0 

def apply_operator_basis_vec_irrep(pattern, op):
    n = len(literal_eval(pattern.name))
    if np.all(op == np.array([1, n])):
        res = apply_operator_irrep_1n(pattern)
    else:
        res = apply_operator_basis_vec_irrep_temp(pattern, op)
    return res

def apply_operator_basis_vec_irrep_temp(pattern, op):
    if np.all(op == 0): return 0

    pattern_list = literal_eval(pattern.name)
    offset = int(op[0] - op[1])
    # offset = 1 means op = E_(k+1, k), offset = -1 means op = E_(k, k+1)
    k = int(op[0]) if offset == -1 else int(op[1])

    l_k = np.array(pattern_list[k - 1]) - np.arange(k)

    if offset == -1 or k != 1:
        l_offset = np.array(pattern_list[k - 1 - offset]) - np.arange(k - offset)
    
    res = 0
    for i in range(k):
        if k == 1 and offset == 1:
            coeff = Rational(1)
        else:
            num = np.prod(l_k[i] - l_offset)
            den_temp = l_k[i] - l_k
            den = np.prod(den_temp[den_temp != 0])
            coeff = Rational(num, den)

        b = add_to_row(pattern, k, i, -offset)

        res = res + coeff * b

    res = offset * res
    return res

def get_op_1n(n):
    op = Symbol(f"{n}", commutative=False)
    for j in reversed(range(2, n)):
        s_j = Symbol(f"{j}", commutative=False)
        op = s_j * op - op * s_j
        op = op.expand()
    return op

def apply_operator_irrep_1n(pattern):
    pattern_list = literal_eval(pattern.name)
    n = len(pattern_list)
    res = 0
    op_1n = get_op_1n(n)
    for s in op_1n.as_ordered_terms():
        coeff, op_prod = s.as_coeff_Mul()
        res_temp = pattern

        for op in reversed(op_prod.args):
            op = int(op.name)
            res_temp = apply_operator_irrep(res_temp.expand(), [op - 1, op])
            if res_temp == 0: break

        res = res + coeff * res_temp
    return res

def apply_operator_irrep(vec, op):
    # if np.all(op == 0) or vec == 0: return 0
    if vec == 0:
        return 0
    if np.all(op == 0):
        return 0

    res = 0
    # Iterate over summands
    for summand in vec.as_ordered_terms():
        coeff, basis_vec = summand.as_coeff_Mul()
        basis_vec_new = apply_operator_basis_vec_irrep(basis_vec, op)
        res = res + coeff * basis_vec_new

    return res

def apply_operator_tensor_product(vec, op):
    # Apply op on a tensor. op is n x 2, where the row k = [i, j] corresponds to applying E_ij on the exterior product in the vertex k
    n = op.shape[0]

    res = 0
    for summand in vec.as_ordered_terms():
        coeff, basis_vec = summand.as_coeff_Mul()
        
        for idx_rep in range(n):
            basis_vec_temp = basis_vec
            if isinstance(basis_vec_temp, TensorProduct):
                basis_vecs_irrep = list(basis_vec_temp.args)
            elif isinstance(basis_vec_temp, Symbol):
                basis_vecs_irrep = [basis_vec_temp]
            
            basis_vecs_irrep[idx_rep] = apply_operator_irrep(basis_vecs_irrep[idx_rep], op[idx_rep, :])
            res = res + coeff * TensorProduct(*basis_vecs_irrep)
    
    res = res.expand()
    return res

def modify_weight(w, op):
    if isinstance(w, str):
        w = literal_eval(w)
    n = len(w)
    a = int(np.unique(op[:, 1][op[:, 1] != 0])) - 1
    if a == n - 1:
        w[0] = w[0] + 1
        w[n - 1] = w[n - 1] - 1
    else:
        w[a] = w[a] - 1
        w[a + 1] = w[a + 1] + 1
    return str(w) 


def generate_all_vecs(lambda_mat, gen_vecs=None):
    n = len(lambda_mat[0])
    basis_tensor_prod = generate_basis_tensor_prod(lambda_mat)

    if gen_vecs is None:
        v_0 = highest_weight_vector_tensor_prod(lambda_mat)
        gen_vecs = [v_0]

    operators = [operator(a, n) for a in range(n)]

    current_basis = sort_gen_set_by_weight(gen_vecs)

    vecs_to_check = [(k, v) for k, values in current_basis.items() for v in values]
    
    while len(vecs_to_check) > 0:
        (w, vec_to_check) = vecs_to_check.pop()
        for op in operators:
            new_vec = apply_operator_tensor_product(vec_to_check, op)
            new_vec = new_vec.simplify().expand(tensorproduct=True)

            if new_vec == 0: continue

            w_new_vec = modify_weight(w, op)
            w_expected = weight_vec_tensor_prod(new_vec)

            assert str(w_expected.tolist()) == w_new_vec

            if w_new_vec not in current_basis.keys():
                current_basis[w_new_vec] = [new_vec]
                vecs_to_check.append((w_new_vec, new_vec))
            else:
                vecs_new_weight = current_basis[w_new_vec]
                if new_vec in vecs_new_weight: continue

                vecs_new_weight_temp = vecs_new_weight + [new_vec]
                if is_linearly_indep(vecs_new_weight_temp, basis_tensor_prod):
                    current_basis[w_new_vec] = vecs_new_weight_temp
                    vecs_to_check.append((w_new_vec, new_vec))

    concatenated_list = [item for sublist in current_basis.values() for item in sublist]
    dim = len(concatenated_list)
    return current_basis, dim
