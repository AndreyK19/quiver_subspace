from gt_patterns import generating_set, generate_all_vecs
from utils import calc_expected_dim

if __name__ == '__main__':
    lambda_mat = [[1, 1, 0],
                  [3, 2, 0],
                  [8, 5, 0]]

    gen_set = generating_set(lambda_mat)
    basis, dim = generate_all_vecs(lambda_mat, gen_set)
    dim_expected = calc_expected_dim(lambda_mat)

    print(f'dim = {dim} \nexpected dim = {dim_expected}')
