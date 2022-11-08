import numpy as np
# Simulate Data
import numpy as np
from math import sqrt

# Generate theta* for linear regression
def generate_theta_star_lin(dim):
    rng1 = np.random.RandomState(0)
    theta_star = rng1.randn(dim)  # Random parameters
    theta_star = theta_star / np.linalg.norm(theta_star)  # Make norm = 1
    return (theta_star)

# Simulate training data for linear simulation
def sim_data_lin(dim, eps, n, rng, theta_star):
    b = np.random.binomial(size=n, n=1, p=eps)  # (n,)
    # Normalize by dimension to small norm
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = (1-b) * rng.normal(0, 1, size=n) + b * \
        rng.normal(0, sqrt(10), size=n)  # (n,)
    y = np.matmul(x, theta_star) + mu
    return x, y

# Simulate contaminated point for linear simulation
def sim_contaminated_lin(dim, n, rng, theta_star):
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = rng.normal(0, sqrt(10), size=n)  # (n,)
    y = np.matmul(x, theta_star) + mu  # (n,)
    return x, y

# Runs n_sim number of simulations computing population and empirical IF on one contaminated point
def if_diff_sim_lin(dim, x_sim, y_sim, oracle_theta_dict, lambda_, n, n_sim, seed, theta_star, if_orth_emp, if_orth_pop):
    oracle_theta = oracle_theta_dict[lambda_]
    if_emp_ls = []
    if_pop_ls = []
    rng = np.random.RandomState(seed)
    x_con_ls, y_con_ls = sim_contaminated_lin(dim, n_sim, rng, theta_star)
    for i in range(n_sim):
        x_con = x_con_ls[i]
        y_con = y_con_ls[i]
        if_emp_ls.append(if_orth_emp(x_sim, y_sim, x_con, y_con, lambda_, n))
        if_pop_ls.append(if_orth_pop(dim, x_con, y_con, oracle_theta, lambda_))
    return (if_emp_ls, if_pop_ls)

# run simulations for multiple lambda values
def run_sim_lin(n_sim, eps, n_ls, lambda_ls, oracle_theta_dict, dim, seed, theta_star, if_orth_emp, if_orth_pop):
    mean_diff_abs_total = []
    sd_diff_abs_total = []
    n_samp = []
    lambda_ = []
    rng = np.random.RandomState(1)
    for n in n_ls:
        x_sim, y_sim = sim_data_lin(dim, eps, n, rng, theta_star)
        for l in lambda_ls:
            if_emp_ls, if_pop_ls = if_diff_sim_lin(
                dim, x_sim, y_sim, oracle_theta_dict, l, n, n_sim, seed, theta_star, if_orth_emp, if_orth_pop)
            bound_val = []
            for i in range(len(if_emp_ls)):
                diff_abs_total = np.abs(if_emp_ls[i]-if_pop_ls[i])
                bound_val.append(np.dot(np.matmul(diff_abs_total, np.eye(
                    dim) * (1 / dim**2 + l)), np.transpose(diff_abs_total)))
            mean_diff_abs_total.append(np.mean(bound_val))
            sd_diff_abs_total.append(np.std(bound_val))
            n_samp.append(n)
            lambda_.append(l)
    return (mean_diff_abs_total, sd_diff_abs_total, n_samp, lambda_)
