# RSRGF
## How to run numerical experiments
1. Select optimization problem and an algorithm and set the parameters in `make_config.py`.
2. Make config.json file using command `python make_config.py`.
3. Run numerical experiments using command `python main.py path/to/config.json`.
4. Check the results in `results/problem_name/problem_parameters/constraints_name/constraints_parameters/algorithm_name/algorithm_parameters` directory.
5. You can compare results using `python result_show.py`. with GUI interface.

## Objective
### SOFTMAX
minimizing softmax loss function.</br>
data_name: "Scotus" or "news20"

### ROBUST LOGISTIC
solve min max problems with logistic function.</br>
data_name: "rcv1" or "news20" or "random",
inner-iteration: iteration to subproblem,
subproblem_eps: stop criteria of subproblem,
delta: max power of noise.

### REGULARIZED
set `problem_name = REGULARIZED + other_problem_name`.
minimizing regularized function 
$$\min_x f(x) + \lambda \|x\|_p^p$$
coeff: $\lambda$,
ord: $p$,
Fused: only False

## algorithm
### RGF(random gradient free method[https://link.springer.com/article/10.1007/s10208-015-9296-2])
mu: smoothing parameter,
sample_size: the number of random sampling per iterations,
lr: step size,
central: use central difference or not,
step_schedule: "constant" or "decrease",

### proposed
mu: smoothing parameter,
reduced_dim:the size of random matrix,
sample_size: the number of random sampling per iterations,
lr: step size,
central: use central difference or not,
step_schedule: "constant" or "decrease",
projection: solve subproblem with random projected subproblem or not.
