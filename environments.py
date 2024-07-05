import torch

DATAPATH = "./data"
DTYPE = torch.float64

if torch.cuda.is_available():   
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

CONFIGPATH = "./configs"
RESULTPATH = "./results"


# algorithm name
RGF = "RGF"
RSRGF = "proposed"

ALGORITHM_PARAMS_KEY = {
    RGF:["mu","sample_size","lr","central","step_schedule"],
    RSRGF:["reduced_dim","mu","sample_size","lr","projection","central","step_schedule"]
}

# objective name
TEST = "test"
QUADRATIC = "Quadratic"
MAXLINEAR = "max-linear"
PIECEWISELINEAR = "piecewise-linear"
SUBSPACENORM = "subspace-norm"
SUBSPACENORM_LOCAL = "subspace-norm local"
LINEARREGRESSION = "LinearRegression"
NONNEGATIVEMATRIXFACTRIZATION = "NMF"
REGULARIZED = "regularized"
ADVERSERIALATTACK = "adverserial attack"
ROBUSTADVERSARIAL = "robust adversarial"
LOGISTIC = "logistic"
SOFTMAX = "softmax"
ROBUSTLOGISTIC = "robust logistic"

OBJECTIVE_PARAMS_KEY = {
    SOFTMAX:["data-name"],
    ROBUSTLOGISTIC:["data-name","inner-iteration","subproblem-eps","delta"],
    REGULARIZED:["ord","coef","fused"]
}