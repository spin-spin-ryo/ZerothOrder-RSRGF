import os
log_path = '.\\results\\regularized LinearRegression\\ord;1_coef;1e-06_fused;False_data-name;random-100-10000_bias;True\\proposed\\lr;0.001_reduced_dim;10_sample_size;1_mu;1e-06_step_schedule;constant\\aaaaaaaaaaaaaaaaaaa'
log_pre = ".\\results\\regularized LinearRegression\\ord;1_coef;1e-06_fused;False_data-name;random-100-10000_bias;True\\proposed\\lr;0.001_reduced_dim;10_sample_size;1_mu;1e-06_step_schedule;constant"

file_name = "test.txt"
test_path = os.path.join(log_pre,file_name)
print(os.path.exists(log_pre))
with open(log_path,"w") as f:
    f.write("ok")