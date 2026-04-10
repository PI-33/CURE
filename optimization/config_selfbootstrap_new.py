# ========================================= config for optimization process ==========================================
# ====================================================================================================================
#
# 此文件为 Self-bootstrap 新方案配置
# 用于与 GT-based Baseline 进行对比实验
# 主要改动：
#   - n_sample_per_step = 10 (小规模测试)
#   - total_steps = 4 (快速验证)
#   - use_self_bootstrap = True (使用执行一致性自举奖励)
#   - em_iterations = 3 (EM算法迭代次数)
#   - use_anchor = True (使用公开例子作为弱监督)
#

# experiment name
exp_name = "exp_selfbootstrap_new"


# the model you want to optimize
pretrained_model = "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct"

# the training data and evaluation data
train_dataset = "CodeContests_train"
eval_dataset = "CodeContests"

# For small-scale experiment: only 4 steps
total_steps = 4

# evaluate every eval_interval steps
eval_interval = 2

# save optimized model every save_interval steps
save_interval = 2




# ============= config for sampling in each step =================

# Paper: "we generate 16 rollouts for unit tests and 16 for code"
k_code = 16
k_case = 16

# Paper: "temperature of 1.0, top-p of 1.0"
temp = 1.0

# number of tasks sampled per step — SMALL SCALE VERSION
# Original: 20 tasks × 16 codes × 16 tests = 5120 rollouts/step
# Small-scale: 10 tasks × 16 codes × 16 tests = 2560 rollouts/step
n_sample_per_step = 10

# GPU usage for vllm inference
gpu_groups = [[0,1],[2,3],[4,5],[6,7]]

# max ground-truth unit test we can use here
max_ground_truth_test = 8

# set to 1 by default
max_input_examples = 1

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 20000

# max token model can generate for each query
max_generation_token = 10000

# the probability for providing public unit test example in prompt
p_give_example = 1.0

# the prompt design for code generation and unit test generation
system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """

system_case_prompts = """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
<|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:\n{{problem}}\n
{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """

# some special requirements for code generation
special_requirements = """You should use input() to input and print() to output in your script. """



# ============= config for execution in each step =================

# For small-scale: 32 chunks for 2560 rollouts
num_chunks = 32

# Paper Table 1 / Figure 4: BoN uses (N=16, M=16)
scale_tuple_list = [(4, 4), (16, 16)]



# ============= config for reward assignment in each step =================

# set True by default
separate_training = True

# Long-CoT model: enable length regularization
enable_efficient = True
max_len_threshold = 8000
min_len_threshold = 1000

# False by default
post_stage = False



# ============= config for self-bootstrapped reward =================

# ✓ NEW MODE: Use execution consistency self-bootstrapped reward
use_self_bootstrap = True

# Number of EM iterations for jointly estimating code quality and test reliability
# Lower values (2-3) for quick verification, higher (5+) for better convergence
em_iterations = 3

# Use public example_input/example_output as weak anchor for code quality initialization
# This helps stabilize the self-bootstrapping process without using full GT
use_anchor = True

# Blending strength for anchor: 0 = no anchor influence, 1 = anchor only
# Recommended: 0.2-0.4 for balance between public examples and generated tests
anchor_strength = 0.3

# Skip problems where the best test weight is below this threshold
# Lower threshold = use more problems (but with weaker signal)
# Higher threshold = use fewer problems (but with stronger signal)
# Recommended: 0.05-0.15
min_weight_threshold = 0.1



# ============= config for training in each step =================

# number of GPUs for training
total_num_nodes = 2

# Paper: "learning rate to 1 × 10^-6"
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# Paper: "KL coefficient β to 0.01"
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True

# max prompt (inquiry) length in collected data
prompt_max_len = 2000

# generation token limit
generate_max_len = 8000

# packing_max_len >= generate_max_len + prompt_max_len
packing_max_len = 20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "optimized"



# ============= config for evaluation during the optimization =================

# Eval uses N=M=16 to match paper Table 1
eval_k_code = 16
eval_k_case = 16
eval_scale_tuple_list = [(4, 4), (16, 16)]
eval_num_chunks = 32
eval_no_example = True
eval_max_test = 8
