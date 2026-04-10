# ========================================= config for optimization process ==========================================
# ====================================================================================================================

# experiment name — auto-generated as "{YYYYMMDD}_{HHMM}_{model_short}_{dataset}" if left empty
exp_name = "0409_Qwen2.5-7B-Instruct-2gpu_350step_pair"


# the model you want to optimize
pretrained_model = "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct"

# the training data and evaluation data
train_dataset = "CodeContests_train"
eval_dataset = "LiveBench"

# 20 steps, ~2h estimated with 8 GPUs (4x sampling parallelism)
total_steps = 350

# evaluate at step 5, 10, 15, 20 to see the trend
eval_interval = 25

# save model checkpoint every 5 steps for post-mortem analysis
save_interval = 100




# ============= config for sampling in each step =================

# Paper: "we generate 16 rollouts for unit tests and 16 for code"
k_code = 16
k_case = 16

# Paper: "temperature of 1.0, top-p of 1.0"
# For long-CoT model (Qwen3-4B), paper mentions lower temp 0.8 — use 1.0 first for standard reproduction
temp = 1.0

# number of tasks sampled per step
# 60 problems/step from CodeContests_train (4529 total), randomly sampled
n_sample_per_step = 100

# GPU usage for vllm inference
# 2 GPUs: 1 engine × 2-way tensor parallel
gpu_groups = [[0,1]]

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

# the prompt design for code generation and unit test generation — same as paper Appendix C.1
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

# should be proportional to k_code * k_case * n_sample_per_step
# 16 * 16 * 20 = 5120, use 32 chunks for parallel execution
num_chunks = 32*8

# Paper Table 1 / Figure 4: BoN uses (N=16, M=16)
# Also report intermediate scales for Figure 4 curves
scale_tuple_list = [(4, 4), (16, 16)]



# ============= config for reward assignment in each step =================

# Reward backend: "gt_based" | "self_bootstrap" | "pairwise_disagreement"
# When "pairwise_disagreement", use_self_bootstrap is ignored for reward assignment.
reward_mode = "pairwise_disagreement"

# Pairwise disagreement (generated tests only): subtract from case raw reward when column is all-identical outputs
pairwise_disagreement_zero_penalty = 0.0

# Apply length_regularize in pairwise_disagreement mode when enable_efficient is True
pairwise_disagreement_use_length_reg = False

# set True by default
separate_training = True

# Paper Section 3.4: long-CoT model uses response-length-guided transformation
# Qwen3-4B IS a long-CoT model, so enable_efficient = True
enable_efficient = True
# Paper: "truncate responses longer than 8K tokens"
max_len_threshold = 8000
min_len_threshold = 1000

# False by default, but suggest True after 100+ optimization steps
post_stage = False



# ============= config for training in each step =================

# number of GPUs for training
# 2 GPUs with DeepSpeed ZeRO-3: more sharding, faster training
total_num_nodes = 1
actor_num_gpus_per_node = 2
# Paper: "learning rate to 1 × 10^-6"
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# Paper: "KL coefficient β to 0.01"
use_kl_loss = False
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
eval_num_chunks = 32*8
eval_no_example = True
eval_max_test = 8
# ============= config for self-bootstrapped reward =================

# ✓ NEW MODE: Use execution consistency self-bootstrapped reward
use_self_bootstrap = False

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

# Whether to apply length_regularize in self-bootstrap mode
# Disabled by default because EM estimates may not be accurate enough
# for sign-based binarization to work correctly
sb_use_length_reg = False