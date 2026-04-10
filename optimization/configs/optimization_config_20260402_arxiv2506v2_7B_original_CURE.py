# =============================================================================
# CURE 论文复现配置 — arXiv:2506.03136v2（v2）Section 4.1.2 + Algorithm 1
#
# 奖励：论文式「原」CURE — 代码奖励 Eq.(3)、单测奖励 Eq.(4)，基于 GT 单测构造
#       execution 矩阵；非 long-CoT 模型关闭 Section 3.4 的长度引导变换。
#       对应本仓库 reward.py 中 use_self_bootstrap=False（非 self-bootstrap 路径）。
#
# 论文写明：Qwen2.5-7B/14B Instruct；每步 N=M=16；temperature=1、top-p=1（sample.py 内 top_p 固定 1.0）；
#          优化 LR=1e-6，KL 系数 β=0.01；7B/14B 训练 350 steps；8×A100。
# 数据：CodeContests 难度≤2，约 4.5k 训练 + 200 评测（见 §4.1.1）。
#
# 硬件：以下为当前机 2×GPU 的折中；若你有 8 卡，请将 gpu_groups 改为 8 路并行、
#       actor_num_gpus_per_node=8，并将 n_sample_per_step 保持为「全训练集」以更接近论文每步覆盖。
# =============================================================================

exp_name = "20260402_arxiv2506v2_7B_CURE_original_reward_CodeContests_s350"

pretrained_model = "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct"

train_dataset = "CodeContests_train"
eval_dataset = "CodeContests"

# §4.1.2: 7B / 14B — 350 steps
total_steps = 350
eval_interval = 35
save_interval = 35

# ============= sampling（§4.1.2）=============
k_code = 16
k_case = 16
temp = 1.0

# 每步随机子集上限；设为 ≥ 训练集大小则每步覆盖全部约 4.5k 题（与论文规模一致，算力需求高）
n_sample_per_step = 5000

# 2×GPU：单引擎 TP=2。8×A100 时可改为例如 [[0],[1],[2],[3],[4],[5],[6],[7]]
gpu_groups = [[0, 1]]

max_ground_truth_test = 8
max_input_examples = 1
max_model_len = 20000
max_generation_token = 10000
p_give_example = 1.0

system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. \
<|redacted_im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
This is the problem:\n{{problem}} <|redacted_im_end|>\n<|im_start|>Assistant: """

system_case_prompts = """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
<|redacted_im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:\n{{problem}}\n
{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|redacted_im_end|>\n<|im_start|>Assistant: """

special_requirements = """You should use input() to input and print() to output in your script. """

# ============= execution =============
# 大规模 n_sample 时可按需增大 num_chunks
num_chunks = 64
scale_tuple_list = [(4, 4), (16, 16)]

# ============= reward：原 CURE（非 §3.4 long-CoT 变换）=============
separate_training = True
# Qwen2.5-7B-Instruct 非论文中的 long-CoT 设置 — 关闭长度引导 reward 变换
enable_efficient = False
max_len_threshold = 8000
min_len_threshold = 1000
post_stage = False

# ============= training（§4.1.2）=============
total_num_nodes = 1
actor_num_gpus_per_node = 2
actor_learning_rate = 1e-6
num_warmup_steps = 0
policy_update_steps = 1
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True
prompt_max_len = 2000
generate_max_len = 8000
packing_max_len = 20000
max_epochs = 1
optimized_model_name = "optimized"

# ============= eval（Table 1: BoN 16×16）=============
eval_k_code = 16
eval_k_case = 16
eval_scale_tuple_list = [(4, 4), (16, 16)]
eval_num_chunks = 32
eval_no_example = True
eval_max_test = 8

# ============= self-bootstrap（关闭 = 论文 Eq.3–4 路线）=============
use_self_bootstrap = False
em_iterations = 3
use_anchor = True
anchor_strength = 0.3
min_weight_threshold = 0.1
sb_use_length_reg = False
reward_diagnostics_include_rollouts = False
reward_diagnostics_max_rollout_problems = 8
reward_diagnostics_truncate_chars = None
