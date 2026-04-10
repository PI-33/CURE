# Analysis Patterns for CURE Logs

## Pattern 1: Diagnosing Reward Filtering Issues

**Symptom**: Training crashes with `IndexError: list index out of range`

**Analysis Steps**:
1. Check metrics section: Are `code_acc` and `case_acc` both 1.0 or both 0.0?
2. Look at timeline: Did execute phase complete?
3. Examine errors: Search for "std.item() == 0"

**Root Causes**:
- All generated code passes tests (reward std = 0, filtered out)
- All generated code fails tests (same filtering)
- Very small dataset (n_sample_per_step too small)

**Fix Indicators**:
- Increase training dataset size
- Increase k_code or k_case (sampling diversity)
- Check if tasks are too easy/hard

## Pattern 2: KL Divergence Explosion

**Symptom**: avg_kl suddenly jumps from 0.3 to 5.0+

**What it means**:
- Policy is deviating too much from reference model
- Learning rate likely too high
- Reward signal corrupted or misaligned

**Recovery**:
- Reduce actor_learning_rate (1e-6 → 1e-7)
- Check reward computation (reward.py)
- Verify RL data is not empty

## Pattern 3: Policy Entropy Collapse

**Symptom**: policy_entropy drops below 0.1

**Warning**: Model is converging to single output (mode collapse)

**Indicators**:
- All generated codes become nearly identical
- case_acc drops to near 0 (can't diversify test cases)
- BoN accuracy plateaus

**Prevention**:
- Monitor entropy at each step
- If <0.2, increase temperature or sampling diversity

## Pattern 4: Critic Loss Not Decreasing

**Symptom**: critic_loss stays high (>1.0) across steps

**Indicates**: Value function not learning

**Typical causes**:
- Rewards have wrong scale (not normalized)
- advantage_normalize = False (should be True)
- Too few training epochs

**Check**:
- reward.py output statistics
- normalize_reward() function behavior

## Pattern 5: Stable Training Progression

**Healthy indicators**:
- code_acc starts ~0.5-0.6, trends to 0.8-0.9
- avg_kl stays in [0.1, 1.0] range
- policy_entropy remains >0.5
- critic_loss decreases monotonically
- No errors between steps

**Timeline analysis**:
```
Step 0: code_acc=0.6, kl=0.25, entropy=0.8
Step 1: code_acc=0.72, kl=0.28, entropy=0.75  ✓ improving
Step 2: code_acc=0.8, kl=0.3, entropy=0.7     ✓ converging steadily
```

## Query Patterns for the AI

After cleaning, use these analysis patterns:

### Debug a specific step
```
Clean log file, then ask:
"Step 2 training failed with IndexError. What was the state of metrics 
in steps 0-1? What indicators suggest reward filtering?"
```

### Compare runs
```
"I've provided two cleaned logs. Compare their metric trends. Which run 
learned faster? Which one has better entropy?"
```

### Identify bottleneck
```
"Which phase takes longest: sampling, execution, or training? 
Show me the timeline and compute phase durations."
```

### Predict convergence
```
"Based on the first 3 steps' metrics, is this training likely to succeed? 
What warning signs do you see?"
```

## Automatic Detection Checklist

The cleaner automatically extracts these. When analyzing, verify:

- [ ] All step markers present (0, 1, 2, ... N)
- [ ] Each step has complete metrics (code_acc, case_acc, BoN)
- [ ] No errors between steps
- [ ] Timeline shows sampling → execution → training flow
- [ ] Metrics show improvement trend (not plateauing/degrading)
- [ ] No sudden jumps in KL or entropy
- [ ] Generation job completes for each step

## Common Issues and Extraction

| Issue | What to Extract | Red Flag |
|-------|-----------------|----------|
| Rewards all filtered | code_acc + case_acc values | Both 0.0 or 1.0 exactly |
| KL explosion | avg_kl trend | >2.0 or jumping 5x |
| Entropy collapse | policy_entropy trend | <0.1 or dropping <0.05/step |
| Mode collapse | case_acc trend | Stuck at 0.1-0.2 |
| Training instability | critic_loss trend | Not decreasing or NaN |
| Dataset mismatch | error messages | KeyError for 'test_input' |
| Empty rollout | timeline gaps | Sampling without execution |
