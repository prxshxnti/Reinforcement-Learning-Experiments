
## For 1000 samples
### DPO.log - First Iteration baseline 

Epoch : 1/3, Loss : 0.76
Epoch : 2/3, Loss : 0.17
Epoch : 3/3, Loss : 0.13
Preference Accuracy : 0.51

### After Bug Fixes [ Calculated seq probs for differed response Tokens + Truncation error]

Epoch : 1/3, Loss : 0.75
Epoch : 2/3, Loss : 0.16
Epoch : 3/3, Loss : 0.11
Preference Accuracy : 0.60
Refernce Accuracy : 0.55
DPO Accuracy : 0.55

### Bug - Reference accuracy is varying when the model is frozen
`Count of samples : 1000`

Epoch : 1/3, Loss : 0.74
Epoch : 2/3, Loss : 0.16
Epoch : 3/3, Loss : 0.10
Preference Accuracy : 0.59
Refernce Accuracy : 0.55
DPO Accuracy : 0.56

`Count of samples : 3000`

Epoch : 1/3, Loss : 0.76
Epoch : 2/3, Loss : 0.16
Epoch : 3/3, Loss : 0.12
Preference Accuracy : 0.57
Refernce Accuracy : 0.53
DPO Accuracy : 0.53

`Count of samples : 5000`

Epoch : 1/3, Loss : 0.75
Epoch : 2/3, Loss : 0.18
Epoch : 3/3, Loss : 0.13
Preference Accuracy : 0.50
Refernce Accuracy : 0.49
DPO Accuracy : 0.49


# Bug fixed 

`Count of samples : 1000`

Preference Accuracy : 0.59
Reference Accuracy : 0.55
DPO Accuracy : 0.55

`Count of samples : 3000`

Preference Accuracy : 0.61
Reference Accuracy : 0.55
DPO Accuracy : 0.56

`Count of samples : 5000`

Preference Accuracy : 0.63
Reference Accuracy : 0.55
DPO Accuracy : 0.56

- Preference accuracy improves from 59% to 63% as training data increases.
- DPO accuracy plateaus at 56% for 3,000 and 5,000 pairs.
- Training cost scales with dataset size.
- Reference accuracy remains stable, as expected.
Preference accuracy measures whether the policy's relative improvement over the reference favors chosen over rejected responses. 
DPO accuracy measures whether the policy's chosen-response log-probability exceeds its rejected-response log-probability. 

## Fix in SFT Training (v2)
1. Dynamic Padding - To improve speed 
2. Check data quality - Percentage of samples with more than 512 tokens : 0.03
Average length of input ids : 78.6098
More than 512 ids lengths : [1088, 529, 558]
Average response tokens retained pct : 99.99342110201268
Zero response is present in 512 length : 0
3. Storing the best checkpoint w.r.t to evaluation instead of the last epoch checkpoint 
4. Early stopping with patience upto 3 evals
5. Running the Pipeline on 5K training samples and fixed 500 samples
6. Gradient Clippping 

## Fix in SFT Training (v2_1)
1. Prompt template: Fixed the format mismatch between training and inference, reducing unwanted `### Explanation ###` and `### Output ###` headers in generated responses.
2. EOS token: Explicitly added EOS to each target and included it in the loss, teaching the model where responses should end.

Evidence from your evaluation: The repeated headers disappeared across the 50 responses(v2 vs v2.1), and several answers now terminate cleanly.

## SFT versions
V1/V2: Trained on instruction + output.
V2.1: Trained on instruction + optional input + output, wrapped in the prompt template.

## Final Verdict

Based on your 50-prompt evaluation, SFT model is qualitatively better than the base model, especially in response quality and repetition. This is a qualitative comparison, not yet a formally scored benchmark.

## Base model vs SFT

| Criterion             | Base model                                   | SFT model                                                |
| --------------------- | -------------------------------------------- | -------------------------------------------------------- |
| Factual answers       | Often repetitive or poorly terminated        | More direct and concise                                  |
| Relevance             | Sometimes drifts or repeats                  | More focused on the question                             |
| Explanation quality   | Repetitive, with limited structure           | More coherent explanations                               |
| Reasoning             | Several incorrect answers                    | Still makes mistakes; no clear overall win established   |
| Instruction following | Often repeats content and misses constraints | Some improvement, but exact constraints are still missed |
| Overall               | Less reliable response behavior              | Better assistant-style responses                         |

## Where SFT improved

- Repetition: Answers such as API definitions and HTTP explanations became more concise instead of repeating until the generation limit.
- Response quality: Several factual answers became clear, relevant, and immediately usable.
- Explanation structure: Some answers explain concepts more coherently instead of looping through the same statement.
- Instruction-following behavior: Some outputs improved, although strict requirements—such as exactly two sentences or a fixed word count—remain a weakness.

## What SFT has not solved

The model still makes reasoning errors, including the number-sequence question, logical deduction, and arithmetic problems. It also occasionally repeats itself or ignores formatting constraints.

Important: These results support an improvement in response behavior, but we have not established a numerical improvement in overall accuracy.

## Goal :
Does DPO or GRPO improve on this fixed SFT baseline under the same evaluation prompts and scoring rubric?


## Type 1 : Reward Model + PPO setup 
Training a reward model using the Bradley–Terry preference loss on the Anthropic HH-RLHF dataset. Score the chosen and rejected responses, then train the model to assign a higher reward to the chosen response.

BaseLine RewardModel Stats: 

Samples used for testing : 500
Epoch 1/5, Loss: 0.7645
Test Preference Accuracy: 0.5080
Epoch 2/5, Loss: 0.5274
Test Preference Accuracy: 0.5340
Epoch 3/5, Loss: 0.3000
Test Preference Accuracy: 0.5760
Epoch 4/5, Loss: 0.0550
Test Preference Accuracy: 0.5660
Epoch 5/5, Loss: 0.0339
Test Preference Accuracy: 0.5720

Reward Model (v2)

1. There is 3 datasets available. Train dataset to train the reward model, validation dataset to find the best checkpoint and the accuracy of the reward model on unseen data. Test dataset is to evaluate the best checkpoint across different versions of our Reward model to know whether increasing training samples or any other change improves the reward model on the whole. 
2. For every epoch, post training, we validate on the validation dataset to get validation accuracy. If its best that previous validation accuracy, we store the checkpoint.
3. After all epochs, Test on the best saved checkpoint (load best checkpoint) and calculate the preference accuracy on test dataset. 

Result after v2 version : 


Samples used for training : 5000
Samples used for validation : 500
Samples used for testing : 500

Epoch 1/5, Loss: 0.7644
Validation Preference Accuracy: 0.5460
Saving checkpoint : Best validation accuracy found !
Saved reward model checkpoint to ./checkpoints/reward_model.pt
Epoch 2/5, Loss: 0.5333
Validation Preference Accuracy: 0.5680
Saving checkpoint : Best validation accuracy found !
Saved reward model checkpoint to ./checkpoints/reward_model.pt
Epoch 3/5, Loss: 0.2688
Validation Preference Accuracy: 0.5700
Saving checkpoint : Best validation accuracy found !
Saved reward model checkpoint to ./checkpoints/reward_model.pt
Epoch 4/5, Loss: 0.0518
Validation Preference Accuracy: 0.5520
Epoch 5/5, Loss: 0.0275
Validation Preference Accuracy: 0.5620
`Test Preference Accuracy: 0.5120`


results:

| Metric                   | Result |
| ------------------------ | ------ |
| Best validation accuracy | 57.0%  |
| Test accuracy            | 51.2%  |
| Training loss (epoch 5)  | 0.0275 |

The gap between validation and test accuracy is concerning, I ran another controlled experiment by Increasing training data from 5,000 to 10,000 pairs. Keeping the same 500 validation and 500 test examples.


### Running on 10k samples

Samples used for training : 10000
Samples used for validation : 500
Samples used for testing : 500

Epoch 1/5, Loss: 0.7109
Validation Preference Accuracy: 0.5940
Saving checkpoint : Best validation accuracy found !
Saved reward model checkpoint to ./checkpoints/reward_model.pt
Epoch 2/5, Loss: 0.5710
Validation Preference Accuracy: 0.6240
Saving checkpoint : Best validation accuracy found !
Saved reward model checkpoint to ./checkpoints/reward_model.pt
Epoch 3/5, Loss: 0.2138
Validation Preference Accuracy: 0.6120
Epoch 4/5, Loss: 0.0486
Validation Preference Accuracy: 0.5960
Epoch 5/5, Loss: 0.0449
Validation Preference Accuracy: 0.6080
`Test Preference Accuracy: 0.5640`
Peak VRAM allocated: 9.69 GB

results:

| Metric                   | Result |
| ------------------------ | ------ |
| Best validation accuracy | 62.4%  |
| Test accuracy            | 56.4%  |
| Training loss (epoch 5)  | 0.0449 |

verdict : 
Increasing Data improves the accuracy of the reward model, thus suggesting our pipeline learns to score preferences given a preference dataset. 

## Bug Fixes in PPO setup

1. Instead of collecting 50 rollout and 3 epochs and minibatches, this setup will consider one policy update as one PPO setup and Runs the pipeline for 50 PPO steps. 
2. KL penalty is calculated incorrectly. 
KL = log probs of old - log probs of ref
log probs of ref policy > log probs of old policy, then KL < 0 
R_total = R_rm - beta * KL
For this case , negative KL increases total reward but in truth KL is how different a probability distribution of a token in comparison of reference module and updated model

`Simplest fix` - torch.abs() applied to the KL value
kl_per_sample = torch.abs(old_logprob_mean - ref_logprob_mean)
Taking abs() prevents negative values, but it does not give a proper KL estimate. Comparing two average log-probabilities, loses token-level information.
For each generated response token y_t, d_t = old log probs - ref log probs where d_t is token by token 
A common non-negative sampled estimator is: k_t = e^-d_t - 1 + d_t
Then average \(k_t\) over response tokens to get the KL penalty per sample. [ This presevers token level divergence info ]

3. Distinguishing generated EOS tokens from padding when both use the same token ID.
As per qwen 0.5B the model uses same id for EOS and padding which puts difficult in loss calcuation to distinguish the EOS token from padding token

We can't reliably distinguish them using token IDs alone when pad_token_id == eos_token_id.

Instead, we use the attention mask and sequence boundaries:
- The original prompt's attention mask tells which prompt tokens are real.
- Generation gives the response tokens and stopping boundary.
- We build the full attention mask using those boundaries, rather than generated != pad_token_id.


Later Enhancements :
1. To use per token probability ratios instead of token level advantages
2. To use a value/critic model


# Post fixes baseline run logs and analysis

Step 1/50 | RM Reward: -8.2500 | KL: 0.0000 | R_total: -8.2637
Step 2/50 | RM Reward: -8.2500 | KL: 0.0000 | R_total: -8.2637
Step 3/50 | RM Reward: -8.2500 | KL: 0.0000 | R_total: -8.2637
Step 4/50 | RM Reward: -8.2500 | KL: 0.0000 | R_total: -8.2637
Step 5/50 | RM Reward: -8.2500 | KL: 0.0000 | R_total: -8.2637
.
.
.
Step 46/50 | RM Reward: -8.0000 | KL: 0.0006 | R_total: -8.0078
Step 47/50 | RM Reward: -8.0000 | KL: 0.0006 | R_total: -8.0078
Step 48/50 | RM Reward: -8.0000 | KL: 0.0006 | R_total: -8.0078
Step 49/50 | RM Reward: -5.8750 | KL: 0.0006 | R_total: -5.8711
Step 50/50 | RM Reward: -5.8750 | KL: 0.0006 | R_total: -5.8711

Observations:

| Metric             | Observation                                          |
| ------------------ | ---------------------------------------------------- |
| Reward-model score | Negative, ranging roughly from −2 to −10             |
| KL                 | Extremely small, around 0–0.001                      |
| Repeated metrics   | Each group of 6 updates has identical rewards and KL |

The repeated values are expected here: since we are reusing the same rollout for 3 epochs, with 2 mini-batches per epoch. So each rollout generates 6 updates.

Major drawbacks:
- Very small KL: The policy may barely have moved from the reference
- Negative reward