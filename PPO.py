import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(sample):
    instruction = sample["instruction"]
    extra_input = sample.get("input", "")

    user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    return f"### Instruction:\n{user}\n\n### Response:\n"


class RewardModel(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.backbone = sft_model.model
        hidden_size = sft_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, dtype=sft_model.dtype)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        positions = torch.arange(
            hidden_states.size(1), device=hidden_states.device
        ).unsqueeze(0).expand_as(attention_mask)

        last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        last_indices = last_indices.clamp(min=0)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, last_indices]


        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)


def token_logprobs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :] # Remove last token since it doesn't predict anything
    target_tokens = input_ids[:, 1:] # Get the target tokens 
    log_probs = torch.log_softmax(logits, dim=-1) # Convert logits to probs
    gathered = torch.gather(log_probs, dim=2, index=target_tokens.unsqueeze(-1)).squeeze(-1) # For each target token index, get that prob from vocabulary
    return gathered


def build_response_mask(prompt_lens, seq_lens, target_len, device):
    positions = torch.arange(target_len, device=device).unsqueeze(0)
    start = (prompt_lens - 1).clamp(min=0).unsqueeze(1)
    end = (seq_lens - 1).clamp(min=0).unsqueeze(1)
    return (positions >= start) & (positions < end)


def masked_mean(values, mask, eps=1e-8):
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp(min=eps)
    return (values * mask_f).sum(dim=1) / denom

@torch.no_grad()
def generate_with_lengths(policy, input_ids, attention_mask,
                          max_new_tokens, eos_token_id):
    generated = input_ids
    finished = torch.zeros(
        input_ids.size(0), dtype=torch.bool, device=input_ids.device
    )
    response_lengths = torch.zeros(
        input_ids.size(0), dtype=torch.long, device=input_ids.device
    )

    for _ in range(max_new_tokens):
        outputs = policy(
            input_ids=generated,
            attention_mask=attention_mask,
        )
        next_token = torch.multinomial(
            torch.softmax(outputs.logits[:, -1, :].float(), dim=-1), 1
        ).squeeze(1)

        # Already-finished rows receive padding; don't count it as response.
        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_token_id),
            next_token,
        )
        generated = torch.cat([generated, next_token[:, None]], dim=1)

        active = ~finished
        response_lengths += active.long()
        finished |= (next_token == eos_token_id)

        attention_mask = torch.cat([
            attention_mask, active.long().unsqueeze(1)
        ], dim=1)

        if finished.all():
            break

    return generated, attention_mask, response_lengths
    


def collect_rollout(policy, ref_model, reward_model, tokenizer, prompts, max_prompt_len, max_new_tokens, beta, device):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated, full_attention_mask, response_lengths = generate_with_lengths(
            policy, encoded["input_ids"], encoded["attention_mask"],
            max_new_tokens, tokenizer.eos_token_id
        )

    prompt_width = encoded["input_ids"].shape[1]
    positions = torch.arange(
        generated.shape[1] - 1, device=device
    ).unsqueeze(0)
    
    response_mask = (
        (positions >= prompt_width - 1)
        & (positions < prompt_width + response_lengths.unsqueeze(1) - 1)
    )
        
    prompt_lens = encoded["attention_mask"].sum(dim=1) # Number of real tokens in each prompt.
    seq_lens = full_attention_mask.sum(dim=1) # Number of real tokens in the full sequence = prompt + generated response.

    with torch.no_grad():
        old_logprobs_tokens = token_logprobs(policy, generated, full_attention_mask)
        ref_logprobs_tokens = token_logprobs(ref_model, generated, full_attention_mask)

        # Calculate token level divergence
        d_t = old_logprobs_tokens - ref_logprobs_tokens
        k_t = torch.exp(-d_t) - 1 + d_t # This is sampled k3 KL estimator (non-negative mathematically)

        # Calculate per sample divergence and apply to reward 
        kl_per_sample = masked_mean(k_t, response_mask)
        rm_reward = reward_model(generated, full_attention_mask)
        total_reward = rm_reward - beta * kl_per_sample

        advantages = (total_reward - total_reward.mean()) / (total_reward.std(unbiased=False) + 1e-8)

        old_logprob_mean = masked_mean(old_logprobs_tokens, response_mask)

    return {
        "input_ids": generated,
        "attention_mask": full_attention_mask,
        "response_mask": response_mask,
        "old_logprob_mean": old_logprob_mean,
        "rm_reward": rm_reward,
        "kl": kl_per_sample,
        "total_reward": total_reward,
        "advantages": advantages,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sft_checkpoint_path = "./checkpoints/sft_final"
    reward_checkpoint_path = "./checkpoints/reward_model.pt"
    ppo_checkpoint_path = "./checkpoints/ppo_final"

    if not os.path.exists(sft_checkpoint_path):
        raise FileNotFoundError("Missing SFT checkpoint at ./checkpoints/sft_final")
    if not os.path.exists(reward_checkpoint_path):
        raise FileNotFoundError("Missing reward model checkpoint at ./checkpoints/reward_model.pt")

    model_kwargs = {"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {}

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, **model_kwargs).to(device) # The model that gets updated
    reference_policy = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, **model_kwargs).to(device) # The model that is used as a reference

    # Frozen reference policy 
    for param in reference_policy.parameters():
        param.requires_grad = False
    reference_policy.eval()

    reward_backbone = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, **model_kwargs).to(device)
    reward_model = RewardModel(reward_backbone).to(device)
    reward_ckpt = torch.load(reward_checkpoint_path, map_location=device)
    reward_model.load_state_dict(reward_ckpt["model_state_dict"])
    reward_model.eval()

    # Freezing the reward model params
    for param in reward_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(policy.parameters(), lr=1e-6, weight_decay=0.01)

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    max_prompts = 512
    prompts = [format_prompt(dataset[i]) for i in range(min(len(dataset), max_prompts))]
    prompt_loader = DataLoader(prompts, batch_size=8, shuffle=True)

    ppo_steps = 50
    ppo_epochs = 3
    mini_batch_size = 4
    clip_eps = 0.2
    beta = 0.05
    max_prompt_len = 256
    max_new_tokens = 96

    policy.train()

    step = 0
    while step < ppo_steps: # 50 steps
        for prompt_batch in prompt_loader: # Each batch has 8 samples
            # 1. Collect rollout
            rollout = collect_rollout(
                policy=policy,
                ref_model=reference_policy,
                reward_model=reward_model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                max_prompt_len=max_prompt_len,
                max_new_tokens=max_new_tokens,
                beta=beta,
                device=device,
            )

            batch_size = rollout["input_ids"].size(0)

            # 2. PPO updates on this rollout
            for _ in range(ppo_epochs): # Each batch goes through 3 epochs
                permutation = torch.randperm(batch_size, device=device) 

                for start in range(0, batch_size, mini_batch_size): # 0, 8, 4 -> gives 2 mini batches of size 4 
                    idx = permutation[start : start + mini_batch_size]

                    mb_input_ids = rollout["input_ids"][idx]
                    mb_attention_mask = rollout["attention_mask"][idx]
                    mb_response_mask = rollout["response_mask"][idx]
                    mb_old_logprob = rollout["old_logprob_mean"][idx]
                    mb_advantages = rollout["advantages"][idx]

                    new_logprobs_tokens = token_logprobs(policy, mb_input_ids, mb_attention_mask)
                    new_logprob_mean = masked_mean(new_logprobs_tokens, mb_response_mask)

                    ratio = torch.exp(new_logprob_mean - mb_old_logprob)
                    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                    surrogate_1 = ratio * mb_advantages
                    surrogate_2 = clipped_ratio * mb_advantages
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                    optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
        
                    # ONE complete PPO step is finished
                    step += 1
                    print(f"Step {step}/{ppo_steps} | RM Reward: {rollout['rm_reward'].mean().item():.4f} | KL: {rollout['kl'].mean().item():.4f} | R_total: {rollout['total_reward'].mean().item():.4f}")
                    if step >= ppo_steps : 
                        break

                if step >= ppo_steps : 
                    break

            if step >= ppo_steps : 
                break

    os.makedirs(ppo_checkpoint_path, exist_ok=True)
    policy.save_pretrained(ppo_checkpoint_path)
    tokenizer.save_pretrained(ppo_checkpoint_path)
    print(f"Saved PPO checkpoint to {ppo_checkpoint_path}")


if __name__ == "__main__":
    main()


