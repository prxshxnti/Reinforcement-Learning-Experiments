from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch
import os
os.makedirs("./checkpoints", exist_ok=True)

torch.cuda.reset_peak_memory_stats()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

split_1 = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_1["train"]
temp_dataset = split_1["test"] 

split_2 = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = split_2["train"]
test_dataset = split_2["test"]

print(f"Size of Train dataset : {len(train_dataset)}")
print(f"Size of Validation dataset : {len(val_dataset)}")
print(f"Size of Test dataset : {len(test_dataset)}")

class RewardDataset(Dataset): 
    def __init__(self, tokenizer, dataset, max_length, samples_count ): 
        self.dataset = dataset.select(range(samples_count))
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        sample = self.dataset[idx] 

        chosen = sample["chosen"]
        chosen_ids = self.tokenizer(chosen, truncation=True, max_length = self.max_length, padding = "max_length", return_tensors="pt")
        
        rejected = sample["rejected"]
        rejected_ids = self.tokenizer(rejected, truncation=True, max_length = self.max_length, padding = "max_length", return_tensors="pt")

        return {
            "chosen_input_ids" : chosen_ids["input_ids"].squeeze(0),
            "chosen_attention_mask" : chosen_ids["attention_mask"].squeeze(0),

            "rejected_input_ids" : rejected_ids["input_ids"].squeeze(0),
            "rejected_attention_mask" : rejected_ids["attention_mask"].squeeze(0)
        }

# Model loading 
model = AutoModelForCausalLM.from_pretrained("./checkpoints/sft_final")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/sft_final")

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 500
TESTING_SAMPLES = 500

# Creating dataset and dataloader 
print(f"Samples used for training : {TRAINING_SAMPLES}")
print(f"Samples used for validation : {VALIDATION_SAMPLES}")
print(f"Samples used for testing : {TESTING_SAMPLES}")
train_dataset = RewardDataset(tokenizer, train_dataset, max_length = 512, samples_count=TRAINING_SAMPLES)
val_dataset = RewardDataset(tokenizer, val_dataset, max_length = 512, samples_count=VALIDATION_SAMPLES)
test_dataset = RewardDataset(tokenizer, test_dataset, max_length = 512, samples_count=TESTING_SAMPLES)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Create a reward model, optimizer, loss fn 
class RewardModel(nn.Module): 
    def __init__(self, sft_model): 
        super().__init__()
        self.backbone = sft_model.model 
        hidden_size = sft_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, dtype=sft_model.dtype)

    def forward(self, input_ids, attention_mask): 
        outputs = self.backbone(input_ids, attention_mask) 
        hidden_states = outputs.last_hidden_state # [B, S, H)
        
        lengths = attention_mask.sum(dim=1) - 1 
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device) 

        last_hidden = hidden_states[batch_indices, lengths]
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)
        
reward_model = RewardModel(model).to(device)
optimizer = AdamW(reward_model.parameters(), lr=2e-5, weight_decay=0.1) 

def loss_fn(chosen_reward, rejected_reward):
    return -F.logsigmoid(chosen_reward - rejected_reward).mean() # Standard preference loss

# Training loop 
num_epochs = 5
best_val_acc = 0.0

for epoch in range(num_epochs):
    reward_model.train()
    total_loss = 0.0
    for batch in train_loader:

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)

        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        # Forward pass
        chosen_reward = reward_model( chosen_input_ids, chosen_attention_mask)
        rejected_reward = reward_model( rejected_input_ids, rejected_attention_mask)

        # Bradley-Terry loss
        loss = loss_fn( chosen_reward, rejected_reward )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Validation loop - Evaluation and test accuracy
    reward_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
    
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    
            # Get rewards
            chosen_reward = reward_model( chosen_input_ids,  chosen_attention_mask )
            rejected_reward = reward_model( rejected_input_ids, rejected_attention_mask )
    
            # Did the model rank chosen higher?
            correct += ( chosen_reward > rejected_reward ).sum().item()
            total += chosen_reward.size(0)
    
    accuracy = correct / total
    print(f"Validation Preference Accuracy: {accuracy:.4f}")
    if accuracy > best_val_acc: 
        best_val_acc = accuracy
        print(f"Saving checkpoint : Best validation accuracy found !")
        torch.save(
            {
                "model_state_dict": reward_model.state_dict(),
                "max_length": 512,
            },
            "./checkpoints/reward_model.pt",
        )
        print("Saved reward model checkpoint to ./checkpoints/reward_model.pt")



# Evaluate on the best checkpoint with Test Dataset 

best_model = torch.load("./checkpoints/reward_model.pt", map_location=device)["model_state_dict"]
reward_model.load_state_dict(best_model)
reward_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)

        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        # Get rewards
        chosen_reward = reward_model( chosen_input_ids,  chosen_attention_mask )
        rejected_reward = reward_model( rejected_input_ids, rejected_attention_mask )

        # Did the model rank chosen higher?
        correct += ( chosen_reward > rejected_reward ).sum().item()
        total += chosen_reward.size(0)

accuracy = correct / total
print(f"Test Preference Accuracy: {accuracy:.4f}")


peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Peak VRAM allocated: {peak_vram_gb:.2f} GB")
