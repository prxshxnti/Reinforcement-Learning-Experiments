import torch 
from datasets import load_dataset
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import time
import os
from functools import partial 
import random
random.seed(42)

os.makedirs("./checkpoints/sft_final", exist_ok=True)
torch.manual_seed(42)
device = ("cuda" if torch.cuda.is_available() else "cpu") 

print(f"Pytorch version : {torch.__version__}")
print(f"Cuda Available ? {torch.cuda.is_available()}")
print(f"Device count : {torch.cuda.device_count()}")
print(f"Device Name : {torch.cuda.get_device_name()}")

# Loading Dataset
alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Len of Alpaca dataset : {len(alpaca_dataset)}")
 
alpaca_dataset = alpaca_dataset.train_test_split(test_size = 0.1, seed=42)
train_dataset , val_dataset = alpaca_dataset["train"], alpaca_dataset["test"]
print(f"Len of train dataset : {len(train_dataset)}")
print(f"Len of validation dataset : {len(val_dataset)}")

print(f"\nOne instance : {val_dataset[0]}\n")
print(f"\nOne instance keys : {val_dataset[0].keys()}\n")

# Removing 6 empty outputs samples found in Dataset auditing 
train_dataset = train_dataset.filter(
    lambda x: isinstance(x["output"], str) and bool(x["output"].strip())
)

def build_prompt_template(instruction, inp=""): 
    user = instruction if not inp else f"{instruction}\n\n{inp}"
    return f"### Instruction:\n{user}\n\n### Response:\n"

# Create Dataset and DataLoader
class AlpacaDataset(Dataset): 
    def __init__(self, dataset, tokenizer, max_length, samples_count): 
        self.dataset = dataset.select(range(samples_count))
        self.tokenizer = tokenizer 
        self.max_length = max_length

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        sample = self.dataset[idx]

        instruction = sample["instruction"]
        output = sample["output"]
        inp = sample["input"]

        prompt_text = build_prompt_template(instruction, inp)
        response_text = output
        
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids 
        attention_mask = [1] * len(input_ids) 
        
        # Create labels
        labels = [-100] * len(prompt_ids) + response_ids

        assert input_ids[-1] == self.tokenizer.eos_token_id  # Input ends with EOS. Ensures the final input token is the end-of-sequence token.
        assert labels[-1] == self.tokenizer.eos_token_id # EOS contributes to training loss. Ensures the model is trained to predict EOS, teaching it when to stop.
        assert all(x == -100 for x in labels[:len(prompt_ids)]) # Prompt loss is masked. Ensures prompt tokens have label `-100`, so the model isn't trained to predict the prompt itself.
        assert len(input_ids) == len(labels) #Input and labels are aligned. Ensures each input token has a corresponding label.
        assert all( labels[i] == input_ids[i] for i in range(len(prompt_ids), len(input_ids)) )
        # Together, these checks verify that the model sees the prompt, learns to generate the answer, and receives a training signal to stop at EOS.

        return {
            "input_ids" : torch.tensor(input_ids, dtype=torch.long), 
            "attention_mask" : torch.tensor(attention_mask, dtype=torch.long), 
            "labels" : torch.tensor(labels, dtype=torch.long), 
        }


# Load the Tokenizer, model from transformers library 
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

train_samples = 10000
test_samples = 500
train_dataset = AlpacaDataset(train_dataset, tokenizer, max_length = 512, samples_count = train_samples)
val_dataset = AlpacaDataset(val_dataset, tokenizer, max_length=512, samples_count = test_samples)


# for j in random.sample(range(len(train_dataset.dataset)), 10):
#     sample = train_dataset.dataset[j]

#     print(f"\n{'='*60}")
#     print(f"Sample {j}")
#     print(f"Instruction: {sample['instruction']}")
#     print(f"Input: {sample['input']}")
#     print(f"Output: {sample['output']}")

#     formatted_text = build_prompt_template(
#         sample["instruction"], sample["input"]
#     ) + sample["output"] + tokenizer.eos_token

#     print(f"Formatted text:\n{formatted_text}")


# print(f"Training samples count : {train_samples}")
# print(f"Test samples count : {test_samples}")

def collate_fn(samples, tokenizer): 
    max_len = 0
    for sample in samples: 
        max_len = max(max_len, len(sample["input_ids"]))

    
    if max_len > 512: 
        max_len = 512 #Truncation
        
    batch = {
        "input_ids" : [],
        "attention_mask" : [],
        "labels" : [],
    }

    for sample in samples: 

        input_ids = sample["input_ids"][:max_len]
        attention_mask = sample["attention_mask"][:max_len]
        labels = sample["labels"][:max_len]

        padding_length = max_len - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros((padding_length, ), dtype=torch.long)])
        labels = torch.cat([labels, torch.full((padding_length, ), -100, dtype=torch.long)])

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    for key in batch:
        batch[key] = torch.stack(batch[key])
    
    return batch 

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn = partial(collate_fn, tokenizer=tokenizer))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn = partial(collate_fn, tokenizer=tokenizer))

print(f"One instance from dataset : {train_dataset[0]['input_ids'].shape}")
print(f"One batch from dataloader : {next(iter(train_loader))['input_ids'].shape}")

# Creating a optimizer 
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 10

# Move the model and data to device 
model = model.to(device)

def generate_response(model, tokenizer, instruction, device):
    model.eval()

    prompt = build_prompt_template(instruction)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response

instruction = "Explain how photosynthesis works."

before_response = generate_response(
    model, tokenizer, instruction, device
)

print("BEFORE FINE-TUNING:")
print(before_response)

best_loss = 1e7
checkpoint_path = "./checkpoints/sft_final"
step_count = 0

# Early stopping rules
patience_counter = 0
patience_limit = 3
min_delta = 0.001

for epoch in range(1, num_epochs + 1): 
    total_train_loss = 0.0
    start_time = time.time()
    for batch in train_loader: 
        model.train()
        batch = {k : v.to(device) for k, v in batch.items()}
        logits = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], labels=batch["labels"])
        
        loss = logits.loss
        total_train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        # Before updating the parameters - clip the gradients 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        step_count += 1

        if step_count % 200 == 0: 
            print(f"Evaluating at {step_count}th step")

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader: 
                    val_batch = {k : v.to(device) for k, v in batch.items()}
                    logits = model(input_ids=val_batch["input_ids"],
                                    attention_mask=val_batch["attention_mask"], labels = val_batch["labels"])
            
                    loss = logits.loss.item()
                    total_val_loss += loss
            
                avg_val_loss = total_val_loss/ len(val_loader) 
                perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

            print(f"Evaluation results :: Validation loss : {avg_val_loss:.4f}") # If the loss consistently improves by tiny amounts, such as 0.0002, and those improvements are meaningful for task, better to reduce threshold to 0.0001.

            if avg_val_loss < best_loss and (best_loss - avg_val_loss) > min_delta: 
                print(f"Best loss found at {step_count}th step") 

                # Save checkpoint    
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"SFT checkpoint saved to {checkpoint_path}")

                best_loss = avg_val_loss
                patience_counter = 0 

            else : 
                patience_counter += 1
                if patience_counter >= patience_limit: 
                    print(f"Exiting inner loop due to exceed in patience limits")
                    break 

    if patience_counter >= patience_limit: 
        print(f"Exiting outer loop due to exceed in patience limits")
        break 
        
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch : {epoch} / {num_epochs} :: Time taken : {time.time() - start_time} :: Train Loss : {avg_train_loss:.4f}")


best_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
best_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

best_model = best_model.to(device)

after_response = generate_response(
    best_model, best_tokenizer, instruction, device
)

print("\nAFTER FINE-TUNING:")
print(after_response)
