import torch 
from datasets import load_dataset
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import time

device = ("cuda" if torch.cuda.is_available() else "cpu") 

print(f"Pytorch version : {torch.__version__}")
print(f"Cuda Available ? {torch.cuda.is_available()}")
print(f"Device count : {torch.cuda.device_count()}")
print(f"Device Name : {torch.cuda.get_device_name()}")

# Loading Dataset
alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Len of Alpaca dataset : {len(alpaca_dataset)}")
 
alpaca_dataset = alpaca_dataset.train_test_split(test_size = 0.1, seed=42)
train_dataset , test_dataset = alpaca_dataset["train"], alpaca_dataset["test"]
print(f"Len of train dataset : {len(train_dataset)}")
print(f"Len of test dataset : {len(test_dataset)}")

print(f"\nOne instance : {test_dataset[0]}\n")
print(f"\nOne instance keys : {test_dataset[0].keys()}\n")


# Create Dataset and DataLoader
class AlpacaDataset(Dataset): 
    def __init__(self, dataset, tokenizer, max_length): 
        pick = min(len(dataset), 1000)
        self.dataset = dataset.select(range(pick))
        self.tokenizer = tokenizer 
        self.max_length = max_length

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        sample = self.dataset[idx]

        instruction = sample["instruction"]
        output = sample["output"]

        prompt_text = f"### Instruction ### \n {instruction} \n "
        response_text = f"### Output ### \n {output} \n " + self.tokenizer.eos_token
        
        prompt_ids = self.tokenizer(prompt_text)["input_ids"]
        response_ids = self.tokenizer(response_text)["input_ids"]

        input_ids = prompt_ids + response_ids 
        
        # Create labels
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncation 
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # Pad both sequences 
        padding_length = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) 

        if padding_length > 0 : 
            input_ids += [self.tokenizer.pad_token_id] * padding_length 
            labels += [-100] * padding_length
            attention_mask += [0] * padding_length

        return {
            "input_ids" : torch.tensor(input_ids), 
            "attention_mask" : torch.tensor(attention_mask), 
            "labels" : torch.tensor(labels)
        }


# Load the Tokenizer, model from transformers library 
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


train_dataset = AlpacaDataset(train_dataset, tokenizer, max_length = 512)
test_dataset = AlpacaDataset(test_dataset, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


print(f"One instance from dataset : {train_dataset[0]['input_ids'].shape}")
print(f"One batch from dataloader : {next(iter(train_loader))['input_ids'].shape}")


# Creating a optimizer 
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 10

# Move the model and data to device 
model = model.to(device)

def generate_response(model, tokenizer, instruction, device):
    model.eval()

    prompt = f"### Instruction ### \n {instruction} \n ### Output ### \n "

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

for epoch in range(1, num_epochs + 1): 
    model.train()
    total_train_loss = 0.0
    start_time = time.time()
    for batch in train_loader: 
        batch = {k : v.to(device) for k, v in batch.items()}
        logits = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], labels=batch["labels"])
        
        loss = logits.loss
        total_train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader: 
            test_batch = {k : v.to(device) for k, v in batch.items()}
            logits = model(input_ids=test_batch["input_ids"],
                           attention_mask=test_batch["attention_mask"], labels = test_batch["labels"])
    
            loss = logits.loss.item()
            total_test_loss += loss
    
        avg_test_loss = total_test_loss/ len(test_loader) 
        perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
        
    print(f"Epoch : {epoch} / {num_epochs} :: Time taken : {time.time() - start_time} :: Train Loss : {avg_train_loss:.4f} :: Test loss : {avg_test_loss:.4f}")


# Save final SFT checkpoint
checkpoint_path = "./checkpoints/sft_final"

model.save_pretrained(checkpoint_path)
tokenizer.save_pretrained(checkpoint_path)

print(f"SFT checkpoint saved to {checkpoint_path}")

after_response = generate_response(
    model, tokenizer, instruction, device
)

print("\nAFTER FINE-TUNING:")
print(after_response)
