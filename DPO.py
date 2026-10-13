
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from functools import partial
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset("Anthropic/hh-rlhf")
train_dataset, test_dataset = ds["train"], ds["test"]
print(f"Len of train dataset: {len(train_dataset)}") #Len of train dataset: 160800
print(f"Len of test dataset: {len(test_dataset)}") #Len of test dataset: 8552

one_sample = train_dataset[1]
print(f"One sample from train dataset  : {one_sample}")

# SFT model: Your starting policy.
reference_model = AutoModelForCausalLM.from_pretrained("./checkpoints/sft_final")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/sft_final")

if reference_model is not None: 
    print(f"Model loaded successfully")

# Reference model: A frozen copy of the SFT model.
for param in reference_model.parameters(): 
    param.requires_grad = False


# Preference dataset: Contains chosen and rejected responses.
# Goal: Increase the relative probability of chosen responses over rejected responses while regularizing against the reference model through the DPO objective.

class PreferenceDataset(Dataset): 
    def __init__(self, dataset, tokenizer, max_length): 
        self.ds = dataset.select(range(1000)) 
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        self.dataset = [] 
        skipped = 0
        for i in range(len(self.ds)): 
            chosen = self.ds[i]["chosen"] 
            rej = self.ds[i]["rejected"] 
            
            chosen_ids = self.tokenizer(chosen, truncation=True, padding="max_length", max_length=self.max_length) 
            rej_ids = self.tokenizer(rej, truncation=True, padding="max_length", max_length=self.max_length) 

            chosen_inp_ids = torch.tensor(chosen_ids["input_ids"])
            chosen_attn_mask = torch.tensor(chosen_ids["attention_mask"]) # Tokens + padding 
            chosen_real_tokens = chosen_attn_mask.sum() # Tokens = prefix + differed response
    
            rejected_inp_ids = torch.tensor(rej_ids["input_ids"])
            rejected_attn_mask = torch.tensor(rej_ids["attention_mask"])
            rejected_real_tokens = rejected_attn_mask.sum()

            common = self._get_common_prefix_length(chosen_inp_ids, chosen_attn_mask, rejected_inp_ids, rejected_attn_mask)

            if (chosen_real_tokens - common) > 0 and (rejected_real_tokens - common) > 0 : #It ensures both sequences have real tokens remaining after the shared prefix.
                self.dataset.append({"chosen" : chosen, "rejected" : rej})
            else: # fully truncated and no differing response
                skipped += 1

        print(f"Skipped instances : {skipped}")
    
    def __len__(self): 
        return len(self.dataset)

    def _get_common_prefix_length(self, chosen_inp_ids, chosen_attn_mask, rejected_inp_ids, rejected_attn_mask):
        # Find the common prefix 
        common = 0 
        for i in range(self.max_length): 
            if chosen_attn_mask[i] and rejected_attn_mask[i]: 
                if chosen_inp_ids[i] == rejected_inp_ids[i]: 
                    common += 1
                else: 
                    break #common prompt/prefix boundary ended
            else: 
                break # One of the input id is a padding tokens - Padding tokens don't come in our prefix 

        return common

    def __getitem__(self, idx): 
        sample = self.dataset[idx]

        chosen_response = sample["chosen"]
        rejected_response = sample["rejected"]

        chosen_ids = self.tokenizer(chosen_response, truncation=True, max_length=self.max_length)
        rejected_ids = self.tokenizer(rejected_response, truncation=True, max_length=self.max_length)

        chosen_inp_ids = torch.tensor(chosen_ids["input_ids"])
        chosen_attn_mask = torch.tensor(chosen_ids["attention_mask"])

        rejected_inp_ids = torch.tensor(rejected_ids["input_ids"])
        rejected_attn_mask = torch.tensor(rejected_ids["attention_mask"])

        common = self._get_common_prefix_length(chosen_inp_ids, chosen_attn_mask, rejected_inp_ids, rejected_attn_mask)
        
        chosen_response_mask = torch.tensor([0] * common + [1] * (len(chosen_inp_ids) - common))
        rejected_response_mask = torch.tensor([0] * common + [1] * (len(rejected_inp_ids) - common))
        
        #response_mask: 0 for the shared prefix, 1 afterward.
        #attention_mask: 1 for real tokens, 0 for padding.
        
        chosen_response_mask = chosen_response_mask * chosen_attn_mask # Making sure the last padding tokens are 0 
        rejected_response_mask = rejected_response_mask * rejected_attn_mask
        
        return {
            "chosen_inp_ids" : chosen_inp_ids, 
            "chosen_attn_mask" : chosen_attn_mask, 
            "rejected_inp_ids" : rejected_inp_ids, 
            "rejected_attn_mask" : rejected_attn_mask,
            "chosen_response_mask" : chosen_response_mask, 
            "rejected_response_mask" : rejected_response_mask
        }


train_dataset = PreferenceDataset(train_dataset, tokenizer, 512)
test_dataset = PreferenceDataset(test_dataset, tokenizer, 512)

def collate_fn(samples, tokenizer):
    # Apply dynamic padding
    max_len = 0
    for sample in samples: 
        max_len = max(max_len, len(sample["chosen_inp_ids"]), len(sample["rejected_inp_ids"]))

    batch = { 
        "chosen_inp_ids" : [], 
        "chosen_attn_mask" : [], 
        "rejected_inp_ids" : [], 
        "rejected_attn_mask" : [],
        "chosen_response_mask" : [], 
        "rejected_response_mask" : []
    }

    for sample in samples: 
        for key in batch: 
            seq = sample[key]
            cur_len = len(seq) 
            pad_len = max_len - cur_len

            pad_token = None
            if "inp_ids" in key: 
                pad_token = tokenizer.pad_token_id
            else: 
                pad_token = 0

            padding = torch.full((pad_len, ), pad_token, dtype = seq.dtype) 
            seq = torch.cat([seq, padding])

            batch[key].append(seq)

    for key in batch: 
        batch[key] = torch.stack(batch[key])

    return batch
    

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn = partial(collate_fn, tokenizer=tokenizer))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn = partial(collate_fn, tokenizer=tokenizer))

one_batch = next(iter(train_loader)) 
print(one_batch.keys())
print(one_batch["chosen_inp_ids"].shape, one_batch["chosen_attn_mask"].shape)


def find_log_probs(model, input_ids, attn_mask, response_mask): 
    output = model(input_ids = input_ids, attention_mask = attn_mask) 
    logits = output.logits # B, S -> B, S, V
    
    # Target tokens are next token, meaning seq : A, B, C, D then A predicts B given A, B predicts C given A, B and C predicts D given A, B, C
    target_tokens = input_ids[:, 1:] # B, S-1

    # last token has nothing next to predict so remove that , we can remove D since there D doesn't predict anything
    logits = logits[:, :-1, :] # B, S-1, V

    # convert logits into log probs 
    log_probs = F.log_softmax(logits, dim=2)  # B, S-1, V

    # For each token's log probs of all possible tokens - we want the prob of target token
    filtered = torch.gather(log_probs, dim=2, index=target_tokens.unsqueeze(-1)) #  (B, S-1, V) -> (B, S - 1, 1) 
    filtered = filtered.squeeze(-1) # to get one prob for each token in a sequence # (B, S-1)

    mask = attn_mask[:, 1:] # (B, S-1)
    # To get one score per sequence, multiply by mask(pad tokens are zeroed out) and sum the log probs 
    seq_probs = filtered * mask

    # DPO pipeline must ignore the prompt tokens when calculating sequence log-probabilities and score only the assistant's response.
    response_mask = response_mask[:, 1:] 
    seq_probs = seq_probs * response_mask
    
    seq_probs = seq_probs.sum(dim=1) # (B, S-1) -> (B, )

    return seq_probs

policy_model = AutoModelForCausalLM.from_pretrained("./checkpoints/sft_final")
for param in policy_model.parameters(): 
    param.requires_grad = True

# Keep the policy model trainable and ref model in evaluation mode
policy_model.train()
reference_model.eval()

# Move to device
policy_model.to(device)
reference_model.to(device)

optimizer = AdamW(params = policy_model.parameters(), lr=1e-5)
num_epochs = 3
train_batches_cnt = len(train_loader)
beta = 0.1

for epoch in range(1, num_epochs+1): 
    total_loss = 0.0
    for batch in train_loader: 
        chosen_inp_ids = batch["chosen_inp_ids"].to(device)
        chosen_attn_mask = batch["chosen_attn_mask"].to(device)
        rejected_inp_ids = batch["rejected_inp_ids"].to(device)
        rejected_attn_mask = batch["rejected_attn_mask"].to(device)
        chosen_response_mask = batch["chosen_response_mask"].to(device)
        rejected_response_mask = batch["rejected_response_mask"].to(device)
    
        policy_chosen_logprobs = find_log_probs(policy_model, chosen_inp_ids, chosen_attn_mask, chosen_response_mask)
        policy_rejected_logprobs = find_log_probs(policy_model, rejected_inp_ids, rejected_attn_mask, rejected_response_mask)
    
        with torch.no_grad(): 
            reference_chosen_logprobs = find_log_probs(reference_model, chosen_inp_ids, chosen_attn_mask, chosen_response_mask)
            reference_rejected_logprobs = find_log_probs(reference_model, rejected_inp_ids, rejected_attn_mask, rejected_response_mask)
    
        r_c = policy_chosen_logprobs - reference_chosen_logprobs
        r_r = policy_rejected_logprobs - reference_rejected_logprobs
    
        loss = - F.logsigmoid(beta * (r_c - r_r))
        loss = loss.mean() 

        total_loss += loss.item()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch : {epoch}/{num_epochs}, Loss : {(total_loss/train_batches_cnt):.2f}")

policy_model.eval()
with torch.no_grad():
    preference_accuracy = 0 
    ref_accuracy = 0 
    dpo_accuracy = 0
    for batch in test_loader: 
        chosen_inp_ids = batch["chosen_inp_ids"].to(device)
        chosen_attn_mask = batch["chosen_attn_mask"].to(device)
        rejected_inp_ids = batch["rejected_inp_ids"].to(device)
        rejected_attn_mask = batch["rejected_attn_mask"].to(device)
        chosen_response_mask = batch["chosen_response_mask"].to(device)
        rejected_response_mask = batch["rejected_response_mask"].to(device)
    
        policy_chosen_logprobs = find_log_probs(policy_model, chosen_inp_ids, chosen_attn_mask, chosen_response_mask) 
        policy_rejected_logprobs = find_log_probs(policy_model, rejected_inp_ids, rejected_attn_mask, rejected_response_mask) 

        reference_chosen_logprobs = find_log_probs(reference_model, chosen_inp_ids, chosen_attn_mask, chosen_response_mask) 
        reference_rejected_logprobs = find_log_probs(reference_model, rejected_inp_ids, rejected_attn_mask, rejected_response_mask) 

        ref_accuracy += (reference_chosen_logprobs > reference_rejected_logprobs).sum().item()
        dpo_accuracy += (policy_chosen_logprobs > policy_rejected_logprobs).sum().item()
    
    
        r_c = policy_chosen_logprobs - reference_chosen_logprobs
        r_r = policy_rejected_logprobs - reference_rejected_logprobs
    
        batch_acc =  r_c > r_r  # boolean tensor for an entire batch : (B, )
        correct_pairs = batch_acc.sum().item()
    
        preference_accuracy +=  correct_pairs
    
    print(f"Preference Accuracy : {(preference_accuracy/len(test_loader.dataset)):.2f}")
    print(f"Refernce Accuracy : {(ref_accuracy/len(test_loader.dataset)):.2f}")
    print(f"DPO Accuracy : {(dpo_accuracy/len(test_loader.dataset)):.2f}")
