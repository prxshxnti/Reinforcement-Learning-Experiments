
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
Count of samples : 1000
Skipped instances : 7
Skipped instances : 11

Epoch : 1/3, Loss : 0.74
Epoch : 2/3, Loss : 0.16
Epoch : 3/3, Loss : 0.10
Preference Accuracy : 0.59
Refernce Accuracy : 0.55
DPO Accuracy : 0.56

Count of samples : 3000
Skipped instances : 25
Skipped instances : 39

Epoch : 1/3, Loss : 0.76
Epoch : 2/3, Loss : 0.16
Epoch : 3/3, Loss : 0.12
Preference Accuracy : 0.57
Refernce Accuracy : 0.53
DPO Accuracy : 0.53

Count of samples : 5000
Skipped instances : 37
Skipped instances : 113

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