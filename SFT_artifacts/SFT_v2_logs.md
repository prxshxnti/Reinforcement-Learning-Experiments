## BEFORE FINE-TUNING:
1. Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. 
 2. The process begins with the absorption of light energy by chlorophyll in the thylakoid membranes of the chloroplasts. 
 3. The light energy is used to split water molecules into hydrogen and oxygen, which are released into the atmosphere. 
 4. The oxygen is used by plants for respiration, while the hydrogen is used to produce glucose, which is used for energy and growth. 
 5. The glucose is then used to produce food for the plant, and the process repeats itself. 
 6. Photosynthesis is a crucial process for the survival of

## Training Logs
Evaluating at 200th step
Evaluation results :: Validation loss : 1.4055
Best loss found at 200th step
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.56s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 400th step
Evaluation results :: Validation loss : 1.4014
Best loss found at 400th step
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.57s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 600th step
Evaluation results :: Validation loss : 1.4001
Best loss found at 600th step
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.81s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 800th step
Evaluation results :: Validation loss : 1.3974
Best loss found at 800th step
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.65s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 1000th step
Evaluation results :: Validation loss : 1.3975
Evaluating at 1200th step
Evaluation results :: Validation loss : 1.3968
Epoch : 1 / 10 :: Time taken : 136.6865622997284 :: Train Loss : 1.4570
Evaluating at 1400th step
Evaluation results :: Validation loss : 1.4377
Exiting inner loop due to exceed in patience limits
Exiting outer loop due to exceed in patience limits
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 13454.14it/s]


## AFTER FINE-TUNING:
Photosynthesis is the process by which plants and other photosynthesizing organisms use sunlight to convert carbon dioxide and water into glucose and oxygen. During photosynthesis, the plant uses the energy from sunlight to convert carbon dioxide and water into glucose, which is then used as a source of energy for the plant. The glucose is then used to produce energy for the plant, and the oxygen is released into the atmosphere. 

### Key Takeways : 
1. After Fine-tuning the response is much better and doesn't have irrelevant markdowns.
2. Every 200 steps is evaluated on a validation dataset and loss is decreasing consistently
3. We are storing the best checkpoint as the model checkpoint to generate responses 