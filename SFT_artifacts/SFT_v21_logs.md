Training samples count : 5000
Test samples count : 500

## BEFORE FINE-TUNING:
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. The process involves the absorption of light energy by chlorophyll in the thylakoid membranes of the chloroplasts, which then use this energy to convert water and carbon dioxide into glucose and oxygen. The glucose produced is used by the plant as a source of energy and nutrients, while the oxygen released is released into the atmosphere as a byproduct.

## Training Logs
Evaluating at 200th step
Evaluation results :: Validation loss : 1.4565
Best loss found at 200th step
Writing model shards: 100%|██████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.77s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 400th step
Evaluation results :: Validation loss : 1.4516
Best loss found at 400th step
Writing model shards: 100%|██████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.82s/it]
SFT checkpoint saved to ./checkpoints/sft_final
Evaluating at 600th step
Evaluation results :: Validation loss : 1.4559
Evaluating at 800th step
Evaluation results :: Validation loss : 1.4550
Evaluating at 1000th step
Evaluation results :: Validation loss : 1.4537
Exiting inner loop due to exceed in patience limits
Exiting outer loop due to exceed in patience limits
Loading weights: 100%|████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 12811.08it/s]

## AFTER FINE-TUNING:
Photosynthesis is the process by which plants and other photosynthesizing organisms use light energy to convert carbon dioxide and water into glucose and oxygen. The process begins when light energy is absorbed by the chlorophyll in the plant's leaves, which then uses the energy to convert carbon dioxide and water into glucose and oxygen. The glucose is then used to produce energy for the plant, while the oxygen is released into the atmosphere.
