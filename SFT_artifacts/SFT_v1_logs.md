
## BEFORE FINE-TUNING:
1. Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. 
 2. The process begins with the absorption of light energy by chlorophyll in the thylakoid membranes of the chloroplasts. 
 3. The light energy is used to split water molecules into hydrogen and oxygen, which are released into the atmosphere. 
 4. The oxygen is used by plants for respiration, while the hydrogen is used to produce glucose, which is used for energy and growth. 
 5. The glucose is then used to produce food for the plant, and the process repeats itself. 
 6. Photosynthesis is a crucial process for the survival of

## Training Logs
Epoch : 1 / 10 :: Time taken : 53.05095458030701 :: Train Loss : 1.4493 :: Test loss : 1.4308
Epoch : 2 / 10 :: Time taken : 53.453360080718994 :: Train Loss : 1.0304 :: Test loss : 1.4835
Epoch : 3 / 10 :: Time taken : 53.811354637145996 :: Train Loss : 0.7005 :: Test loss : 1.5947
Epoch : 4 / 10 :: Time taken : 53.782092332839966 :: Train Loss : 0.4337 :: Test loss : 1.8117
Epoch : 5 / 10 :: Time taken : 53.75809168815613 :: Train Loss : 0.2678 :: Test loss : 2.0337
Epoch : 6 / 10 :: Time taken : 53.703777551651 :: Train Loss : 0.1695 :: Test loss : 2.2528
Epoch : 7 / 10 :: Time taken : 53.66743588447571 :: Train Loss : 0.0975 :: Test loss : 2.4191
Epoch : 8 / 10 :: Time taken : 53.6791558265686 :: Train Loss : 0.0605 :: Test loss : 2.6051
Epoch : 9 / 10 :: Time taken : 53.687114238739014 :: Train Loss : 0.0395 :: Test loss : 2.8180
Epoch : 10 / 10 :: Time taken : 53.774578332901 :: Train Loss : 0.0269 :: Test loss : 2.9032
Writing model shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.04s/it]
SFT checkpoint saved to ./checkpoints/sft_final

## AFTER FINE-TUNING:
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy, and then convert that energy into physical and mechanical energy to provide them with food. The process involves the conversion of carbon dioxide and water into glucose, a simple sugar, which the plant or algae can use for energy and growth. The glucose is then stored for future use. 
### Explanation for Experts: 
 Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy, and then convert that energy into physical and mechanical energy to provide them with food. The process involves the conversion of carbon dioxide and water into glucose, a simple sugar, which the



### Key Takeways : 
1. After Fine-tuning the response has these markdowns saying `Explanation for Experts` which is not a clean response message
2. Across epochs the train loss decreases but the test loss increases drastically meaning the model is severly overfitting 
The best checkpoint is from epoch 1 
3. We are storing the last (worst checkpoint) as the model checkpoint to generate responses 

