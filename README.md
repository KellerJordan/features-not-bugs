# features-not-bugs

Replication of [Adversarial Examples are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175).

Seems to give way better numbers than the paper?


To run:
```
python main.py
```

Expected output:
```
100%|████████████████████████████████████████████████████| 64/64 [01:17<00:00,  1.22s/it]
train correct: 49990
train loss: 0.005844439025968313
test correct: 9380
test correct w/ tta: 9455
100%|██████████████████████████████████████████████████| 100/100 [00:35<00:00,  2.81it/s]
max r: tensor(2.0000)
attack success rate: 0.941
attack success rate (with augmentation): 0.48476
100%|████████████████████████████████████████████████████| 64/64 [01:33<00:00,  1.46s/it]
train correct: 49620
train loss: 0.04633096605539322
test correct: 8106
test correct w/ tta: 8293
```
