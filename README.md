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

This corresponds with the experiment generating `D_rand` for CIFAR-10, which yielded 63.3% accuracy in the paper (Table 1).

To get the accuracy for `D_det`, run
```
python main.py --det
```
Which will yield much worse results than the paper.

To get results for `D_det` which are closer to the paper (~35%), we need to use some noise when generating the PGD perturbations:
```
python main.py --det --num-noise=4
```

The difference in numbers is possibly caused by the paper using ResNet-50 while this code uses a 9-layer ResNet.


