# RUNNER
This is the code repository of paper **RUNNER: Responsible UNfair NEuron Repair for Enhancing Deep Neural Network Fairness**.

This repository implements RUNNER and a variety of baselines on the Adult dataset. More code on other datasets would be released after acceptance.

## Operation
For example, run Vanilla train

```
python main.py --method van --mode dp --lam 1
```

run RUNNER

```
python main.py --method NeuronImportance_GapReg --mode eo --lam 5 --neuron_ratio 0.05
```

## Parameters for RUNNER
For easier comparison, we select hyper-parameters for each method to enable the trained models to have relatively close AP values. For our method RUNNER, we set the hyper-parameter k as follows:

|      | Adult | COMPAS | Credit | LSAC | CelebA (wavy) | CelebA (attractive) |
| ---- | ----- | ------ | ------ | ---- | ------------- | ------------------- |
| EO   | 10%   | 5%     | 5%     | 5%   | 50%           | 5%                  |
| DP   | 10%   | 5%     | 5%     | 5%   | 20%           | 5%                  |

## Parameters for Baselines
