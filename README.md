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

## Implementations
To ensure convergence, on the tabular datasets, for the Vanilla, Oversample, Reweighing, RUNNER, FairSmote, and ROC methods, we train 5 epochs. For the FairNeuron method, we train 10 epochs to ensure convergence. And for the adversarial method, we follow previous work and we train 15 epochs. For the image datasets, we train Vanilla, Oversample, and RUNNER for 10 epochs and train the adversarial method for 20 epochs.

## Other Parameters for RUNNER
For easier comparison, we select hyper-parameters for each method to enable the trained models to have relatively close AP values. For our method RUNNER, we set the hyper-parameter k as follows:

|      | Adult | COMPAS | Credit | LSAC | CelebA (wavy) | CelebA (attractive) |
| ---- | ----- | ------ | ------ | ---- | ------------- | ------------------- |
| EO   | 10%   | 5%     | 5%     | 5%   | 50%           | 5%                  |
| DP   | 10%   | 5%     | 5%     | 5%   | 20%           | 5%                  |

## Other Parameters for Adversarial
The learning rate for the adversary is 1e-4.

## Other Parameters for FairNeuron
We follow FairNeuron to conduct a comparison experiment of these hyperparameters. 𝜃 varies between the interval [10−4, 1] and 𝛾 varies between the interval [0.5, 1]. Note that we use logarithmic coordinates for 𝜃 since its value is sampled proportionally.

