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
The hidden size of MLP is 200. We use Adam as the learning optimizer and the batch size is set to 1000 for the $\Delta$ DP metric and 2000 for the $\Delta$ EO metric. The learning rate is set as 0.001. We use Adam as the learning optimizer and the batch size is set as 64 for the $\Delta$ DP metric and 128 for the $\Delta$ EO metric. The learning rate is set as 0.0001.
To ensure convergence, on the tabular datasets, for the Vanilla, Oversample, Reweighing, RUNNER, FairSmote, and ROC methods, we train 5 epochs. For the FairNeuron method, we train 10 epochs to ensure convergence. And for the adversarial method, we follow previous work and we train 15 epochs. For the image datasets, we train Vanilla, Oversample, and RUNNER for 10 epochs and train the adversarial method for 20 epochs.

## Other Parameters for RUNNER
For easier comparison, we select hyper-parameters for each method to enable the trained models to have relatively close AP values. For example, to achieve this purpose, the $\lambda$ is set as 1.0 and 0.3 for the $\Delta$ EO and $\Delta$ DP on the Adult dataset. For our method RUNNER, we set the hyper-parameter k as follows:

|      | Adult | COMPAS | Credit | LSAC | CelebA (wavy) | CelebA (attractive) |
| ---- | ----- | ------ | ------ | ---- | ------------- | ------------------- |
| DP   | 50%   | 5%     | 5%     | 5%   | 20%           | 5%                  |
| EO   | 5%   | 5%     | 5%     | 5%   | 50%           | 20%                  |

Different from Vanilla, Oversample, Reweighing, and FairSmote, other methods rely on hyper-parameters setting. We introduce the hyper-parameter settings as follows:
## Other Parameters for Adversarial
The learning rate for the adversary is 1e-4. The training loss is L = $L_{cls}$ + $\lambda$ $L_{adv}$. The $\lambda$ is set as 0.5. 

## Other Parameters for FairNeuron
We follow FairNeuron to conduct a comparison experiment of these hyperparameters. 𝜃 varies between the interval [1e-4, 1] and 𝛾 varies between the interval [0.5, 1]. Note that we use logarithmic coordinates for 𝜃 since its value is sampled proportionally.

## Other Parameters for Decision theory for discrimination-aware classification.
The threshold $\theta$ is set as 0.6.
