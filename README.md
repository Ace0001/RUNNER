# RUNNER
This is the code repository of paper **RUNNER: Responsible UNfair NEuron Repair for Enhancing Deep Neural Network Fairness**.

This repository implements RUNNER and a variety of baselines.

## Operation
For example, run Vanilla train

```
python main.py --method van --mode dp --lam 1
```

run RUNNER

```
python main.py --method NeuronImportance_GapReg --mode eo --lam 5 --neuron_ratio 0.05
```
