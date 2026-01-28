# BatchBALD for Survival Analysis with Probe Depth

This repository implements BatchBALD for survival analysis with oracle probe depth constraints.


## Problem Setting


Here is what I want and the story: I have a survival dataset and I am comparing different active learning methods on it. I want to make a number of adjustments to make it work for survival. For one the survival time has been cut into time bins. Another things is the model output is ensemble outputs as it is a bayesian model so I get a bunch of logits... the output in particular is K,N,T (number of samples from model, number of instances, number of time bins).  Next we introduce for this problem setting an "increment"... the idea is if the increment or probe depth is k, you can only learn about the next k years. So if the probed data instance is censored at time bin 2 and k is 2 they can only learn until bin 4, so if the person died in bin 5 then there is no way to figure it out and the person is left censored at bin 4. Now I am comparing acquisition funcitons in this special data setting. I take in data_train, which has censored and uncensored instances, then I randomly select a proportion of them to artificially censor. What this means is their artificial_time becomes a random time between 0 and their true time (so if they are uncensored they are not censored with artificial_event = 0, and if they are already censored they are further censored with aritifical_event again = 0). What happens is once our acquisition function chooses some points to decensor, these points are sent to an oracle with a predetermined probe depth/increment and only decensors up to that probe depth (the k we discussed earlier, probe depth and increment are the same here). For example, if a person is censored at bin 5, imagine their true time of death is bin 9, and the probe depth is 2, then the oracle makes the person now censored at time bin 7! The model is retrained with this time bin 7 point and evaluated (so in this scenario the true time bin is never revealed! But in some situations it could be!). This is the setting we are playing... On the other hand if the true bin was bin 6 then it would have been revealed by the oracle thus in some sense it is advantageous for the acquisition function to try and pick points it predicts will die within the increment bounds!


- Survival data with discrete time bins
- Bayesian models output (K, N, T) predictions
- Oracle reveals information up to probe depth k
- Single-shot active learning evaluation

## Structure

- `src/data/`: Synthetic survival data generation
- `src/models/`: Bayesian survival models
- `src/oracle/`: Oracle simulation with probe depth
- `src/acquisition/`: Acquisition functions (BatchBALD, entropy, variance)
- `src/evaluation/`: Survival metrics
- `experiments/`: Experimental comparison scripts
- `tests/`: Unit and integration tests

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python experiments/run_comparison.py
```
