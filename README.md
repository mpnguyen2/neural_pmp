# Neural Pontryagin maximum principle
Pontryagin maximum principle based neural ODE net for handling optimal control problem with time-series data in both finite and infinite dimensional settings

A few examples in finite-dimensional setting:
TBA (Updates will be available soon)

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Running NeuralPMP

To run the code, use the following command:
```
bash run_train.sh
```

## Default Hyperparameters
Training Env: Cartpole (deterministic version)
### Hamiltonian Network
Total number of episodes: 1024
Max number of iterations: 1000000
Update interval: 10
Rate to train while sampling: 1
Sample_batch_size: 32
Batch size: 32
Learning rate: 0.001

### Adjoint Net Training
Total number of episodes: 256
Max number of iterations: 2000
Batch size: 128
Learning rate: 0.0001