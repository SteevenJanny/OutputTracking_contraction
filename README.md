# Deep learning-based output tracking via regulation and contraction theory

This repository contains the code associated to the paper.

## Abstract

In this paper, we deal with output tracking control problems for input-affine
nonlinear systems. We propose a deep learning-based solution whose foundations lay on control
theory. We design a two-step controller: a contraction-based feedback stabilizer and feedforward
action. The first component guarantees convergence to the steady-state trajectory on which the
tracking error is zero. The second one is inherited from output regulation theory and provides
forward invariantness of such a trajectory along the solutions of the system. To alleviate the
need for heavy analytical computations or online optimization, we rely on deep neural networks,
where we link their approximation error to the tracking one. Mimicking the analytical control
structure, we split the learning task in two separate modules. We propose a switching objective
function balancing feasibility of the solution and performance improvement. To validate the
proposed design, we test our solution on a challenging environment.

## Training 
You can train the model by running the following command:

**Leader**:
```python train_leader.py --epochs 500 --lr 0.001 --window_length 200 --noise_range 0.05```

**P metric**:
```python find_P.py --epochs 100 --lr 0.003 --train_parameter 'true'```

**Policy**: 
```python find_beta.py --epochs 100 --lr 0.003```

## Testing
You can test the model by running the following command:

```python evaluate_controller.py ```

