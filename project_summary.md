# Project Summary: Model Optimization for m-height Prediction

This document summarizes the iterative process of experimenting with and optimizing a ResNet-based model for the m-height prediction task.

## Initial Baseline

The starting point was a `ResNet` architecture using the **ReLU** activation function, which had a baseline validation loss of **1.13**. All subsequent experiments were aimed at improving upon this baseline.

## Phase 1: Activation Function Exploration

The first set of experiments involved replacing the standard `ReLU` activation function with more modern alternatives to see if they could improve performance.

1.  **SwiGLU:** This was the initial idea. A `ResNet` with `SwiGLU` was implemented and trained.
2.  **GELU:** A popular activation function from Transformer architectures, known for its smoothness.
3.  **LeakyReLU:** A simple but effective variant of `ReLU` designed to prevent the "dying ReLU" problem.

**Result:** While all models trained successfully, none of them outperformed the original `ReLU` baseline. `LeakyReLU` was the best performer among the new activations.

## Phase 2: Architectural Improvements - Attention Mechanism

To help the model focus on more relevant features, a **Squeeze-and-Excitation (SE) block** was integrated into the `ReLU-ResNet` architecture.

*   **Initial Training:** The first training run was stopped early and showed a worse performance than the baseline.
*   **Continued Training:** We observed that the loss was still decreasing, so we continued the training. This significantly improved the result, making the `SE-ResNet` very competitive.

## Phase 3: Hyperparameter Tuning

Given the promising results of the `SE-ResNet`, we used the **Optuna** framework to perform a hyperparameter search to find the optimal configuration for this architecture. The search included learning rate, weight decay, number of blocks, and the SE block's reduction ratio.

**Result:** The fine-tuned `SE-ResNet` achieved a new best validation loss of **1.1273**, slightly outperforming the original `ReLU` baseline.

## Phase 4: Advanced Mathematical Constraints

We explored two novel strategies to embed domain knowledge directly into the model and loss function.

1.  **Violation-Informed Loss:** A custom loss function was implemented to penalize the model if it predicted an m-height lower than the height of any randomly sampled codeword from the corresponding generator matrix. An accelerated, vectorized version was created to improve training speed.
2.  **Lower Bound Enforcement:** The model's output layer was modified to be `1.0 + Softplus(z)`, mathematically guaranteeing that the predicted m-height would always be greater than or equal to 1.

These strategies were tested on both the `SE-ResNet` and the original `ReLU-ResNet`.

**Result:** The combination of the **`ReLU-ResNet` + Lower Bound + Violation Loss** proved to be highly effective. We then ran another Optuna hyperparameter search on this specific combination.

## Final Results Summary

After all experiments, here is the final ranking of the models based on their best validation LogMSE loss:

| Rank | Model Description                                       | Best Validation Loss (LogMSE) |
| :--- | :------------------------------------------------------ | :---------------------------- |
| 1    | **ReLU-ResNet + Lower Bound + Violation Loss (Fine-Tuned)** | **1.1231**                    |
| 2    | Fine-Tuned SE-ResNet                                    | 1.1273                        |
| 3    | Original ReLU ResNet (Baseline)                         | 1.13                          |
| 4    | SE-ResNet + Lower Bound & Violation Loss                | 1.1301                        |
| 5    | SE-ResNet + Violation Loss                              | 1.1343                        |
| 6    | ReLU-ResNet + Violation Loss                            | 1.1402                        |
| 7    | LeakyReLU ResNet                                        | 1.1977                        |
| 8    | GELU ResNet                                             | 1.2126                        |
| 9    | SwiGLU ResNet                                           | 1.2449                        |
| 10   | ResNet SwiGLU with Lower Bound                          | 1.2442                        |

## Conclusion

The most successful model was the **`ReLU-ResNet` combined with both the Lower Bound enforcement and the custom Violation Loss, after being fine-tuned with Optuna**. This indicates that embedding the problem's mathematical constraints directly into the model's architecture and training process increase the performance of the model.
