# Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design

This is the official GitHub repository accompanying the paper:

**_Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design_** 
Leo Klarner, Tim G. J. Rudner, Garrett M. Morris, Charlotte M. Deane, Yee Whye Teh **ICML 2024**.

<p align="center">
  &#151; <a href="https://arxiv.org/abs/2407.11942"><b>View Paper</b></a> &#151;
</p>

---

**Abstract**: Generative models have the potential to accelerate key steps in the discovery of novel molecular therapeutics and materials. Diffusion models have recently emerged as a powerful approach, excelling at unconditional sample generation and, with data-driven guidance, conditional generation within their training domain. Reliably sampling from high-value regions beyond the training data, however, remains an open challenge -- with current methods predominantly focusing on modifying the diffusion process itself. In this paper, we develop context-guided diffusion (CGD), a simple plug-and-play method that leverages unlabeled data and smoothness constraints to improve the out-of-distribution generalization of guided diffusion models. We demonstrate that this approach leads to substantial performance gains across various settings, including continuous, discrete, and graph-structured diffusion processes with applications across drug discovery, materials science, and protein design.

## Overview

The primary purpose of this repository is to provide everything needed to reproduce the experiments in our paper. It is research code that has been written with fast prototyping and iterative experimentation in mind and we will continue to improve it over the coming weeks. If anything is unclear or not working as intended, please do not hesitate to reach out or open an issue. The repository is organized as follows:

- `small_molecules/`: contains everything pertaining to the Graph-Structured Diffusion for Small Molecules experiments presented in Section 5.1 and Appendix B.2 of the paper.
- `materials/`: contains everything pertaining to the Equivariant Diffusion For Materials experiments from Section 5.2 and Appendix B.3 of the paper. 
- `proteins/`: contains everything pertaining to the Discrete Diffusion for Protein Sequences experiments from Section 5.3 and Appendix B.4 of the paper.

## Algorithm

If you are simply interested in applying context-guided diffusion to your own problem setting, rather than reproducing the results in our paper, you can do so with the following implementation of our regularization term:

```python
def cgd_regularization_term(
    model_predictions,
    context_embeddings,
    diffusion_time_steps,
    covariance_scale_hyper,
    diagonal_offset_hyper,
    target_variance_hyper,
):
    """
    Compute the context-guided diffusion regularization term.

    Args:
        model_predictions: Predictions of the guidance model on a noised 
            context batch sampled from a problem-informed context set.
        context_embeddings: The embeddings of the noised context points, derived 
            either from a pre-trained or randomly initialized model.
        diffusion_time_steps: The diffusion time steps used to noise the context
            points, sampled from the uniform distribution over [0, T].
        covariance_scale_hyper: The covariance scale hyperparameter, used to        
            determine the strength of the smoothness constraints in K(x).
            Optionally scaled with the noising schedule of the forward process.
        diagonal_offset_hyper: The diagonal offset hyperparameter, used to
            determine how closely the predictions have to match m(x).
            Optionally scaled with the noising schedule of the forward process
        target_variance_hyper: The target variance hyperparameter, used to
            determine the level of predictive uncertainty on the context set.

    Returns:
        The context-guided diffusion regularization term.
    """

    # construct the covariance matrix and multiply it with 
    # the covariance scale hyperparameter
    K = torch.matmul(context_embeddings, context_embeddings.T)
    K = K * covariance_scale_hyper

    # add the diagonal offset hyperparameter to the diagonal of K
    K = K + torch.eye(K.shape[0]) * diagonal_offset_hyper

    # split the model predictions into mean and variance heads
    num_output_dims = model_predictions.shape[-1]
    mean_preds = model_predictions[:, :(num_output_dims // 2)]
    var_preds = softplus(model_predictions[:, (num_output_dims // 2):])

    # specify mean functions that encode the desired behavior of
    # reverting to the context set mean and variance hyperparameter
    mean_target = torch.zeros_like(mean_preds) # assuming standardized labels
    var_target = torch.ones_like(var_preds) * target_variance_hyper

    # compute the Mahalanobis distance between the predictions and the
    # mean functions defined above through their log-likelihood under
    # a multivariate Gaussian distribution with covariance K
    means_likelihood = MultivariateNormal(mean_target.T, K)
    vars_likelihood = MultivariateNormal(var_target.T, K)

    mean_log_p = means_likelihood.log_prob(mean_preds.T)
    var_log_p = vars_likelihood.log_prob(var_preds.T)
    log_ps = torch.cat([mean_log_p, var_log_p], dim=0)

    return -log_ps.sum()
```

## Reference

If the paper or code has been useful to you, please consider citing it:

```
@inproceedings{
    klarnercontext,
    title={Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design},
    author={Klarner, Leo and Rudner, Tim GJ and Morris, Garrett M and Deane, Charlotte and Teh, Yee Whye},
    booktitle={Forty-first International Conference on Machine Learning}
}