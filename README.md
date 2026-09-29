# Conditional 1-D diffusion experiments

Research scratch repository built on top of
[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
(MIT, Phil Wang). The upstream code is his. The additions listed below are mine and were
written in summer 2024 as part of an exploratory project on conditional diffusion models for
low-dimensional, regression-style data (a response vector conditioned on a continuous
covariate vector) rather than images.

## What was added

Everything lives in `denoising_diffusion_pytorch/` next to the upstream modules. The package
`__init__.py` is intentionally emptied, so import modules by path, for example
`from denoising_diffusion_pytorch.elucidated_diffusion_1d_cond import ElucidatedDiffusion1Dcond`.

| Module | What it is |
|---|---|
| `elucidated_diffusion_1d.py` | EDM (Karras et al., 2022) preconditioning, Heun and DPM-Solver++ samplers and a minimal training loop, adapted to 1-D sequences with unbounded, un-normalised data. |
| `elucidated_diffusion_1d_cond.py` | The same with a continuous conditioning vector threaded through preconditioning, both samplers and training, plus train/test loss tracking. |
| `karras_unet_1d_cont.py` | Magnitude-preserving U-Net (Karras et al., 2023) for 1-D inputs, where a continuous conditioning vector is embedded and combined with the noise-level embedding in place of class labels. |
| `cond1D.py` | Classifier-free-guidance conditional DDPM (U-Net, Gaussian diffusion, trainer) ported to 1-D with continuous rather than categorical conditioning. |
| `guided_diffusion1d.py` | 1-D DDPM with classifier-guidance hooks (`cond_fn`) in the ancestral and DDIM samplers. |
| `class_free_simple.py` | A minimal from-scratch implementation of the classifier-free guidance loss and sampler in the log-SNR parameterisation of Ho and Salimans. |
| `toy1D.py`, `1Dtest.ipynb` | A scratch toy model and a smoke test of the upstream 1-D API on random data. |

Smaller changes to upstream files: `karras_unet_1d.py` (scalar-safe interpolation, ported
learning-rate schedule, conditioning hooks) and reformatting plus annotations in
`denoising_diffusion_pytorch_1d.py`, `elucidated_diffusion.py` and `karras_unet.py`.

## License

MIT, as upstream. Copyright for the original code remains with Phil Wang; see `LICENSE`.
