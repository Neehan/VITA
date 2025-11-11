# VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting

Official implementation of VITA, a variational pretraining framework that learns weather representations from rich satellite data and transfers them to yield prediction tasks with limited ground-based measurements.

**Paper**: [arXiv:2508.03589](https://arxiv.org/abs/2508.03589)

## Overview

VITA addresses the data asymmetry problem in agricultural AI: pretraining uses 31 meteorological variables from NASA POWER satellite data, while deployment relies on only 6 basic weather features. Through variational pretraining with a seasonality-aware sinusoidal prior, VITA achieves state-of-the-art performance in predicting corn and soybean yields across 763 U.S. Corn Belt counties, particularly during extreme weather years.

## Citation

```bibtex
@misc{hasan2025vitavariationalpretrainingtransformers,
      title={VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting},
      author={Adib Hasan and Mardavij Roozbehani and Munther Dahleh},
      year={2025},
      eprint={2508.03589},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.03589},
}
```
