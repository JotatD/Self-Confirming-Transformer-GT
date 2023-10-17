# SCT

## Source code for [*Self-Confirming Transformer for Locally Consistent Online Adaptation in Multi-Agent Reinforcement Learning.*](https://arxiv.org/abs/2310.04579)
Implementation based on [kzl](https://github.com/kzl/decision-transformer). Dataset and environment based on [ling-pan's fork of MPE](https://github.com/ling-pan/OMAR)

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [References](#references)
	- [Citing](#citing)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

## Background
<img src="/public/3agsct.png"/>

Offline reinforcement learning (RL) leverages previously collected data to extract policies that return satisfying performance in online environments. However, offline RL suffers from the distribution shift between the offline dataset and the online environment. In the multi-agent RL (MARL) setting, this distribution shift may arise from the nonstationary opponents (exogenous agents beyond control) in the online testing who display distinct behaviors from those recorded in the offline dataset. Hence, the key to the broader deployment of offline MARL is the online adaptation to nonstationary opponents. Recent advances in large language models have demonstrated the surprising generalization ability of the transformer architecture in sequence modeling, which prompts one to wonder *whether the offline-trained transformer policy adapts to nonstationary opponents during online testing*. This work proposes the self-confirming loss (SCL) in offline transformer training to address the online nonstationarity, which is motivated by the self-confirming equilibrium (SCE) in game theory. The gist is that the transformer learns to predict the opponents' future moves based on which it acts accordingly. As a weaker variant of Nash equilibrium (NE), SCE (equivalently, SCL) only requires local consistency: the agent's local observations do not deviate from its conjectures, leading to a more adaptable policy than the one dictated by NE focusing on global optimality. We evaluate the online adaptability of the self-confirming transformer (SCT) by playing against nonstationary opponents employing a variety of policies, from the random one to the benchmark MARL policies. Experimental results demonstrate that SCT can adapt to nonstationary opponents online, achieving higher returns than vanilla transformers and offline MARL baselines.

## Install
. To create a conda environment and install the requirements:
```sh
conda env create -n your_env_name -f conda_env.yml
```

## Usage

Train a self-confirming transformer for the simple tag environment with the following command.

```sh
python experiment.py --env simple_tag --dataset expert --opo_ope cat --forward double --path "path/to/data/simple_tag" --type expert --dataset seed_0_data 
```

Train a conjectural transformer for the simple world environment with the following command.

```sh
python experiment.py --env simple_world --dataset expert --opo_ope cat --forward single --path "path/to/data/simple_tag" --type expert --dataset seed_0_data 
```

We provide options for aggregating the observation hidden states with the opo_ope option (cat for concatenation, add for adding, normal for no aggreagation).


## References
You can find the full paper [here](https://arxiv.org/abs/2310.04579).

### Citing:

If you find this work useful in your research, please cite:

```bibtex
@misc{li2023selfconfirming,
      title={Self-Confirming Transformer for Locally Consistent Online Adaptation in Multi-Agent Reinforcement Learning}, 

      author={Tao Li and Juan Guevara and Xinghong Xie and Quanyan Zhu},

      year={2023},

      eprint={2310.04579},

      archivePrefix={arXiv},

      primaryClass={cs.LG}
}
```

## Maintainers
[@Juan Guevara](https://github.com/JotatD). 

## Contributing
Feel free to dive in! [Open an issue](https://github.com/NYU-LARX/Self-Confirming-Transformer/pulls) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors ✨
All authors worked really hard on all aspects of ideation, code, and writing.


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/NYU-LARX/Self-Confirming-Transformer/blob/main/LICENCE.txt) [MIT](https://github.com/NYU-LARX/Self-Confirming-Transformer/blob/main/LICENCE.txt) © NYU Larx Group