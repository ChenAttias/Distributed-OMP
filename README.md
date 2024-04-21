# Distributed-OMP

This repository enables the reproduction of Figures 1-4 of the paper "Recovery Guarantees for Distributed-OMP," accepted for publication at AISTATS 2024. It contains implementations of the distributed Orthogonal Matching Pursuit (OMP) algorithms discussed in the paper, the simulation setup files, and a Jupyter notebook that serves as an execution guide and demonstrates how to reproduce the figures.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 installed
- The following Python libraries are installed: NumPy, SciPy, matplotlib, scikit-learn, rpy2, joblib
- R v4.0.3 installed

## Installation

To install the necessary libraries, run the following command:

```bash
pip install numpy scipy matplotlib scikit-learn rpy2 joblib
```


## Structure of the Repository
This repository contains the following files for conducting simulations and reproducing results:

- setupX.py: Scripts (setup1a.py, setup1b.py, setup2.py, setup3.py, setup4.py) for configuring each simulation scenario.
- algs.py: Implements the distributed OMP algorithms discussed in the paper.
- gens.py: Functions for generating matrices used in simulations.
- helpers.py: Utility functions supporting simulation and algorithm processes.
- run_exp.py: Executes the experiments with given parameters, utilizing the above components.

## Usage

To reproduce the figures from the paper, please follow the instructions in the `execution_guide.ipynb` Jupyter notebook.

## License

This project is licensed under the [GPL2 License](LICENSE.md) - see the file for details.

## Citation

If you use this software in your research, please cite it as follows:

```
@inproceedings{amiraz2024recovery,
  title = 	 {Recovery Guarantees for Distributed-{OMP}},
  author =       {Amiraz, Chen and Krauthgamer, Robert and Nadler, Boaz},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {802--810},
  year = 	 {2024},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR}
}

```

## Contact

If you have any questions, please contact me at `chenamiraz@gmail.com`.
