# Hierarchical Decomposition of Shortcut DFA Large Function Tables: An Empirical Study on Memory-Time Boundaries

This project is based on the open-source implementation of **Shortcut2Secrets** (TCHES 2025) available at [SJTUwwz/DFA_Elisabeth_family](https://github.com/SJTUwwz/DFA_Elisabeth_family). 

We conduct an empirical comparison of several organization methods for the filter table \(T_{\{6,7\}}\) (including uniform 15x4, inverted index, and NLUT symmetric compression) on the Elisabeth-4 and Elisabeth-b4 architectures. The implementation was developed with the assistance of Claude (AI coding assistant).

## Getting Started

### Prerequisites & Dependencies
Make sure you have the following installed:
* Python 3
* numpy
* psutil

### Setup
Before running the experiments, you need to clone the original repository into this directory:
```bash
git clone https://github.com/SJTUwwz/DFA_Elisabeth_family
