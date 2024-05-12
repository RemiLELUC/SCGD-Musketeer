# SGD with Coordinate Sampling: Theory and Practice

This is the supplementary material for the article:  'SGD with Coordinate Sampling: Theory and Practice', *Journal of Machine Learning Research 23 (2022)*,  by Rémi LELUC and François PORTIER. [(paper)](https://www.jmlr.org/papers/v23/21-1240.html)

This implementation is made by [Rémi LELUC](https://remileluc.github.io/).

## Citation

```bibtex
@article{leluc2022sgd,
  title={SGD with Coordinate Sampling: Theory and Practice},
  author={Leluc, R{\'e}mi and Portier, Fran{\c{c}}ois},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={342},
  pages={1--47},
  year={2022}
}
```

## Description 

- optimize_f.mp4: video for visual understanding of optimization paths (see Appendix B)
- optimize_h.mp4: video for visual understanding of optimization paths (see Appendix B)
- code/         : folder with the code of the experimental results, the code is written in Python3

## Folder code/

Dependecies in Python 3
- requirements.txt : dependencies

### Folders description

I)StoFirstOrder_optimization

I.1) linear_models   : contains all the code related to the numerical experiments 
on regularized linear models with stochastic first-order methods (Appendices D and F)

I.2) neural_networks: contains all the code related to the numerical experiments 
with neural networkks (on MNIST, Fashion-MNIST and CIFAR10 datasets) with stochastic first-order methods (Appendix D)

II)ZeroOrder_optimization

II.1) ZO_linear_models   : contains all the code related to the numerical experiments 
on regularized linear models with zero-order r methods (Appendix E)

II.2) ZO_neural_networks: contains all the code related to the numerical experiments 
with neural networkks (on MNIST, Fashion-MNIST and KMNIST datasets) with zero-order methods (Appendix E)

### Folders details

I.1) linear_models:

- graphs/ : contains the graphs (.pdf format) of the article
- results/: contains the results (.npy format) of the numerical experiments for regularized linear models
with a folder appendix/ for all the additional results
- scripts/: contains the Python scripts such as

Python scripts
- utils.py   : tool functions for simulated data
- models.py  : script to implement the Ridge regression and Logistic regression
- adaptive.py: main script with the different methods (SGD, Uniform, Musketeer) 
- simus.py   : script to perform simulations

Python notebooks to load and plot/save the results
- PLOT_Ridge.ipynb   : notebook to load and plot/save the results of Ridge regression
- PLOT_Logistic.ipynb: notebook to load and plot/save the results of Logistic regression

I.2) neural_networks:

- graphs/ : contains the graphs (.pdf format) of the article for Mnist, Fashion, Cifar10
- results/: contains the results (.npy format) of the numerical experiments for Neural Networks
- scripts/: contains the Python scripts such as

Python scripts
- musketeer_optimizer.py: PyTorch optimizer
- train_mnist.py  : script to train NN on Mnist   with SGD and Musketeer (eta=1,2,10)
- train_fashion.py: script to train NN on Fashion with SGD and Musketeer (eta=1,2,10)
- train_cifar10.py: script to train NN on Cifar10 with SGD and Musketeer (eta=1,2,10)

Python notebooks to load and plot/save the results
- plot_graphs_nn.ipynb     : notebook to load and plot/save the graphs of the article
- display_accuracy_nn.ipynb: notebook to load and plot/save the test accuracies

II.1) ZO_linear_models:

- graphs/ : contains the graphs (.pdf format) of the article
- results/: contains the results (.npy format) of the numerical experiments for regularized linear models
with a folder results_appendix/ for all the additional results (Appendix E)
- scripts/: contains the Python scripts and notebooks such as

Python scripts
- models.py: script to implement the Ridge regression and Logistic regression
- source.py: main script with the different methods (Full, Uniform, Nesterov, Musketeer-avg/abs/sqr) 
- simus.py : script to perform simulations
Python notebooks to run experiments and save the results
- ZO_Ridge.ipynb   : notebook to run experiments on ZO-Ridge regression
- ZO_Logistic.ipynb: notebook to run experiments on ZO-Logistic regression
Python notebooks to load and plot/save the results
- PLOT_Ridge.ipynb   : notebook to load and plot/save the results of Ridge regression
- PLOT_Logistic.ipynb: notebook to load and plot/save the results of Logistic regression

II.2) ZO_neural_networks:

- graphs/ : contains the graphs (.pdf format) of the article for Mnist, Fashion, Kmnist
- results/: contains the results (.npy format) of the numerical experiments for Neural Networks
- scripts/: contains the Python scripts such as

Python scripts (ZO optimization)
- train_mnist.py  : script to train NN on Mnist
- train_fashion.py: script to train NN on Fashion-Mnist 
- train_kmnist.py : script to train NN on Kmnist

Python notebooks to load and plot/save the results
- plot_graphs_zo_nn.ipynb : notebook to load and plot/save the graphs of the article
- plt_accuracy_zo_nn.ipynb: notebook to load and plot/save the test accuracies




