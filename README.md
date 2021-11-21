# Search-and-Optimization
This is a search and optimization library



# Table of Contents

* [Welcome?](#Welcome)

* [Requirements](#requirements)

* [Documentation](#documentation)

* [How to use](#how-to-use)
* [Algorithms](#Algorithms)
  * Descent Method
    * Gradient Descent
    * Newton's Method
    * Conjugate Descent
  * Stochastic Search
    * Simulated Annealing
    * Cross Entropy Method
    * Search Gradient
  * Path Search
    * A* Search
    * Minimax Search
    * RRT(Rapid Exploring Random Tree)
  * Markov Decision Process
    * Value Evaluation
    * Policy Evaluation
  * Markov Decision Process with Unknown Environment
    * Monte Carlo Policy Evaluation
    * Temporal Difference Policy Evaluatoin
    * Tabular Q-Learning

# Welcome

This is a library for search and optimization algorithms. The basic topics are covered which include Descent Method, Stochastic Search, Path Search, MDP-related and RL related algorithms. By using this library, you are expected to see basic ideas behind the algorithms through simple but intuitive visualizations.

Hope you can have fun in search and optimization! Any problems with the algorithm or implementation or other problems please feel free to contact me!

# Requirements

The implementation is quite simple and intuitive. If you can use conda, you are ready to go! If not, the requirements are:

* Python 3.8.x
* numpy
* matplotlib

# Documentation

Under construction now. You can start with this README file :)

# How to use

1. clone this repo.

```bash
git clone https://github.com/ruipengZ/Search-and-Optimization.git
```

2. Install the required dependency

   * Using conda

     ```bash
     conda env create -f conda.yml
     ```

   * Using pip

     ```bash
     pip install -r requirements.txt
     ```

3. Run certain algorithms in the library then see the visualization.

# Algorithms

This repo covers the basic topics in Search and Optimization, including Descent Method, Stochastic Search, Path Search, MDP-related and RL related algorithms.

## Descent Method

Descent Method in general starts at some initial point and tries to get to the local minimum in descendent way. It includes gradient descent, Newton's Method and conjugate descent.

### Gradient Descent

Gradient Descent is widely used in optimization and machine learning areas because its simple calculation. It tries to take a small step towards the gradient descent direction to minimize the function.

Here are the one and two dimensional example visualization for gradient descent. 

* One Dimensional Case

<img src="/Users/ruipeng/Desktop/CSE257/Search-and-Optimization/Descent_Method/gif/GD_1.gif" style="zoom:80%;" />

* Two Dimensional Case

![](/Users/ruipeng/Desktop/CSE257/Search-and-Optimization/Descent_Method/gif/GD_2.gif)
