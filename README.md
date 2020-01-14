Universal Approximation with Certified Networks <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
==============================================================================================================================================================================================

This repository contains the associated code for the paper 
[Universal Approximation with Certified Networks](https://openreview.net/forum?id=B1gX8kBtPr) 
which has been accepted to appear at ICLR 2020.


Background
----------

Many systems have been proposed to certify that neural networks are robust to adversarial attack, for example, the SRILab's [ERAN](https://github.com/eth-sri/eran).
In response to the observation that many networks are emperically robust to adversarial attack yet not provable with these techniques, 
provably robust training has been developed, an early example being SRILab's [DiffAI](https://github.com/eth-sri/diffai).

While training neural networks to be certifiably robust with efficient techniques is a continually improving front, there still appears to be a significant gap 
between what these training systems have achieved versus standard robust training. In our paper [Universal Approximation with Certified Networks](https://openreview.net/forum?id=B1gX8kBtPr), 
we strengthening the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) 
by showing that ReLU-networks which are interval-certifiable are also universal approximators.
We targeted interval certification as it is the simplest and most efficient certification method.
Its simplicity implies that our result extends to more precise commonly used domains such as that proposed by [Singh et al. 2019](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf).

<!-- This result suggests that further research might one day uncover network architectures or training methods for efficiently certifiable robust learning. -->

Theorem
--------

Our theorem says that for all d > 0, 
and for every continuous function, f, 
from a compact subset, C, of a real d-dimensional space to the real line, 
there is a ReLU network, n,
such that we can use interval analysis to prove that every box B which is a subset of C, 
n(B) is a subset of f(B)±d. 

Code for Construction
---------------------

![TODO Better Picture Here](https://gitlab.inf.ethz.ch/mirman/universalprovableapproximation/raw/master/media/overview.png)

Our paper proves the above theorem by constructing such a network n, using the fact that f(B) exists.
A visualization of the components of the resulting network is shown above.
In this repository, we provide code in both Haskell and Python for constructing this network given an input function f and the code to compute f(B) exactly.

This repository contains two files:

* Python:  PythonConstruction.py
* Haskell: HaskellConstruction.hs

Python Usage
------------

Tested With:
* Python 3.6.4 
* numpy 1.14.1

The class ```Interval``` provides intervals together with the functionality to
execute the arithmetik operations found in ReLU networks. To initialize an
object of  class ```func_I``` one provides the concrete function
```func``` mapping points in [0,1]<sup>2</sup> to ℝ, a function ```Delta_k_func```
generating the sets Δ<sub>k</sub>, the minimum ```minimum``` and the maximum
```maximum``` of ```func``` on [0,1]<sup>2</sup>. Furthermore, to simplify the
implementation we restirct ourselfs here to Lipschitz continuous function. The
Lipschitz constant ```lip_const``` of ```func``` on [0,1]<sup>2</sup> is the last
argument. 

This is implemented and tested for the functions
```sqrt_func```(**x**):= (1 + x<sub>0</sub> + x<sub>1</sub>)<sup>1/2</sup> and
```square_func```(**x**):= x<sub>0</sub><sup>2</sup> +
x<sub>1</sub><sup>2</sup>. 

Haskell Usage
-------------

Tested With:
* GHC 8.0.1

This module exposes a function "buildProvableNet f l u epsilon delta m" 
which takes a function f represented as closed box to min and max in that region, 
a lower bound vector l of the region, an upper bound vector r of the region, 
an epsilon and a delta to use, and an integer constant m such that if |x - y| <= 1/m, then |f(y)−f(x)| <= delta.

The function "test" demonstrates its usage.  This can be run by executing the following command:

```
ghc HaskellConstruction.hs -e "test"
```


Citing This Work
----------------

```
@inproceedings{
    baader2020universal,
    title={Universal Approximation with Certified Networks},
    author={Maximilian Baader and Matthew Mirman and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=B1gX8kBtPr}
}
```

Contributors
------------

* [Maximilian Baader](https://www.sri.inf.ethz.ch/people/max) - mbaader@inf.ethz.ch
* [Matthew Mirman](https://www.mirman.com) - matt@mirman.com
* [Martin Vechev](https://www.sri.inf.ethz.ch/vechev.php) - martin.vechev@inf.ethz.ch



License and Copyright
---------------------

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [MIT License](https://opensource.org/licenses/MIT)
