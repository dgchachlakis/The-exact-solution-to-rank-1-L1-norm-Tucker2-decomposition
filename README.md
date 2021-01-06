## The exact solution to rank-1 L1-norm Tucker2 decomposition ##

In this repo we implent algorithms for the exact solution to rank-1 L1-norm Tucker2 decompostion of 3-ways tensors as they have been presented in [[1]](https://ieeexplore.ieee.org/document/8248754). 

Formally, given a collection of matrix measurements $\mathbf X_1, \mathbf X_2,\ldots, \mathbf X_N \in \mathbb R^{D \times M}$, the scripts provided solve
$$\underset{\begin{smallmatrix}\mathbf u \in \mathbb R^D~;~\|u\|_2=1\\\mathbf v \in \mathbb R^D~;~\|v\|_2=1\end{smallmatrix}}{\text{max.}}\sum\limits_{n=1}^N |\mathbf u^\top\mathbf X_n\mathbf v|,$$
exactly. 

* IEEEXplore article: https://ieeexplore.ieee.org/document/8248754
* arXiv Preprint: https://arxiv.org/abs/1710.11306
* Source code: https://github.com/dgchachlakis/The-exact-solution-to-rank-1-L1-norm-Tucker2-decomposition

---
**Questions/issues**

Inquiries regarding the scripts provided below are cordially welcome. In case you spot a bug, please let me know. If you use some piece of code for your own work, please cite the article above.

---
**Citing**

If you use our algorihtms for the exact solution to rank-1 L1-norm Tucker2 decomposition, please cite [[1]](https://ieeexplore.ieee.org/document/8248754).
```
@article{rank1L1Tucker2,
    author={P. P. {Markopoulos} and D. G. {Chachlakis} and E. E. {Papalexakis}},
    journal={IEEE Signal Processing Letters}, 
    title={The Exact Solution to Rank-1 L1-Norm TUCKER2 Decomposition}, 
    year={2018},
    volume={25},
    number={4},
    pages={511-515},
    doi={10.1109/LSP.2018.2790901}}
```
|[[1]](https://ieeexplore.ieee.org/document/8248754)|P. P. Markopoulos, D. G. Chachlakis and E. E. Papalexakis, "The Exact Solution to Rank-1 L1-Norm TUCKER2 Decomposition," in IEEE Signal Processing Letters, vol. 25, no. 4, pp. 511-515, April 2018.|
|-----|--------|

 
