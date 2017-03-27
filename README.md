README
======

This repository contains two implementations of the model that is described in 
the paper [_From calls to communities: a model for time varying social 
networks_](https://arxiv.org/abs/1506.00393) by Laurent _et al._.
It takes the original approach of a activity-driven network model by Perra 
_et al._ and adds three additional mechanisms:

1. a reinforcement process to model memory-driven interaction dynamics
2. focal and cyclic closure to capture patterns responsible for emerging community structures
3. a node removal process 

Note that the model implemented here additionally includes a peer influence mechanism, which was the introduced in my Master’s thesis.



additional requirements
-----------------------

  * graph-tool
