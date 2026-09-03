"""

File to test stuff Prof asked for





NEED TO TEST

A. Batched QP solve operations
    a. How many QPs can we solve in parallel?
    b. How does the solve time scale with the number of QPs?
    c. Ensure everything runs on GPU (obviously)
B. Plots
    a. Test plotting functions on sbx-rl agent
    b. "" For dry test of nn-mpc agent



"""


import jax
import jax.numpy as jnp

import moreau
from moreau.jax import Solver

