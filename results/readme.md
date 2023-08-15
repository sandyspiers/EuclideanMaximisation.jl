# Numerical Results

This director contains numerical results of the **Euclidean Max-Sum Cutting Algorithms** on various test sets.
These results have been included in [1].
For a detailed discussion on these results, please refer to [1].

## Contents

- `data/`
  - Constrained diversity problem (`CDP/`) and generalized diversity problem (`GDP/`) test instances, available within the [MDPLIB 2.0 test library](https://www.uv.es/rmarti/paper/mdp.html).
    An instance list is added to each folder for reproducibility.
  - Dataset of Australia postcodes, `australian_postcodes.csv`, available [here](https://github.com/matthewproctor/australianpostcodes).
  - Instance generators for randomized instances of the CDP and GDP.  
    These generators produced the instances used in [1].
- `results.log`, recording the performance of all solution methods on the above test instances (excluding the post-code dataset, see `analysis/australian_postcode_test` for more).
- `run_experiment.py` used to read and run the above test instances.
- `analysis/`
  - Scripts and python notebooks used to analyse the numerical results of the different algorithms.

## References

[1] Hoa T. Bui, Sandy Spiers, Ryan Loxton, *Cutting Plane Algorithms on Euclidean Distance Maximisation are Exact*, Manuscript under review.
