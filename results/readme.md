# Numerical Results

This director contains numerical results of the **Euclidean Max-Sum Cutting Algorithms** on various test sets.
These results have been included in [1].

## Contents

- `data/`
  - Constrained diversity problem (`CDP/`) and generalized diversity problem (`GDP/`) test instances, available within the [MDPLIB 2.0 test library](https://www.uv.es/rmarti/paper/mdp.html).
  - [Australian post code location dataset](https://github.com/matthewproctor/australianpostcodes).
  - Instance generators for randomized instances of the CDP and GDP.  These generators produced the instances used in [1].
- `results.log`, outlining the performance of all solution methods on the above test instances (excluding the post-code dataset, see analysis for more information).
- `run.py` used to read and run the above test instances.
- `analysis/`
  - Scripts and python notebooks used to analyse the numerical results of the different algorithms.

## References

[1] Hoa Bui, Sandy Spiers, Ryan Loxton, *Cutting Plane Algorithm on Euclidean Distance Max-Sum is Exact*
