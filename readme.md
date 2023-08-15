# Euclidean Max-Sum Cut Algorithm (EMSCA)

This repo contains an exact solver for the general Euclidean max-sum problems of type

```txt
max  <Qx,x>
s.t. Ax <= a
     x >= 0.
```

## Dependencies

- numpy
- docplex
- cplex

## Contents

- `emsca/`
  - This directory contains the key part of this repo, that is the `EmsModel` in `emsca.model`.
    This model is used to create and solve Euclidean max-sum problems.
    The `EmsModel` is just an inheritance of a `docplex.mp` model, and so constraints and variables can be added in a similar way.
    There are 4 exact solvers, documented in `docs/algorithms.md` and in [1].
- `tests/`
  - Pytest for the `EmsModel`
- `docs/`
  - Documenation on the algorithms and parameters used
- `results/`
  - Documentation, scripts and analysis for the results shown in [1].
    See `results/readme.md` for more information.

## References

[1] Hoa T. Bui, Sandy Spiers, Ryan Loxton, *Cutting Plane Algorithms on Euclidean Distance Maximisation are Exact*, Manuscript under review.
