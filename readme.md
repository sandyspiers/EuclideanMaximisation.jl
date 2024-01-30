# Euclidean Max-Sum Cut Algorithm (EMSCA)

**DEPRECATED VERSION**: This version was first submitted to COR and arXiv, and is now deprecated.
However, the new code base is written in Julia, so we keep this Python version avalible for those interested.

This repo contains an exact solver for binary quadratic programs type

```txt
max  <Qx,x>
s.t. Ax <= a
     x >= 0.
```

where `Q` is a Euclidean distance matrix.
This repo is based on [1].

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

## Results Replication

To replicate the results from [1], firstly populate `results/data` with benchmark instances from the [MDPLIB 2.0 test library](https://www.uv.es/rmarti/paper/mdp.html) (see `results/readme.md` for more information).
Then, run

```bash
python results/run.py
```

This will run through all numerical experiments described in [1], and save the results to `results/results.log`.

## References

[1] Hoa T. Bui, Sandy Spiers, Ryan Loxton, 2023. *Cutting Plane Algorithms are Exact for Euclidean Max-Sum Problems*, Manuscript under review.
