# EuclideanMaximisation.jl

This repo is associated with `Solving Euclidean Max-Sum problems exactly with cutting planes` [1].
It contains two exact cutting-plane solvers for binary quadratic programs of type

```txt
max  <Qx,x>
s.t. Ax <= a
     x >= 0.
```

where `Q` is a Euclidean distance matrix.

## Usage

To use one of the available solvers, begin by creating an `EmsModel` instance.
This object should contain a `JuMP` model, a Euclidean distance matrix, and a list of location-associated decision variables.

```julia
using JuMP
using EuclideanMaximisation.Model: EmsModel, build_edm, check_model_valid
# create the JuMP mip model
mdl = JuMP.Model()
n = 10
x = @variable(mdl, 0 <= x[1:n] <= 1, Bin)
weights = rand(0:100, n)
capacity = sum(weights) * 0.1
@constraint(mdl, weights' * x <= capacity)
# create the euclidean distance matrix
locations = rand(0:100, (n, 2))
edm = build_edm(locations)
# create ems model
ems = EmsModel(; mdl = mdl, loc_dvars = x, edm = edm)
# check the mode
@assert check_model_valid(ems)
```

Then call `solve!(ems)` to solve the model using repeated outer-approximation.
To change to a different solution method, use `solve!(ems, method = "fcard")`.
The solutions methods are described in more detail in the next section.

```julia
using EuclideanMaximisation.Solvers: solve!
solve!(ems)
```

The `solve!` function respects the time limit set on the JuMP model.

## Solvers

- **Repeated ILP**: `repoa`

The first method generates valid cuts by solving the outer approximation subproblem to optimality.
This is described in detail in section 2.2 of [1].

- **Forced Cardinality**: `fcardXX`

The second cutting method fixes cardinality at maximum, and iteratively reduces this until an optimal solution is confirmed.
Each iteration is solved using a branch-and-cut approach, and the cuts are reused in future iterations.
An upper bounding problem is solved after each iteration to help terminate early.

LP-cuts can be added at each iterations to help quickly approximate the objective function.
To keep solutions close to integer, a trust-region constraint is added to keep LP solutions close to the best known incumbent.
To incorporate LP-tangents, add an integer in [0,100] to the end of "fcard".
This will then be interpreted a as percentage trust region.
This means "fcard" and "fcard0" are equivalent and do not add any LP tangents, as there is essentially 0 trust region.
"fcard50" allows LP tangent to be added that are within $0.5n$ of the incumbent solution.
"fcard100" add all LP tangents, ignoring the trust region constraint.
This method is described in detail in section 2.2 of [1].

- **Glover linearisation**: `glov`
- **Quadratic Programming**: `quad`

## Experiments

You can use a YAML file to setup and run a large scale experiments to test the different solution algorithms.
For example:

```yaml
- name: fcdp-1-thread
  run: true
  generator: file_capacitated_diversity_problem
  solver: [repoa, fcard, fcard50, fcard100, quad, glov]
  optimizer: [CPLEX, Gurobi]
  timelimit: 600
  workers: 16
  optimizer_threads: 1
  instance:
    filename: data/instance/CDP (Const.)/*

- name: fcdp-16-thread
  run: true
  generator: file_capacitated_diversity_problem
  solver: [repoa, fcard, fcard50, fcard100, quad, glov]
  optimizer: [CPLEX, Gurobi]
  timelimit: 600
  workers: 1
  optimizer_threads: 16
  instance:
    filename: data/instance/CDP (Const.)/*

- name: rcdp-1-thread
  run: true
  generator: random_capacitated_diversity_problem
  solver: [repoa, fcard, fcard50, fcard100]
  optimizer: [CPLEX, Gurobi]
  timelimit: 600
  workers: 16
  optimizer_threads: 1
  instance:
    n: [1000, 1500, 2000, 2500, 3000]
    s: [2, 10, 20]
    b: [0.2, 0.3]
    seed: 1
    repeats: 5
```

This YAML file defines 3 experiments to conduct.
You can then run these experiments using

```julia
using EuclideanMaximisation.Experimenter:run_experiment
run_experiment("experiments.yml")
```

or by using the docker file.

The setup of each experiment is then saved in its own yaml file under `data/setup/experiment_name.yml`, and the results saved under `data/setup/experiment_name.csv`.

## Results Analysis

We have saved the results reported in [1], and saved their performance profiles under the `analysis` folder.

## Docker

A docker file is provided for reproducibility.
To use the docker file, first prepare the `bin/` folder by adding the following files:

- `mdplib_2.0.zip`: Avaliable [here](https://universitatdevalencia-my.sharepoint.com/personal/rafael_marti_uv_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frafael%5Fmarti%5Fuv%5Fes%2FDocuments%2FMis%20Datos%2FPapers%2FMaximum%20Diversity%2FInstancias%2Fmdplib%5F2%2E0%2Ezip&parent=%2Fpersonal%2Frafael%5Fmarti%5Fuv%5Fes%2FDocuments%2FMis%20Datos%2FPapers%2FMaximum%20Diversity%2FInstancias&ga=1)
- `cplex*.bin` installer
- `gurobi.lic` license
- `startup.jl` (provided)
- `cplex_intaller.properties` (provided)

Note that to run any file-based experiments from MDPLIB, you must first unzip this package and put the contents in `data/instance`.
The easiest way to do this is to run

```shell
./sh/unzip_mdplib.sh
```

Then, to open a REPL session with everything loaded, run

```shell
docker compose run euclid-max
```

To run the experiments defined in `experiments.yml` in the background, run

```shell
nohup docker compose run experiments &
```

This will run the experiments in the background, as they can take a long time.

## References

[1] [Bui, H. T., Spiers, S., & Loxton, R. (2024). Solving Euclidean Max-Sum problems exactly with cutting planes. Computers & Operations Research, 168, 106682.](https://doi.org/10.1016/j.cor.2024.106682)
