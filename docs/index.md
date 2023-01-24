
**Date**:
24-01-2023

**Useful links**:
[Source Code](https://github.com/ezegonmac/TFG-CellularAutomata)

This site contains the project documentation for the
`Cellular Automata Properties Prediction` Final Degree Project
developed by Ezequiel González for his 22-23 last year in ETSII college of Seville, Spain.

It lets you generate different types of datasets based on 3 rule types.
With those you can:

- Generate interesting plots based mainly in their density evolution over time.
- Plot individual estates.
- Train, evaluate and test Machine Learning models to predict new data.
- Compare those models performance with plots.

## Main Workflow

The main idea is to provide an easy to use library to make this process as automated as possible.

First you define the CA based on certain attributes like initial state, size, rules etc. Those 3 rules can be also easy extended. This is made in the package `CA`.

Then you can build the dataset with this new CA and the free parameters. Those datasets are stored in `data/datasets`. This is made in package `datasets`.

Now, with the new built dataset, you can start analyzing it via the multiple plots that are available in the package `statistics`.

Finally, you can train, evaluate and test Machine Learning models to predict new data. This is made in the package `learning`.

## Table Of Contents

1. [Getting Started](getting-started.md)
2. [Reference](reference.md)
3. [Explanation](explanation.md)