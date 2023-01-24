
## Requirements

- Python 3.10.7
- pipenv (2022.10.11) for virtual environment management

## Setup

First clone the repository from Github:
```
git clone https://github.com/ezegonmac/TFG-CellularAutomata
```

Create a virtual environment:
```
cd TFG-CellularAutomata
pipenv shell
```

Install requirements:
```
pip install -r requirements.txt
```

## Usage

The main package is `src`. It contains the following principal subpackages:

- `CA`: Contains the CA class and the 3 rule types.
- `datasets`: Contains the definition of datasets and the functions to build them.
- `statistics`: Contains the functions to generate statistical plots of the datasets.
- `learning`: Contains the functions to train, evaluate and test Machine Learning models.

Those subpackages are explained in detail in the [Reference](reference.md) section.
Their functions must be called from their specific `main` files. Those files are located in the `src` folder.

Those main files are:

- `main_datasets.py`: Main functions for generating the default datasets.
- `main_density_datasets.py`: Main functions for generating the density datasets.
- `main_density_plots.py`: Main functions for generating the density plots.
- `main_doc`: Main functions for generating the documentation figures.
- `main_learning.py`: Main functions for managing the learning process.
- `main_states.py`: Main functions for generating states plots.

Now you can comment/uncomment the functions you want to run in those files, or create your own functions.
You can run those files from the `src` folder with the following command:
```
python src/main_file.py
```

If you dont have your virtual environment activated, you should run the following command first:
```
pipenv shell
```
