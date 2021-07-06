# gena

## Virtual environment

```bash
# to install dependencies
poetry install

# to enter an interactive shell
poetry shell

# to exit an interactive shell
exit

# how to add a package
poetry add torch

# or
poetry add torch==1.4.0

# or
poetry add torch>=1.4.0

# or modify pyproject.toml file and then run (this will update poetry.lock file)
poetry lock

# dont forget to commit and push poetry.lock and pyproject.toml files
```
