# fdth

A python port of the [fdth](https://github.com/jcfaria/fdth) R library.

Feito como projeto da disciplina "Probabilidade e Estatística" (Ciência
da Computação, UESC).

TODO: contributors, credits and license (public domain?)

## examples

Examples can be found at the `examples/python` folder.

## development

First of all, clone the repository (para um tutorial em português
disso, olhe [este arquivo](HelpGit.md)).

Then, set up a virtual environment:

```sh
# on linux and windows
python -m venv venv

# on linux
source venv/bin/activate

# on windows (command prompt)
venv\Scripts\activate.bat

# on windows (PowerShell)
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate.ps1
```

Install the package to the venv (needs to be done only once):

```sh
pip install -e .
```

## tools

Use `unittest` for running automatic tests (included in python):

```sh
python -m unittest discover -s tests
```

Use `black` for code formatting (`pip install black`):

```sh
black .
```

Use `pdoc` for doc generation (`pip install pdoc`):

```sh
pdoc -o doc fdth
```

Use `mypy` for type checking (`pip install mypy`):

```sh
mypy --strict --cache-fine-grained .
```
