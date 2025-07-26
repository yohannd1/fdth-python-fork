# fdth

A python port of the [fdth](https://github.com/jcfaria/fdth) R library.

## examples

Examples can be found at the `examples/python` folder.

## (portuguese) o que falta a fazer

Este pacote foi desenvolvido como projeto para a disciplina
"Probabilidade e Estatística" (Ciência da Computação, UESC), no semestre
2025.1.

O projeto já está quase pronto, mas tem alguns problemas que ainda
precisam ser consertados. Caso seja desejado, alguém pode pegar isso
como projeto de novo - aqui um sumário:

- a representação visual dos plots numéricos não está perfeita;

- o funcionamento da classe MultipleFDT não está correspondente ao R
(foi erro da gente ao transcrever o código antigo - usar o [código do
semestre anterior](https://github.com/yuriccosta/fdth-python) pode
ajudar a entender direito como deveria ser);

  - em especial, levar em conta o argumento `by` que não foi levado em
  conta;

- passar resto da documentação (e comentários) para inglês;

- houve uma regressão em `Binning.from_sturges`, e o resultado difere do
original;

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

## credits

[Original version](https://github.com/jcfaria/fdth) created by [José
Cláudio Faria](https://github.com/jcfaria), [Ivan Bezerra
Allaman](https://github.com/ivanalaman) and [Jakson Alves de
Aquino](https://github.com/jalvesaq).

[Initial python port](https://github.com/yuriccosta/fdth-python) by
[Emyle Silva](https://github.com/EmyleSilva), [Lucas Gabriel
Ferreira](https://github.com/lgferreiracic), [Yuri Coutinho
Costa](https://github.com/yuriccosta), and [Maria
Clara](https://github.com/MaryClaraSimoes).

Current version made by Gabriel Galdino, Luciene Mª Torquato C. Batista,
Stella Ribas, Thainá Guimarães and Yohanan Santana.
