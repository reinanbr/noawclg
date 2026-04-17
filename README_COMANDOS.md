# README de Comandos Utilizados

Este arquivo resume os comandos usados no fluxo recente de trabalho do projeto.

## 1. Preparacao do ambiente (Pipenv)

Instalar dependencias e garantir ambiente pronto:

```bash
pipenv install
pipenv run python -m pip install --upgrade pip
pipenv run python -m pip install -e .
```

Instalar ferramentas do CI local:

```bash
pipenv run python -m pip install ruff mypy types-requests pytest pytest-cov numpy xarray requests urllib3 netCDF4 cfgrib eccodes build twine
```

## 2. Qualidade de codigo (lint e tipagem)

Lint:

```bash
pipenv run ruff check noawclg tests setup.py
```

Formatacao (apenas verificacao):

```bash
pipenv run ruff format --check noawclg tests setup.py
```

Type-check:

```bash
pipenv run mypy noawclg/gfs_dataset.py --ignore-missing-imports
```

## 3. Testes

Rodar somente o novo arquivo de testes da `main`:

```bash
pipenv run pytest tests/test_main.py -q
```

Rodar testes offline (como no CI):

```bash
pipenv run pytest tests/ -v -m "not integration" --tb=short --cov=noawclg --cov-report=term-missing --cov-report=xml --junitxml=junit/test-results-local.xml
```

Rodar testes de integracao (rede):

```bash
pipenv run pytest tests/ -v -m integration --tb=short
```

## 4. Build e validacao de distribuicao

Build dos artefatos:

```bash
pipenv run python -m build
```

Validacao dos pacotes gerados:

```bash
pipenv run twine check dist/*
```

## 5. Git (commit, tag e push)

Ver estado do repositorio:

```bash
git status --short
git branch --show-current
git remote -v
```

Adicionar arquivos de release:

```bash
git add CHANGELOG.md README.md WORKLOG_2.2.md tests/test_main.py noawclg/main.py setup.py
```

Commit:

```bash
git commit -m "release: 2.2 docs and tests for get_noaa_data"
```

Criar tag anotada:

```bash
git tag -a 2.2 -m "Release 2.2"
```

Push da branch e tag:

```bash
git push origin main
git push origin 2.2
```

## 6. Comando unico para validar pipeline local

```bash
set -e
pipenv run ruff check noawclg tests setup.py
pipenv run ruff format --check noawclg tests setup.py
pipenv run mypy noawclg/gfs_dataset.py --ignore-missing-imports
pipenv run pytest tests/ -v -m "not integration" --tb=short --cov=noawclg --cov-report=term-missing --cov-report=xml --junitxml=junit/test-results-local.xml
pipenv run pytest tests/ -v -m integration --tb=short || ([ $? -eq 5 ] && echo "[SKIP] no integration tests")
pipenv run python -m build
pipenv run twine check dist/*
```
