# DIVE-Doc's Web Demo platform.
[![CI Pipeline](https://github.com/JayRay5/DIVE_Doc_plateform/actions/workflows/push.yml/badge.svg)](https://github.com/JayRay5/DIVE_Doc_plateform/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
###### TD;LR
This project is the open-source code of the [demo web platform]() for the model [DIVE-Doc](https://github.com/JayRay5/DIVE-Doc) presented at VisionDocs @ICCV2025. 

---

## 🛠️ MLOps & Quality Assurance

Following **MLOps best practices**, this repository ensures code robustness through a strict CI/CD pipeline:

* **⚡ Ruff:** Fast linting and formatting.
* **🛡️ Bandit:** Static security analysis to prevent vulnerabilities.
* **🧪 Pytest:** Unit tests and Smoke tests (API & Frontend health checks).
* **🤖 Dependabot:** Weekly dependency vulnerability monitoring.

## Installation & Setup
1- Install dependencies
```bash
conda create --name divedoc-platform-env python=3.12.12
conda activate divedoc-platform-env
pip install -r requirements.txt
```

2- Install the git hook <br>
I added a git hook that is executed during each commit and before each push locally and run: <br>
 - bandit: to check security issues
 - ruff: to check quality and format
 - pytest: to check main functions' sanity
```bash
pre-commit install
pre-commit install --hook-type pre-push
```

3- Run the server locally
```bash
#terminal 1
python app.py
```

```bash
#terminal 2
uvicorn src.main:app --host 127.0.0.1 --port 8000
```
