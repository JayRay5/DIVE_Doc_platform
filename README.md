# DIVE-Doc's Web Demo platform.
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
[![CI/CD Pipeline](https://github.com/JayRay5/DIVE_Doc_plateform/actions/workflows/main_pipeline.yml/badge.svg)](https://github.com/JayRay5/DIVE_Doc_plateform/actions/workflows/main_pipeline.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Rayane/DIVE-Doc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
###### TL;DR
This project contains the open-source code for the [demo web platform](https://huggingface.co/spaces/JayRay5/DIVE-Doc-docvqa), which utilizes the [DIVE-Doc](https://github.com/JayRay5/DIVE-Doc) model presented at VisionDocs @ICCV2025. 

---

## 🛠️ MLOps & Quality Assurance

Following **MLOps best practices**, this repository ensures code robustness through a strict CI/CD pipeline:

* **⚡ Ruff:** Fast linting and formatting.
* **🛡️ Bandit:** Static security analysis to prevent vulnerabilities.
* **🧪 Pytest:** Unit tests and Smoke tests (API & Frontend health checks).
* **🤖 Dependabot:** Weekly dependency vulnerability monitoring.
* **🐳 Docker:** Fully containerized application ensuring reproducibility across environments.
* **🚀 GitHub Actions:** Automated deployment pipeline pushing the image to Hugging Face Spaces.

## Installation & Setup
1- Install dependencies
```bash
conda create --name divedoc-platform-env python=3.12.12
conda activate divedoc-platform-env
pip install -r requirements.txt
```
You will need a HuggingFace token in order to use processors and models of this repository. Please, go to your [HuggingFace](https://huggingface.co/settings/tokens) account and create a token that gives you the right to use [PaliGEMMA](https://huggingface.co/google/paligemma-3b-ft-docvqa-896) and [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) processors.<br>
Then, add this token into your divedoc-platform-env virtual environment by running the following command:
```bash
conda env config vars set HF_TOKEN="your_token"
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
