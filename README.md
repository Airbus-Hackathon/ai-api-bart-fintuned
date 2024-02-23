# AI API BART FINETUNE

This repository contains the code for the FastAPI microservice allowing to interact with IA model and summarize content.

The inference uses CPU due to costs constraints.

- Inference average time: 15 s
- RAM: 7 GB
- CPU: 1100 mCPU

A deployment on a GPU enhanced machine would improve inference time.

[API documentation](https://documenter.getpostman.com/view/13953520/2sA2rCTgxc)
[Model checkpoint file](https://drive.google.com/file/d/16deBHJvsHPKI1yPaQmD2VBFycbp-TXGp/view?usp=sharing)

## Build docker

```bash
docker build -f ./Dockerfile . --tag=antoineleguillou/legal-summarizer:vX.X.X
```

## Get image from dockerhub

```bash
docker pull antoineleguillou/legal-summarizer:v1.2.0
```

## Download the model checkpoint file

[download the output.ckpt file](https://drive.usercontent.google.com/download?id=16deBHJvsHPKI1yPaQmD2VBFycbp-TXGp&export=download)

Please rename the file to: output.ckpt and place it into ./models directory

## Start in development

```bash
# create a models directory
mkdir -p models
# install dependencies
pip install -r requirements.txt
# start fast api server on port 8000
python apimodel.py
```

## Evaluate resources consumption (UNIX system)

```bash
# get fast api process pid
lsof -i tcp:8000
# monitor resources usage
top -pid <PID>
```
