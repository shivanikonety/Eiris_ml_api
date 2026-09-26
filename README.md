# Iris ML API

## Overview

This project implements an Iris flower species prediction system using
a Random Forest machine learning model and FastAPI.

## Technologies

- Python
- Scikit-learn
- NumPy
- Joblib
- FastAPI
- Uvicorn

## Model

Random Forest Classifier

## Dataset

Iris dataset from Scikit-learn.

## Run the Project

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000

## API Documentation

http://localhost:8000/docs

## Endpoint

POST /predict