# Documentation for deploying the application

## Introduction

The current repository set up allows us to containerize the application using docker and deploy it to a service of our choice - I chose google cloud. This document will outline the steps required to deploy the application to google cloud.

## Prerequisites
- Docker
- Google Cloud SDK
- Docker CLI
- Google Cloud CLI

## Steps
- Log into Docker using CLI
- Run `docker-compose build` to build the docker image
- Run `docker-compose push --env-file .env` to push the image to docker hub
- Log into Google Cloud using CLI
- Run `gcloud builds submit --tag gcr.io/smart-theory-404203/graft` to build the image on google cloud

## Deploying to Google Cloud
- Run `gcloud run deploy --image gcr.io/smart-theory-404203/graft --platform managed` to deploy the application to google cloud
- Run `gcloud run services list` to get the URL of the deployed application
- Navigate to the URL to view the application

## Future Improvements
Instead of stroing `.npz` files in the repository, we can store them in a cloud storage bucket and load them from there:
- Put `creds.json` file in `root` directory. 
- Set `GCS` to `True` in `config.py` to enable this.