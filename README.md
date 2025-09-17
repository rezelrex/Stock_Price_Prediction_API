AI-Powered Financial Market Forecasting System
Project Overview
This project is an end-to-end predictive analytics pipeline that leverages a Deep Learning model to forecast financial market trends. The core of the system is a machine learning model deployed as a production-ready API, showcasing a complete MLOps (Machine Learning Operations) workflow from model training to cloud deployment.

Key Features
Predictive AI Model: A sophisticated LSTM (Long Short-Term Memory) neural network trained on historical financial time-series data.

RESTful API: A high-performance API built with FastAPI to serve on-demand predictions.

Containerization: The entire application is packaged into a Docker container, ensuring consistent and reproducible deployments across different environments.

Cloud Deployment: The Dockerized application is deployed to AWS Elastic Beanstalk, making the API publicly accessible and scalable.

Technologies Used
Category

Technology

Machine Learning

PyTorch / TensorFlow

API Framework

FastAPI

Containerization

Docker

Cloud Platform

AWS (Elastic Beanstalk)

Languages

Python

How to Use the API
The API provides a single /predict endpoint that accepts a POST request with historical stock data and returns a predicted price.

Request Body
The API expects a JSON payload with a data field containing a list of objects. Each object should represent a single data point with Open, High, Low, and Volume values.

{
  "data": [
    {
      "Open": 150.0,
      "High": 152.0,
      "Low": 149.0,
      "Volume": 1000000.0
    }
  ]
}


Example API Call
You can test the API using a command-line tool like curl. Replace the placeholder URL with your live API endpoint.

curl -X POST [http://dev-env.eba-dcngvv3y.ap-southeast-1.elasticbeanstalk.com/predict](http://dev-env.eba-dcngvv3y.ap-southeast-1.elasticbeanstalk.com/predict) \
-H "Content-Type: application/json" \
-d "{\"data\": [{\"Open\": 150.0, \"High\": 152.0, \"Low\": 149.0, \"Volume\": 1000000.0}]}"


Project Architecture
The architecture follows a standard MLOps pipeline.

Model Training: The LSTM model is trained offline.

API Integration: The trained model is loaded and served by a FastAPI application.

Dockerization: The FastAPI app and its dependencies are containerized.

Deployment: The Docker image is deployed to AWS, where it runs as a live web service.

This setup ensures a seamless transition from a theoretical model to a practical, scalable application.

Local Setup
To run this project locally, you will need to have Docker installed.

Clone the repository:

git clone [https://github.com/rezelrex/Stock-Prediction-API.git](https://github.com/rezelrex/Stock-Prediction-API.git)
cd Stock-Prediction-API


Build the Docker image:

docker build -t stock-prediction-api .


Run the container:

docker run -p 8000:8000 stock-prediction-api
