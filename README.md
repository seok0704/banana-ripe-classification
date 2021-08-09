# Banana Ripe Level Classification Project

## Project Overview
In this project, we propose to develop a classification model that predicts the ripeness of the banana and aim to build a machine learning pipeline using Amazon SageMaker that can be deployed on a mobile app or web application that processes the real-world, user-supplied images. Given an image of the banana, the algorithm will identify the banana's ripeness level. If the image supplied is not a banana, the model will also detect for this and output a message accordingly.

<p align="center">
</p>

## Model Training Steps
1. We collect data / images through web scraping and create labels for bananas through visual inspection. Afterwards, analysis is performed on the dataset and gather insights about the data. Also, we format our directory in the form that PyTorch, a deep learning framework, can accept and split the data into train, test, and validation
2. We gather additional data of images other than banana. This is to validate our pretrained model to make sure that the pretrained model does not predict a false positive or falsely predict banana when it is not.
3. Using the collected images of bananas and not-bananas, we try various pretrained architecture such as ResNet101 and VGG16 to determine the performance of the banana object prediction.
4. For classification of ripeness, baseline model of simple CNN with fully connected layers that achieved the accuracy of 80%.
5. For the development of the final model, we use transfer learning from pretrained models such as ResNet101 to train and develop our final model. For development of the model, we use PyTorch on SageMaker server to accelerate our training
6. For final refinement of the model, we perform hyperparameter tuning using Sage Maker and select the best performing model and its hyperparameter


## Project Deployment
The initial deployment plan was to call Amazon SageMaker model endpoint using Amazon API Gateway and AWS Lambda from a website, but hosting the SageMaker endpoint costs money hourly. Also, hosting the static version of the model through Heroku exceeds its memory quota since the ImageNets are quite big.
<p align="center">
  <img src="https://user-images.githubusercontent.com/17075250/128637110-93c1d5d3-ce87-4ddb-a4c1-537de557c538.png"/>
</p>

Instead, the trained model is pulled from S3 and was dockerized, which the image is then uploaded to Amazon Elastic Container Registry (Amazon ECR). Then we create the lambda function from the container image stored in Amazon ECR. The AWS offers free tier and only get to pay for what you use, as you use it, with no minimum fees and upfront commitments.
<p align="center">
  <img src="https://user-images.githubusercontent.com/17075250/128637533-5ff4e146-41b9-4d7f-b45a-b9094fd81968.png"/>
</p>

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
	git clone https://github.com/seok0704/banana-ripe-classification.git
	```
2. Download the [Fruit 360 dataset](https://www.kaggle.com/moltean/fruits/download). Unzip the folder and place only the 'Test' folder in the directory, 'data/'.

3. Make sure you have already installed the necessary Python packages according to the README in the program repository.
	  ```	
    pip install -r requirements.txt
	  ```	

4. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	  ```
		jupyter notebook development-notebook.ipynb
	  ```

### Tools / Libraries Used
* Jupyter Notebook
* Amazon SageMaker
* Pandas
* Numpy
* PyTorch
* Matplotlib
* Plotly
* SKLearn

