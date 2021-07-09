# Banana Ripe Level Classification Project

## Project Overview
In this project, we propose to develop a classification model that predicts the ripeness of the banana and aim to build a machine learning pipeline using Amazon SageMaker that can be deployed on a mobile app or web application that processes the real-world, user-supplied images. Given an image of the banana, the algorithm will identify the banana's ripeness level. If the image supplied is not a banana, the model will also detect for this and output a message accordingly.

![image](https://user-images.githubusercontent.com/17075250/125018919-fe5efa00-e043-11eb-84e7-cc19cd82e694.png)
![image](https://user-images.githubusercontent.com/17075250/125018930-01f28100-e044-11eb-93c3-94180da91ab7.png)
![image](https://user-images.githubusercontent.com/17075250/125018936-0454db00-e044-11eb-8726-15165b41818a.png)
![image](https://user-images.githubusercontent.com/17075250/125018942-074fcb80-e044-11eb-94df-1bd1ade2de93.png)



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

