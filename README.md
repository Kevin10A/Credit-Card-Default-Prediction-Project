# Credit-Card-Default-Prediction-Project
# Credit Card Default Prediction

### Overview
This project aims to predict whether a credit cardholder will default on their payment using machine learning models. The goal is to help financial institutions identify high-risk customers based on their credit history and other relevant factors.

### Project Structure
- `data/`: Contains the datasets used for training and testing the models.
- `notebooks/`: Jupyter notebooks with detailed data exploration, preprocessing, model training, and evaluation.
- `models/`: Contains saved models that can be used for predictions.
- `scripts/`: Python scripts for data preprocessing and model execution.
- `README.md`: Project documentation.

### Features
- Implements multiple machine learning models, including:
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbour (KNN)
- Data preparation includes converting categorical variables into numerical values and exploring the dataset's structure.

### Installation
To set up the environment for this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Credit-Card-Default-Prediction.git
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Data Preparation and Exploration:** The initial step involves loading the datasets and obtaining information such as the shape, column names, data types, and statistical summary. This helps understand the dataset's structure and content.

2. **Model Training and Evaluation:** The Jupyter notebook provides code to train models like Decision Tree, Random Forest, and KNN to predict credit card default.

3. **Prediction:** Use the trained models to make predictions on new data.

### Dataset
The datasets used in this project include:
- **creditdefault_train.csv**: Training dataset containing customer information.
- **creditdefault_test.csv**: Test dataset used to evaluate the model's performance.


### Evaluation Metrics
The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

### Results
The results section in the notebook includes an analysis of model performance, feature importance, and visualizations of the predictions made by each model.

### Future Improvements
- Hyperparameter tuning to improve the accuracy of the models.
- Implementation of more advanced machine learning algorithms like Gradient Boosting or Neural Networks.
- Exploration of additional feature engineering techniques.

### Contributing
If you would like to contribute to this project, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.

