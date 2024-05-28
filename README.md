# Titanic Survival Prediction ⛵️

This project aims to predict the survival of passengers on the Titanic based on various features such as age, sex, passenger class, and more. The dataset used for this project is from the Kaggle competition "Titanic: Machine Learning from Disaster". 💻

## Dataset 📂

The dataset consists of two files:

- `train.csv`: Contains the data for training the machine learning model, including the survival information for each passenger.
- `test.csv`: Contains the data for testing the trained model, without the survival information.

## Approach 🧭

The project follows these steps:

1. **Data Exploration**: The dataset is explored to understand the features and their distributions. Visualizations are created to gain insights into the data. 📊

2. **Feature Engineering**: New features are created from the existing ones, and some features are dropped or imputed based on their relevance and missing values. ⚙️

3. **Model Training**: A Random Forest Classifier is trained on the training dataset using scikit-learn. 🌲

4. **Model Evaluation**: The trained model is evaluated on the test set, and its accuracy and classification report are printed. 📈

5. **Prediction**: The trained model is used to make predictions on the test dataset. 🔮

## Usage 🚀

1. Install the required Python packages:
2. Run the `main.py` script, which will execute the entire pipeline:
   The script will output the model's accuracy, classification report, and make predictions on the test set.

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or additions to the project. 🎉

## License 📄

This project is licensed under the [MIT License](LICENSE).
