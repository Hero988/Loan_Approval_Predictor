import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
import os
import shutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ConditionalImputer(TransformerMixin):
    """
    A custom transformer that conditionally applies imputation to a dataset.

    Parameters:
    - strategy (str): The strategy for imputation ('mean', 'median', etc.).

    Attributes:
    - imputer (SimpleImputer or None): The imputer instance if required.
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        """Fit the imputer to the data if there are missing values."""
        if X.isnull().sum().any():
            self.imputer = SimpleImputer(strategy=self.strategy)
            self.imputer.fit(X)
        return self

    def transform(self, X):
        """Apply the imputation to the data if an imputer has been fitted."""
        if self.imputer is not None:
            return self.imputer.transform(X)
        return X

# Class to save feature names and now also the target column name
class FeatureNameSaver(BaseEstimator, TransformerMixin):
    """
    A transformer for saving feature names and optionally the target column name. 
    This class is useful for retaining feature names after transformations 
    that might lose these metadata (e.g., during scaling or encoding).

    Parameters:
    - target_column (str, optional): The name of the target column. 
      This is useful for distinguishing the target from the features in a dataset.

    Attributes:
    - feature_names (list): List of feature names.
    - target_column (str): Name of the target column if provided.
    """

    def __init__(self, target_column=None):
        self.feature_names = None
        self.target_column = target_column

    def fit(self, X, y=None):
        self.feature_names = X.columns.tolist()
        return self

    def transform(self, X):
        return X

    def get_feature_names(self):
        return self.feature_names

    def get_target_column(self):
        return self.target_column

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Dynamically preprocess the data by detecting binary categorical columns and converting 
    them to numeric binary values (0 and 1), and return the mappings.

    This function identifies any columns with exactly two unique categories and maps them 
    to 0 and 1. It returns both the preprocessed data and the mappings used.

    Parameters:
    - data (DataFrame): The input data with potential categorical text labels.

    Returns:
    - tuple: A tuple containing the preprocessed DataFrame and a dictionary of mappings for each binary column.
    """
    data = data.copy()
    mappings = {}  # Dictionary to store mappings for each column

    for column in data.columns:
        if len(data[column].unique()) == 2:
            unique_values = sorted(data[column].unique())
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            reverse_mapping = {0: unique_values[0], 1: unique_values[1]}  # Reverse mapping
            data[column] = data[column].map(mapping)
            mappings[column] = reverse_mapping  # Store reverse mapping
            print(f"Column '{column}' mapped with {mapping}")

    return data, mappings

def setup_model_pipeline(numerical_cols, target_column):
    """
    Setup the model pipeline with preprocessing, feature name saving, and classifier.

    Parameters:
    - numerical_cols (list): List of column names to include in preprocessing.

    Returns:
    - Pipeline: The complete preprocessing and classification pipeline.
    """
    # FeatureNameSaver is part of the pipeline for numerical columns
    numerical_transformer = Pipeline(steps=[
        ('feature_name_saver', FeatureNameSaver(target_column=target_column)),
        ('conditional_imputer', ConditionalImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols)
    ], remainder='passthrough')

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

def split_and_save_data(data, file_path_train_test, file_path_unseen):
    """
    Split the data into two halves and save each half to a CSV file.

    Parameters:
    - data (DataFrame): The complete dataset.
    - file_path_train_test (str): File path where the first half of the data will be saved.
    - file_path_unseen (str): File path where the second half of the data will be saved.

    Returns:
    - None
    """
    # Determine the index at which to split the data
    split_index = int(len(data) * 0.5)
    
    # Split the data into two halves
    train_test_data = data.iloc[:split_index]
    unseen_data = data.iloc[split_index:]

    # Save each dataset to a CSV file
    train_test_data.to_csv(file_path_train_test, index=False)
    unseen_data.to_csv(file_path_unseen, index=False)

def split_data_train_and_test(data, target_column):
    """
    Split data into training and testing datasets.

    Parameters:
    - data (DataFrame): The complete dataset.

    Returns:
    - tuple: Contains the training and testing datasets (X_train, X_test, y_train, y_test).
    """

    # Drop 'loan_status' and separate features and target
    X = data.drop([f'{target_column}'], axis=1)
    y = data[f'{target_column}']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    return X_train, X_test, y_train, y_test

def save_predictions(y_test, y_pred, file_path):
    """
    Save the actual and predicted labels to a CSV file.

    Parameters:
    - y_test (Series/DataFrame): The actual labels from the test set.
    - y_pred (list/array): The predicted labels from the model.
    - file_path (str): The path where the CSV file will be saved.

    Returns:
    - None
    """
    # Ensure y_pred is a Series with the same index as y_test
    y_pred_series = pd.Series(y_pred, index=y_test.index, name='Predicted')

    # Combine the actual and predicted labels into a single DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_series
    })

    # Save to CSV
    results_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred, classes, accuracy):
    """
    This function prints and plots the confusion matrix.

    Parameters:
    - y_true: array-like, Ground truth (correct) target values.
    - y_pred: array-like, Estimated targets as returned by a classifier.
    - classes: list, An array of labels that appear in the data.
    """
    title = f'Confusion matrix, accuracy of the model: {accuracy:.2f}%'
    cmap = plt.cm.Blues

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Determine the set of labels that appear in y_true or y_pred
    labels = set(y_true).union(set(y_pred))
    
    # Filter classes to include only those that appear in labels
    filtered_classes = [cls for cls in classes if cls in labels]

    # Using seaborn to make the heatmap visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=filtered_classes, yticklabels=filtered_classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

def plot_feature_importance(importances, feature_names):
    """
    Plots the importance of each feature in a bar chart format.

    Parameters:
    - importances (array): The feature importances from the model.
    - feature_names (list): List of names corresponding to the features.
    """
    title = 'Feature Importance'

    # Create array indices sorted by importance of the feature
    indices = np.argsort(importances)

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]

    # Create plot with improved layout and larger figure size
    plt.figure(figsize=(12, len(indices) * 0.4))  # Dynamically scale figure height to number of features
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), names, fontsize=10)  # Adjust fontsize for better readability if necessary
    plt.xlabel('Relative Importance')
    plt.tight_layout()  # Adjust layout to make room for label names
    plt.savefig('feature_importance.png')  # Save as PNG to preserve detail

def user_choose_dataset():
    """
    Finds all files in the current directory that contain '_dataset' in their names,
    Displays a list of dataset files and allows the user to choose one.

    Returns:
    - str: The full path to the chosen dataset file.
    """
    directory = os.getcwd()  # Get the current working directory
    dataset_files = []
    for filename in os.listdir(directory):
        if "_dataset" in filename:  # Change to check for '_dataset' anywhere in the filename:
            full_path = os.path.join(directory, filename)
            dataset_files.append(full_path)

    if not dataset_files:
        print("No dataset files found.")
        return None

    print("Please choose a dataset by entering the corresponding number:")
    for index, dataset in enumerate(dataset_files):
        print(f"{index + 1}. {os.path.basename(dataset)}")  # Show only the filename for simplicity

    while True:
        try:
            choice = int(input("Enter your choice: ")) - 1
            if 0 <= choice < len(dataset_files):
                return dataset_files[choice]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(dataset_files)}.")
        except ValueError:
            print("Please enter a valid integer.")

def user_choose_target_column(data):
    """
    Displays binary columns (columns with exactly two unique values) of the DataFrame 
    and lets the user choose one to be the target column.

    Parameters:
    - data (DataFrame): The DataFrame from which the user will choose the target column.

    Returns:
    - str: The name of the column chosen by the user as the target.
    """
    # Identify binary columns
    binary_columns = [col for col in data.columns if data[col].nunique() == 2]

    # Check if there are any binary columns
    if not binary_columns:
        print("No binary columns found.")
        return None

    # Display the binary columns with an index number
    print("Please choose a target column from the list below:")
    for index, column in enumerate(binary_columns):
        print(f"{index + 1}. {column}")

    # Get user input for the column choice
    while True:
        try:
            choice = int(input("Enter the number corresponding to the target column: ")) - 1
            if 0 <= choice < len(binary_columns):
                return binary_columns[choice]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(binary_columns)}.")
        except ValueError:
            print("Please enter a valid number.")

def convert_numeric_to_categorical(data, mappings):
    """
    Converts numeric values back to categorical labels based on provided mappings.
    
    Parameters:
    - data (DataFrame): The DataFrame containing numeric values to be converted.
    - mappings (dict): A dictionary containing mappings for each column. Each key should be the column name,
      and the value should be another dictionary mapping numeric values back to categorical labels.
      
    Returns:
    - DataFrame: The DataFrame with numeric values converted back to categorical labels.
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    converted_data = data.copy()

    # Iterate over each mapping item
    for column, mapping_dict in mappings.items():
        if column in converted_data.columns:
            # Apply the reverse mapping to the column if it exists in the DataFrame
            converted_data[column] = converted_data[column].map(mapping_dict).fillna(converted_data[column])

    return converted_data

def train_and_evaluate_model():
    """
    Main function to run the preprocessing, model training, evaluation, and model saving workflow.
    """
    # Let the user choose the dataset based on the uploaded datasets
    data_main = user_choose_dataset()

    # Get the filename of that dataset chosen
    data_main_filename = os.path.basename(data_main)

    # Load and preprocess the dataset
    main_data = load_data(f'{data_main}')

    # Let the user choose the coloumn that the model will learn
    target_column = user_choose_target_column(main_data)

    # Preprocess the data in a way the model will understand
    main_data_preprocessed, mappings = preprocess_data(main_data)

    # Save split data
    split_and_save_data(main_data_preprocessed, 'train_test_data.csv', 'unseen_data.csv')

    # Load training and testing data
    train_test_data = load_data('train_test_data.csv')
    # First, drop the target column from the DataFrame
    data_without_target = train_test_data.drop([target_column], axis=1)

    # Then, select only the numerical columns from the DataFrame without the target column
    numerical_cols = data_without_target.select_dtypes(include=['number']).columns.tolist()

    # Create a copy of the data that includes only the numerical columns and the target column
    train_test_data_copy = train_test_data[numerical_cols + [target_column]]

    # Setup model pipeline
    model_pipeline = setup_model_pipeline(numerical_cols, target_column)

    # Split and train
    X_train, X_test, y_train, y_test = split_data_train_and_test(train_test_data_copy, target_column)
    model_pipeline.fit(X_train, y_train)

    # Predictions and Target convert back to labels
    y_pred = model_pipeline.predict(X_test)
    target_column_mapping = mappings.get(target_column, {})
    y_pred_original = [target_column_mapping[pred] for pred in y_pred]
    y_test_original = y_test.map(target_column_mapping)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy of the model: {accuracy:.2f}%")

    save_predictions(y_test_original, y_pred_original, 'enhanced_test_data.csv')

    # Visualize the confusion matrix
    class_names = np.array(list(target_column_mapping.values()))
    plot_confusion_matrix(y_test, y_pred, class_names, accuracy)

    # Extracting the RandomForestClassifier from the pipeline
    classifier = model_pipeline.named_steps['classifier']

    # Feature importance from the RandomForest model
    feature_importances = classifier.feature_importances_

    # Visualize feature importance
    plot_feature_importance(feature_importances, numerical_cols)

    # Save model with mappings
    with open('model_pipeline.pkl', 'wb') as f:
        joblib.dump((model_pipeline, mappings), f)

    # Save the scaler
    scaler = model_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler']
    joblib.dump(scaler, 'scaler.pkl')

    save_directory = f'model_random_forest_{accuracy:.2f}%_{data_main_filename}'

    unseen_data = load_data('unseen_data.csv')
    # Copy the dataset to avoid modifying the original data
    data_copy = unseen_data.copy()
    data_no_target = data_copy.drop(columns=[target_column])
    data_no_target.to_csv('unseen_data_no_target.csv', index=False)

    unseen_data_original = convert_numeric_to_categorical(unseen_data, mappings)
    unseen_data_original.to_csv('unseen_data_target.csv', index=False)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save all files except the specified ones
    exclude_files = ['Loan_Approval_Predictor.py', f'{data_main_filename}', 'LoanApprovalPrediction_dataset.csv', 'loan_approval_dataset.csv']

    # Copy the file and move it to the save_directory
    shutil.copy(f'{data_main_filename}', save_directory)

    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

def choose_directory_to_predict():
    """
    Lists all subdirectories in the current working directory except '.git' and allows the user to choose one.

    Returns:
    - str: The path to the chosen directory, or None if no valid selection is made.
    """
    current_path = os.getcwd()  # Get the current working directory
    # Filter directories, excluding '.git'
    directories = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d)) and d != '.git']

    if not directories:
        print("No directories found in the current directory.")
        return None

    print("Please choose a directory by entering the corresponding number:")
    for index, directory in enumerate(directories):
        print(f"{index + 1}. {directory}")

    while True:
        try:
            choice = int(input("Enter your choice: ")) - 1
            if 0 <= choice < len(directories):
                return os.path.join(current_path, directories[choice])  # Return the full path of the chosen directory
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(directories)}.")
        except ValueError:
            print("Please enter a valid integer.")

def get_user_input(feature_names, mappings):
    user_data = {}
    for feature in feature_names:
        # Ask for input
        user_input = input(f"Enter value for {feature}: ")

        # Check and apply mapping if necessary
        if feature in mappings and user_input in mappings[feature]:
            user_data[feature] = mappings[feature][user_input]
        else:
            try:
                # Attempt to convert numerical inputs directly
                user_data[feature] = float(user_input)
            except ValueError:
                user_data[feature] = user_input  # Keep as string if conversion fails

    return user_data

def load_model_and_get_columns(file_path):
    try:
        # Load the model pipeline and mappings from the file
        with open(file_path, 'rb') as f:
            model_pipeline, mappings = joblib.load(f)
        
        # Access the 'preprocessor' which is a ColumnTransformer in this case
        if 'preprocessor' in model_pipeline.named_steps:
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            # Find the FeatureNameSaver within the transformers of the ColumnTransformer
            for name, transformer, columns in preprocessor.transformers_:
                if 'feature_name_saver' in transformer.named_steps:
                    feature_name_saver = transformer.named_steps['feature_name_saver']
                    feature_names = feature_name_saver.get_feature_names()
                    target_column = feature_name_saver.get_target_column()
                    return model_pipeline, feature_names, mappings, target_column
        else:
            print("No 'preprocessor' found in the pipeline.")
    except Exception as e:
        print(f"Failed to load or process the pipeline: {e}")

    return None, None, None, None

def use_trained_model_to_predict(choice):
    """
    Uses a pre-trained model to make predictions based on user input or a dataset. 
    The model and mappings are loaded from a specified directory, and predictions are output.

    Parameters:
    - choice (str): Determines the mode of prediction. '2' for individual prediction 
      from user input and '3' for batch predictions from a dataset.

    Side Effects:
    - Prints the prediction result directly to console if choice is '2'.
    - Saves the predictions to a CSV file if choice is '3'.
    """
    chosen_directory = choose_directory_to_predict()

    model_path = os.path.join(chosen_directory, 'model_pipeline.pkl')

    model, feature_names, mappings, target_column = load_model_and_get_columns(model_path)

    if choice == '2':
        user_data = get_user_input(feature_names, mappings)

        # Convert user data to DataFrame for prediction
        user_data_df = pd.DataFrame([user_data])

        # Make prediction
        prediction = model.predict(user_data_df)
        target_column_mapping = mappings.get(target_column, {})
        # Applying the mapping to each prediction and directly accessing the first item
        prediction_original = [target_column_mapping[pred] for pred in prediction][0]  # Access the first item
        print("Predicted Output:", prediction_original)
    elif choice == '3':
        data_path = os.path.join(chosen_directory, 'unseen_data_no_target.csv')
        # Load the dataset
        data = pd.read_csv(data_path)

        predictions = model.predict(data)
        target_column_mapping = mappings.get(target_column, {})
        predictions_original = [target_column_mapping[pred] for pred in predictions]

        data['Predictions'] = predictions_original

        data_with_predictions = convert_numeric_to_categorical(data, mappings)

        # Combine the directory path and filename
        file_path = os.path.join(chosen_directory, 'unseen_data_with_predictions.csv')

        # Save the DataFrame to a CSV file
        data_with_predictions.to_csv(file_path, index=False)

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Train and Evaluate Model")
        print("2 - Use Trained Model for Individual Prediction")
        print("3 - Use Trained Model for Multiple Predictions")
        print("4 - Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            train_and_evaluate_model()
        elif choice == '2':
            use_trained_model_to_predict(choice)
        elif choice == '3':
            use_trained_model_to_predict(choice)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    main_menu()
