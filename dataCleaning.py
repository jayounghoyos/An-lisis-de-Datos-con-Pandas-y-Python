#Authors: Martín Valencia, María Acevedo, Juan Andrés Young 

import pandas as pd 

df = pd.read_csv("Titanic-Dataset.csv")


'''
This code performs essential data preprocessing steps on the Titanic dataset to prepare it for building an AI model that predicts passenger survival.
 
1. Loading the dataset:
    - The dataset is loaded from a CSV file into a pandas DataFrame, which allows for easy data manipulation and analysis.

2. Dropping irrelevant or redundant columns:
    - 'PassengerId': A unique identifier for each passenger, which does not provide predictive information.
    - 'Ticket': Ticket numbers are mostly arbitrary and unlikely to help the model learn survival patterns.
    - 'Name': Names are unique and textual; without further processing (like extracting titles), they do not directly contribute to prediction.
    - 'Cabin': This column has many missing values and inconsistent data, making it unreliable without complex imputation.
    - 'Embarked': Although it could have some predictive value, it is dropped here, possibly to simplify the model or due to missing values.
    - 'Fare': Fare might correlate with survival but is excluded here, likely to focus on other features or due to data quality considerations.
    Dropping these columns reduces noise and dimensionality, helping the model focus on more relevant features.

3. Handling missing values in the 'Age' column:
    - The 'Age' feature is important because age can influence survival chances (e.g., children might have been prioritized).
    - Missing age values are filled with the mean age of all passengers, a common and simple imputation technique.
    - This ensures no missing values remain, which is necessary because many machine learning algorithms cannot handle missing data.

4. Printing the cleaned DataFrame:
    - This step allows verification that the data has been cleaned correctly, with the intended columns removed and missing values handled.

Overall, these preprocessing steps clean and simplify the dataset, making it suitable for training machine learning models to predict which types of passengers were more likely to survive the Titanic disaster.


'''

df = df.drop("Ticket", axis = 1)
df = df.drop("PassengerId", axis = 1)
df = df.drop("Embarked", axis = 1)
df = df.drop("Fare", axis = 1)
df = df.drop("Name", axis = 1)
df = df.drop("Cabin", axis = 1)

df = df.fillna({"Age" : df["Age"].mean()}, inplace = True)

print(df)


