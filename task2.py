import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Fill missing values in 'Age' with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a large number of missing values
data.drop(columns=['Cabin'], inplace=True)

# Drop rows with missing 'Fare' values
data.dropna(subset=['Fare'], inplace=True)

# Convert 'Sex' and 'Embarked' to categorical variables
data['Sex'] = data['Sex'].astype('category')
data['Embarked'] = data['Embarked'].astype('category')

# Display the cleaned dataset
data.info()

plt.figure(figsize=(10, 6))
plt.hist(data['Age'], bins=20, edgecolor='black')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

gender_counts = data['Sex'].value_counts()

plt.figure(figsize=(8, 5))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution of Titanic Passengers')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

survival_rate_by_gender = data.groupby('Sex')['Survived'].mean()

plt.figure(figsize=(8, 5))
survival_rate_by_gender.plot(kind='bar', color=['green', 'red'])
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()

plt.figure(figsize=(8, 5))
survival_rate_by_class.plot(kind='bar', color=['purple', 'orange', 'blue'])
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

