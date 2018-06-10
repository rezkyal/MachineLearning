# Kaggle Titanic: Machine Learning from Disaster :ship:
You can get the dataset and the description from [here](https://www.kaggle.com/c/titanic/data)

Goal : Predict whether someone is survived the disaster or not

Dataset used : train.csv (training), test.csv (testing)

Type : Classification

Step used in this code:
1. Data selection and preprocessing
    - Drop the unique attribute (PassenggerId,Name,Ticket,Cabin)
    - Handling categorical data('Sex','Embarked')
    - Replace missval with mean
    - Normalization
    - Split attribute and target class
2. Data transformation
    - Find outliers using isolation forest
    - Split data for test and train
3. Modelling
    - Comparing 2 algorithm:
      - Decision Tree Classification (max_depth=5)
      - KNN (n_neighbors=5)
      - SVM (kernel='linear')
4. Evaluating
    - Using Cross validation (fold=10)

## Result
![image of result](https://github.com/rezkyal/MachineLearning/image/titanic.jpg)