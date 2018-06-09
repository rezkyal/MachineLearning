## Kaggle PUBG Match Death and Statistics using Apache Spark (Pyspark)
You can get the dataset and the description from [here](https://www.kaggle.com/skihikingkevin/pubg-match-deaths)

This code using apache spark **(pyspark)** as the basic library, can be implemented to apache hadoop for big data

Goal : Predict whether someone playing in solo will get chicken dinner or not by knowing some attribute: ("game_size","player_kills","player_dbno","player_assists","player_dmg")

Dataset used : agg_match_stats_4.csv

Step used in this code:
1. Data selection and preprocessing
  - Change the datatype of each attribute to how is supposed to be
  - Select the used attribute only using sparkSQL
  - Change the target class from the ranking of the team (rank 1-100) to chicken dinner :chicken: or not (1 for chicken dinner and 0 for not)
2. Data transformation
  - Using chi-square to pick 2 most influential attribute to the classification
  - Separate data into trainingSet and testSet with random split
3. Modelling
  - Comparing 2 algorithm:
    - Decision Tree (depth=10)
    - Random Forest (depth=10)
4. Evaluating
  - Using Cross validation (binary classification evaluator, fold=3)

## Result
100 % accuracy for both algorithm