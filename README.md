The following is the link for the dataset used
https://www.kaggle.com/datasets/samuelcortinhas/time-series-practice-dataset?resource=download

This dataset consists of date, store, product and number_sold indicating sales for different stores and products. Here our task is now to define a threshold so we can define an incident.
Lets take the threshold to be 1000 for number_sold column, and if the sales exceed 1000, we label 'incident' column as 1. Now task becomes to identify conditions that lead to an incident(sales>1000).

For the sliding window technique we will use the last 5 values and see the next 3 values if there are any incidents. For example, for 20 march, we will take data from 15-20 march and also see 21-23 march if there is an incident. The model learns from training data this way.  
effectively:
Past values used = 5 (denoted as P in our code)
Future values used = 3 (denoted as F in our code)

After formulating the sliding window sizes for the training data, we have to pick a model. LogisticRegression model is the most appropriate model to use here. This is because of a number of reasons:
1) relatively fast training time -> crucial since we have over 230K rows of training data.
2) Logistic regression works well because the sliding window converts time-series data into fixed numerical features.

The results can be observed as follows:
<img width="756" height="200" alt="image" src="https://github.com/user-attachments/assets/0502ad8a-6659-4e3b-b1af-2e0cde4ed0f6" />
**Accuracy** -> 97.2% -> This is very high showing that the model works well
**Precision** -> 79.81% -> Almost 80% of the predicted incidents were correct and 20% were false alarms. This is an area for improvement.
**Recall** -> 98.65% -> This shows our model detected almost all real incidents. Very few incidents were missed here

For incident detection often Recall is given more priority as missing an incident is not desirable. 
Some of the limitations I observed:
1) Logistic Regression cannot capture long temporal dependencies
2) Precision is 79.81% which is okay but can be improved
3) Incident definition maybe too simplistic. 
