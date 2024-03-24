# Yan Konichshev
# yk2602
# Machine Learning, Spring 2024

```bash
1) A brief statement (~paragraph) of what was done to answer the question (narratively
explaining what you did in code to answer the question, at a high level).
2) A brief statement (~paragraph) as to why this was done (why the question was
answered in this way, not by doing something else. Some kind of rationale as to why you
did x and not y or z to answer the question – why is what you did a suitable approach?).
3) A brief statement (~paragraph) as to what was found. This should be as objective and
specific as possible – just the results/facts. Do make sure to include numbers and a
figure (=a graph or plot) in your statement, to substantiate and illustrate it, respectively.
4) A brief statement (~paragraph) as to what you think the findings mean. This is your
interpretation of your findings and should answer the original question.
```

# Question 1: Build a logistic regression model. Doing so: What is the best predictor of diabetes and what is the AUC of this model? 

1. First of all, data cleaning. I have prepared the data by examining the frequency distributions and understanding the essense of the data we are working with. Surprisingly, there were almost no rows containing NaN values, so I simply dropped them and normalized all the categorical and ordinal variables (i.e. BMI, general/mental/physical health and etc.) After that was done, I have one-hot-encoded categorical (sex and zodiac sign) variables, and built a simple logistic regression model training it using the preprocessed data. Additionally, I have performed a 10 fold cross validation to tune hyperparameters achieve better generalization of the model.

<img src="question1_fig1.png" alt="Fig. 1.1 - Zodiac Sign Distribution" height="300">

2. I cleaned the data the way I did, simply because I wanted to derive all my predictors to a common scale, so that the model will see which predictors should be given more weight by itself. I have decided to perform a 10-fold CV strategy to make sure that my model would be more or less uniformly performant whenever it sees "new" data.

3. I have found that the AUC for the approach I exploited for the logistic regression model is **0.82**, which is quite good given the simple nature of the model and approach I am utilizing. In addition to that, I have found out that the most relevant predictor in my case is **General Health** with a drop of **0.01585844970860062**. Please see the graphs below for more details.

<img src="question1_fig2.png" alt="Fig. 1.2 - ROC curve for the full model" height="300">
<img src="question1_fig3.png" alt="Fig. 1.3 - Performance of different LogReg models with a single dropped predictor" height="300">

4. 

---

# Question 2: Build a SVM. Doing so: What is the best predictor of diabetes and what is the AUC of this model? 

1. ABCD
2. ABCD
3. ABCD
4. ABCD

---

# Question 3: Use a single, individual decision tree. Doing so: What is the best predictor of diabetes and what is the AUC of this model? 

1. ABCD
2. ABCD
3. ABCD
4. ABCD

---

# Question 4: Build a random forest model. Doing so: What is the best predictor of diabetes and what is the AUC of this model?

1. ABCD
2. ABCD
3. ABCD
4. ABCD

---

# Question 5: Build a model using adaBoost. Doing so: What is the best predictor of diabetes and what is the AUC of this model?

1. ABCD
2. ABCD
3. ABCD
4. ABCD

---

# Extra Credit

## A) Which of these 5 models is the best to predict diabetes in this dataset? 



---

## B) Tell us something interesting about this dataset that is not already covered by the questions above and that is not obvious.
