### Homework #5: Unsupervised learning: Wines. Machine Learning, Spring 2024

### Yan Konichshev; yk2602

---

### Question 1: Do a PCA on the data. How many Eigenvalues are above 1? Plotting the 2D solution (projecting the data on the first 2 principal components), how much of the variance is explained by these two dimensions, and how would you interpret them?

1. 
2. 
3. 
<figure>
  <img src="pics/question1_fig1.png" alt="Fig. 1.1 - Confusion matrix." height="300">
  <figcaption>Fig. 1.1 - Confusion matrix.</figcaption>
</figure>
<figure>
  <img src="pics/question1_fig2.png" alt="Fig. 1.2 - ROC curve for Perceptron." height="300">
  <figcaption>Fig. 1.2 - ROC curve for Perceptron.</figcaption>
</figure>

4. 

---

### Question 2: Use t-SNE on the data. How does KL-divergence depend on Perplexity (vary Perplexity from 5 to 150)? Make sure to plot this relationship. Also, show a plot of the 2D component with a Perplexity of 20.

1. 
2. 
3. 

<figure>
  <img src="pics/question2_fig1.png" alt="Fig. 2.1 - Heat map of different models and AUC scores" height="300">
  <figcaption>Fig. 2.1 - Heat map of different models and AUC scores.</figcaption>
</figure>
<figure>
  <img src="pics/question2_fig2.png" alt="Fig. 2.2 - Heat map of different models and RMSE scores" height="300">
  <figcaption>Fig. 2.2 - Heat map of different models and RMSE scores.</figcaption>
</figure>
   
4. 

---

### Question 3: Use MDS on the data. Try a 2-dimensional embedding. What is the resulting stress of this embedding? Also, plot this solution and comment on how it compares to t-SNE.

1. 
2. 
3. 
<figure>
  <img src="pics/question3_fig1.png" alt="Fig. 3.1 - Heat map of different models and AUC scores" height="300">
  <figcaption>Fig. 3.1 - Heat map of different models and AUC scores.</figcaption>
</figure>
<figure>
  <img src="pics/question3_fig2.png" alt="Fig. 3.2 - Heat map of different models and RMSE scores" height="300">
  <figcaption>Fig. 3.2 - Heat map of different models and RMSE scores.</figcaption>
</figure>

4. 

---

### Question 4: Building on one of the dimensionality reduction methods above that yielded a 2D solution (1-3, your choice), use the Silhouette method to determine the optimal number of clusters and then use kMeans with that number (k) to produce a plot that represents each wine as a dot in a 2D space in the color of its cluster. What is the total sum of the distance of all points to their respective clusters centers, of this solution?

1. 
2. 
3. 

<figure>
  <img src="pics/question4_fig1.png" alt="Fig. 4.1 - Heat map of different models and RMSE scores" height="300">
  <figcaption>Fig. 4.1 - Heat map of different models and RMSE scores.</figcaption>
</figure>

4. 

---

### Question 5: Building on one of the dimensionality reduction methods above that yielded a 2D solution (1-3, your choice), use dBScan to produce a plot that represents each wine as a dot in a 2D space in the color of its cluster. Make sure to suitably pick the radius of the perimeter (“epsilon”) and the minimal number of points within the perimeter to form a cluster (“minPoints”) and comment on your choice of these two hyperparameters.

1. 
2. 
3. 

<figure>
  <img src="pics/question5_fig1.png" alt="Fig. 5.1 - RMSE vs. epochs function" height="300">
  <figcaption>Fig. 5.1 - RMSE vs. epochs function.</figcaption>
</figure>

4. 

---

### Extra Credit

#### A) Given your answers to all of these questions taken together, how many different kinds of wine do you think there are and how do they differ?

---

#### B) Is there anything of interest you learned about wines from exploring this dataset with unsupervised machine learning method that is worth noting and not already covered in the questions above?
