# Summarizing-Complex-Graphical-Models-of-Multiple-Chronic-Conditions

# Background:
It is important but challenging to understand the interactions of multiple chronic conditions (MCC) and how they develop over time in patients and populations. Clinical data on MCC can now be represented using graphical models to study their interaction and identify the path toward the development of MCC. However, the current graphical models representing MCC are often complex and difficult to analyze. Therefore, it is necessary to develop improved methods for generating these models.

# Objective:
This study aimed to summarize the complex graphical models of MCC interactions to improve comprehension and aid analysis.

# Methods:
We examined the emergence of 5 chronic medical conditions (ie, traumatic brain injury [TBI], posttraumatic stress disorder [PTSD], depression [Depr], substance abuse [SuAb], and back pain [BaPa]) over 5 years among 257,633 veteran patients. We developed 3 algorithms that utilize the second eigenvalue of the graph Laplacian to summarize the complex graphical models of MCC by removing less significant edges. The first algorithm learns a sparse probabilistic graphical model of MCC interactions directly from the data. The second algorithm summarizes an existing probabilistic graphical model of MCC interactions when a supporting data set is available. The third algorithm, which is a variation of the second algorithm, summarizes the existing graphical model of MCC interactions with no supporting data. Finally, we examined the coappearance of the 100 most common terms in the literature of MCC to validate the performance of the proposed model.

# Results:
The proposed summarization algorithms demonstrate considerable performance in extracting major connections among MCC without reducing the predictive accuracy of the resulting graphical models. For the model learned directly from the data, the area under the curve (AUC) performance for predicting TBI, PTSD, BaPa, SuAb, and Depr, respectively, during the next 4 years is as follows—year 2: 79.91%, 84.04%, 78.83%, 82.50%, and 81.47%; year 3: 76.23%, 80.61%, 73.51%, 79.84%, and 77.13%; year 4: 72.38%, 78.22%, 72.96%, 77.92%, and 72.65%; and year 5: 69.51%, 76.15%, 73.04%, 76.72%, and 69.99%, respectively. This demonstrates an overall 12.07% increase in the cumulative sum of AUC in comparison with the classic multilevel temporal Bayesian network.

# Conclusions:
Using graph summarization can improve the interpretability and the predictive power of the complex graphical models of MCC.
