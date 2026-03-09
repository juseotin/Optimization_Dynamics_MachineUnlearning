# Autonomous Machine Unlearning via Projected Gradient Ascent

**Securing Financial Privacy in Convex Optimization Models**

[Project Website](https://juseotin.github.io/Optimization_Dynamics_MachineUnlearning) 

This repository contains the code, data, and findings for my UC San Diego HDSI Data Science Capstone project. The research investigates whether Projected Gradient Ascent can serve as a mathematically verifiable proxy for exact retraining in convex environments, specifically utilizing Logistic Regression on the Statlog German Credit Dataset.

## Key Discoveries

* **The Geometric Audit:** We isolated the unlearning dynamics from stochastic noise. While single user unlearning achieves surgical mathematical precision, production scaling reveals a Batch Interference Effect that heavily dilutes individual precision.
* **The Stopping Point Paradox:** Ascending the gradient indefinitely destroys general model utility. We identified a 34 step Safety Window and developed an autonomous stopping criterion using the validation loss on the retain set.
* **The Privacy Deadlock:** Membership Inference Attacks exposed a critical vulnerability. Highly confident outliers become trapped on the flat plateau of the sigmoid curve, preventing true amnesia without destroying the global model. 

## Repository Structure

* `code/`: Contains the Jupyter Notebooks and Python scripts for the Projected Gradient Ascent implementation, Geometric Audit, and Membership Inference Attack evaluations.
* `figure/`: Visualizations generated during the research, including the geometric audit scatterplots, the stopping paradox line graphs, and the MIA confidence histograms.
* `poster.tex`: The LaTeX source code for the final capstone showcase poster.
* `main.tex`: The LaTeX source code for the full academic research report.
* `index.html`: The source code for the interactive project website.

## Reproducing the Results

To run the unlearning algorithms locally:

1. Clone this repository to your local machine.
2. Install the required dependencies (NumPy, scikit learn, matplotlib, pandas).
3. Run the primary unlearning notebook located in the `code/` directory to reproduce the Geometric Audit and MIA evaluations.

## Author

**Justin Minseob Seo** UC San Diego, Data Science (2026)  
Mentor: Professor Jun Kun Wang
