# Optimization Dynamics in Machine Unlearning via Projected Gradient Ascent

**Author:** Justin (Minseob) Seo  
**University:** UC San Diego  

## 📌 Project Overview
This repository contains the Quarter 2 checkpoint code for my capstone project on **Machine Unlearning**. 

In Quarter 1, we explored heuristic unlearning methods (RMU, UNDIAL) on deep neural networks. In Quarter 2, we pivoted to a rigorous theoretical analysis of **Convex Optimization** to isolate the fundamental dynamics of unlearning without the stochastic noise of deep learning.

**The Core Question:** *Can Projected Gradient Ascent (PGA) serve as a valid mathematical proxy for "Exact Retraining" in convex models?*

## 🚀 Key Findings
1.  **The "Efficiency Window":** We identified a convex region where unlearning effectively removes targeted data before catastrophic forgetting occurs.
    * **Linear Regression (Abalone):** Optimal stopping point at **Step 77**.
    * **Logistic Regression (Breast Cancer):** Optimal stopping point at **Step 120**.
2.  **Geometric Alignment:** We mathematically proved that the unlearning update vector is **strictly parallel** (Cosine Similarity $\approx -1.0$) to the feature vector of the forgotten data, confirming the method acts as a precise projection.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `ES_unlearning_gradient_ascent.ipynb` | **Main Notebook.** Contains the complete experimental pipeline for both Linear and Logistic Regression unlearning. |
| `README.md` | Project documentation (this file). |

## 🧪 Experiments & Code Breakdown

The main notebook (`ES_unlearning_gradient_ascent.ipynb`) is organized into two major experiments:

### Part 1: Linear Regression (The "Batching" Discovery)
* **Dataset:** `abalone_scale` (Regression)
* **Method:** Ridge Regression with Projected Gradient Ascent.
* **Key Discovery:** Unlearning single points yielded weak signals. Implementing **Batch Unlearning** (Top 20 Outliers) amplified the gradient and revealed the "U-Shaped" distance curve.

### Part 2: Logistic Regression (Classification Robustness)
* **Dataset:** `breast_cancer` (Binary Classification)
* **Method:** Logistic Regression with Step Decay Learning Rate.
* **Key Result:** Successfully replicated the efficiency window in a classification setting, proving the robustness of the finding.

### Part 3: Geometric Sanity Checks (Advisor Requested)
* **Ratio Test:** Verifies that the element-wise ratio between the weight update ($\Delta w$) and feature vector ($x$) is constant.
* **Cosine Similarity:** Confirms the update direction is collinear ($-1.0$) with the data subspace.

## 📊 How to Run
1.  Open the notebook in **Google Colab** or a local Jupyter environment.
2.  Run the cells sequentially. The notebook will:
    * Download the datasets automatically via `sklearn` and `libsvm`.
    * Train the "Original" and "Gold Standard" (Retrained) models.
    * Execute the Unlearning Algorithm (Gradient Ascent).
    * Generate the **Optimization Trajectory** graphs (Distance vs. Steps).
    * Print the **Geometric Validation** metrics (Cosine Similarity).

## 🛠 Dependencies
* `numpy`
* `matplotlib`
* `scikit-learn`
* `requests`

---
*This project is part of the Data Science Capstone sequence at UCSD.*
