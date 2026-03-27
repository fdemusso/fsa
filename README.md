# **FSA**
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Sentence_transformers_(SBERT)-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="SBERT" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/Joblib-0056D2?style=for-the-badge&logo=data&logoColor=white" alt="Joblib" />
  <img src="https://img.shields.io/badge/Apache_Parquet-6F42C1?style=for-the-badge&logo=apache&logoColor=white" alt="Parquet" />
</p>

### *Beyond Metadata: A Geometric Approach to Movie Recommendations*

Most recommendation engines fail because they are either too "safe" (recommending only blockbusters) or too "random" (ignoring plot subtext). This project implements a **Hybrid Semantic Engine** that uses **Sentence-BERT (SBERT)** to map movie plots into a high-dimensional latent space, combined with a custom-weighted ranking system to prioritize quality and niche discovery.

---

## **📂 The Architecture: Navigating the Books**

The project is structured into three progressive stages, each serving a specific diagnostic and engineering purpose:

### **Book 01: Data Ingestion & Feature Engineering**
* **The Goal:** Transform raw, messy movie metadata into a structured "Content Matrix."
* **Key Work:** Extracting genres and directors using sparse encoding and generating **384-dimensional dense embeddings** for plot summaries using the `all-MiniLM-L6-v2` transformer.

### **Book 02: EDA & Geometric Diagnostics**
* **The Goal:** Prove that our features are mathematically sound before building the engine.
* **Key Work:** * **Statistical Moments:** Analysis of score distributions ($S_{final}$).
    * **Dimensionality Reduction:** Using **TruncatedSVD** to project 18,000+ films into a 2D manifold.
    * **The "Mannaia" Logic:** Initial testing of statistical confidence filters to separate "ghost entries" (low-vote anomalies) from legitimate niche cinema.

### **Book 03: Engine Calibration & Deployment**
* **The Goal:** The "Synthesis" phase. Tuning the weights of the final ranking function.
* **Key Work:** Implementing the **High-Level Interactive Engine**. This is where we balance semantic similarity against professional critique and audience sentiment.

---

## **🛠️ The "Secret Sauce": Technical Choices**

### **1. Semantic Plot Embeddings (SBERT)**
Unlike keyword matching (TF-IDF), SBERT understands that a movie about "space exploration" is semantically closer to "astronautical journey" even if they share no words. This ensures the engine captures the **thematic essence** of a film.

### **2. Anti-Blockbuster Penalty (Non-Linear)**
To prevent the system from becoming a "Top 10 Marvel Movies" generator, I implemented a **dynamic popularity penalty**. It applies a logarithmic dampening to high-volume titles, allowing the engine to surface "Hidden Gems" that share the same DNA as mainstream hits.

### **3. The "Mannaia" (Statistical Confidence)**
A common failure in ranking is the "Small Sample Bias" (a movie with one 10/10 rating ranking #1).
* **The Fix:** A dual-gate logic filter:
    * **Sustain:** Keep if $Critics \ge 50$ OR $Audience \ge 100$.
    * **Ghosting:** Wipe if both are below the threshold.

---

## **⚖️ Trade-offs: Pros & Cons**

### **The Pros**
* **Discovery over Safety:** Excellent at finding "If you liked X, try this obscure Y."
* **Geometric Integrity:** Recommendations are mathematically proven to be neighbors in the latent space.
* **High Precision:** The weighted blend ($0.82$ Semantic / $0.18$ Quality) ensures results are relevant AND good.

### **The Cons (The "Truth" Section)**
* **Cold Start Problem:** New movies with zero reviews are filtered out by the "Mannaia" to preserve system integrity.
* **Computationally Heavy:** SBERT embeddings require significant RAM/GPU for the initial encoding phase.
* **Data Dependency:** The engine is only as good as the plot summaries; vague descriptions lead to weaker semantic clusters.

### **🚀 How to Run (The Fast Track)**

To deploy the engine and start querying, follow this minimal sequence:

1.  **Initialize Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Build the Foundation (Book 01):**
    Run `Book 01` to process raw metadata and generate the **SBERT Embeddings**. This creates the `content_matrix.pkl` and cleaned assets required for inference.
3.  **Launch the Engine (Book 03):**
    Open `Book 03` and execute the **High-Level Interface**. The system will load the serialized assets and open the interactive prompt for real-time recommendations.

---

### **🔬 Scientific Audit (Optional)**

If you wish to verify the **mathematical integrity** of the model:
* **Run Book 02:** This notebook provides a deep dive into the EDA, including statistical moments of the $S_{final}$ metric, Shannon entropy of features, and the **Manifold Visualization** (TruncatedSVD) to confirm geometric cluster convergence.

---

### **Final Reflection**
This project demonstrates a full-stack data science workflow: from raw NLP engineering to a user-facing interactive tool. It prioritizes **mathematical rigor** over "black-box" magic, ensuring that every recommendation can be audited, visualized, and justified.