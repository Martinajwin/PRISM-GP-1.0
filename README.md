# Pareto Resolved Interpretable Screening Model for GSK-3β and PIM-1 (PRISM-GP 1.0)

### Overview
PRISM-GP 1.0 is a Streamlit web application implementing a rigorous, parallel machine learning pipeline to concurrently screen and predict dual-target inhibitors for GSK-3β and PIM-1 kinases. Operating as a transparent "glass-box" framework, it employs strictly 2D topological Mordred molecular descriptors alongside multi-objective Explainable Boosting Machines (EBMs). The pipeline utilizes an imbalance-corrected Log-Odds decision function and has been extensively validated against extreme decoy datasets (including a <0.5% False Discovery Rate for PAINS) to ensure robust identification of selective and dual-acting compounds without generating false positives.

### Features
* **Flexible Input:** Input SMILES strings manually or via batch CSV upload.
* **Automated Feature Extraction:** Automatically computes the specific, mathematically optimized subsets of 25 elite 2D descriptors required for each respective kinase target.
* **Parallel Dual-Target Workflow:** Concurrently evaluates molecular affinity against both GSK-3β and PIM-1, mapping the Log-Odds outputs onto a Pareto-optimization decision boundary.
* **Strict Applicability Domain (AD) Logic:** Implements an independent Principal Component Analysis (PCA) AD constraint utilizing a Mahalanobis squared distance limit ($D^2 \le 5.991$) to flag structural extrapolations and prevent out-of-domain artifacts from advancing.
* **Integrated Diagnostics & Export:** Download standard binary screening results or detailed AD spatial diagnostic reports for further molecular dynamics or synthesis planning.

---

### Access the Web Tool
You can access and use the PRISM-GP 1.0 virtual screening pipeline directly through your web browser without any installation:

🔗 **[Launch PRISM-GP 1.0 Web Tool Here](https://prism-gp-1-dual-screening.streamlit.app/)** *(Note: Update this URL to match your actual deployed Streamlit link)*

---

### Citation
If you utilize the PRISM-GP 1.0 webtool or its concepts in your research, please cite:

> **PRISM-GP 1.0 Webtool** | D. Kumar, A. J. Martin | Manipal Academy of Higher Education (MAHE) | Version 1.0 (2026).
> **Webtool URL:** *(https://prism-gp-1-dual-screening.streamlit.app/)* 

---

### Copyright & Intellectual Property

**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**

**Authors/Creators:** Dileep Kumar and Ajwin Joseph Martin

The source code, algorithms, parallel consensus logic, Log-Odds thresholds, and trained models associated with PRISM-GP 1.0 are the exclusive intellectual property of Manipal Academy of Higher Education (MAHE). This repository is hosted on the creators' personal account and made public for the sole purpose of deploying the Streamlit web application and facilitating transparency for academic peer review.

**Permissions:**
* You are permitted to view the source code for educational and peer-review purposes.
* You are permitted to use the deployed web tool via the provided Streamlit URL for your own virtual screening tasks, provided proper citation is given.

**Restrictions:**
* You may **NOT** copy, reproduce, distribute, modify, or create derivative works from this codebase.
* You may **NOT** use the code or models for any commercial or private non-commercial deployment without explicit written permission from the copyright owner (MAHE) and the authors.

For licensing inquiries or permission requests, please contact the authors directly.
