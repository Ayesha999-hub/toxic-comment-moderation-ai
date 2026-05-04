# 🧠 Toxic Comment Moderation with Responsible AI

This project implements a complete **end-to-end AI moderation system** using the Jigsaw Unintended Bias dataset. It goes beyond standard classification by incorporating **fairness analysis, adversarial robustness, mitigation strategies, and a production-style guardrail pipeline**.

---

##  Project Overview

The goal of this project is to simulate a real-world content moderation system similar to those used by large platforms (e.g., YouTube, Meta).

The system includes:

-  Transformer-based toxicity classifier (DistilBERT)
-  Bias audit across demographic groups
-  Adversarial attack simulations
-  Fairness mitigation techniques
-  Production-grade moderation pipeline

---

##  Dataset

We use the **Jigsaw Unintended Bias in Toxicity Classification dataset**, which contains ~1.8M comments with identity annotations.

Key columns used:
- `comment_text` — input text
- `toxic` — toxicity score
- `black`, `white` — identity attributes for fairness analysis
├── part1.ipynb # Baseline model (DistilBERT)
├── part2.ipynb # Bias audit
├── part3.ipynb # Adversarial attacks
├── part4.ipynb # Mitigation techniques
├── part5.ipynb # Guardrail pipeline demo
├── pipeline.py # ModerationPipeline class
├── requirements.txt
└── README.md


---

##  Model (Part 1)

- Model: `distilbert-base-uncased`
- Task: Binary toxicity classification
- Metrics:
  - Accuracy: ~0.95
  - F1 Score: ~0.62
  - AUC: ~0.94

The model performs well overall but shows imbalance in detecting toxic content.

---

## ⚖️ Bias Audit (Part 2)

We evaluated fairness across two cohorts:

- **High-Black cohort**
- **Reference (White) cohort**

### Findings:
- Unequal True Positive Rates (TPR)
- Differences in False Positive Rates (FPR)
- Evidence of group-level disparity

 Note: Cohort sizes were small, so results have high variance.

---

##  Adversarial Attacks (Part 3)

### Attack 1: Text Evasion
- Unicode + spacing + duplication
- Attack Success Rate: **~92%**

 Model is highly vulnerable to obfuscation.

### Attack 2: Data Poisoning (Simulated)
- 5% label flipping
- Expected degradation in F1 and increase in missed toxic content

---

##  Mitigation Techniques (Part 4)

We explored:

- Oversampling (improves recall slightly)
- Reweighing (fairness vs accuracy tradeoff)
- Threshold optimization (simulated)

### Key Insight:
> Improving fairness often reduces accuracy — tradeoffs are unavoidable.

---

##  Guardrail Pipeline (Part 5)

A production-style moderation system with 3 layers:

### Layer 1: Rule-based Filter
- 20+ regex patterns
- Blocks explicit threats instantly

### Layer 2: ML Model
- DistilBERT predictions
- Thresholds:
  - ≥ 0.6 → Block
  - ≤ 0.4 → Allow

### Layer 3: Human Review
- Uncertainty band (0.4–0.6)

---

###  Pipeline Results (1000 samples)

- Blocked: 61
- Allowed: 936
- Sent to review: 3

 Efficient system with low human review load

---

##  Key Learnings

- High accuracy ≠ fairness
- Transformer models are vulnerable to adversarial text
- Bias mitigation introduces tradeoffs
- Production systems require multi-layer safeguards

---

##  Setup Instructions

```bash
pip install -r requirements.txt
---

##  Project Structure
Run notebook in order:

part1.ipynb
part2.ipynb
part3.ipynb
part4.ipynb
part5.ipynb
Environment
Python 3.10+
PyTorch
HuggingFace Transformers
Scikit-learn
Google Colab (T4 GPU)
Notes
Dataset not included due to size
Model checkpoints not included
Some mitigation techniques are simulated due to compute limits
Conclusion

This project demonstrates how to build a responsible AI system that goes beyond accuracy by addressing:

Bias
Robustness
Real-world deployment constraints
