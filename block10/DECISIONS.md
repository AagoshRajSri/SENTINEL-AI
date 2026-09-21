# Sentinel-AI: Decision Log

1. **AmazonHelp Dataset Selection:** We chose the AmazonHelp Twitter dataset because it provides a realistic, high-volume mix of routine logistics inquiries and severe customer edge cases. This allowed us to build and evaluate the system against genuine, noisy dialogue rather than clean synthetic data.

2. **15,000-Row RAG Database:** We constrained our historical retrieval dataset to 15,000 human-resolved tickets to balance semantic variety with local memory limitations. This size ensures the local FAISS index loads quickly while still providing robust contextual grounding for the reply drafter.

3. **Human-Labeled Golden Set:** We manually annotated a 250-item evaluation set to establish an undeniable ground truth for intent and escalation. Relying on human judgment over automated labeling ensures our baseline metrics reflect true operational safety.

4. **Stratified Evaluation Set:** We intentionally oversampled rare, high-risk edge cases (63 of the 250 cases) in our golden set rather than strictly matching real-world distributions. This prevents the evaluation from masking critical failures on rare issues behind high accuracy on routine queries.

5. **Pydantic v2 Schema Enforcement:** We used Pydantic to strictly enforce the JSON output structure of the classification LLM. This prevents pipeline crashes caused by hallucinated keys or missing escalation flags, ensuring downstream components always receive reliable, typed data.

6. **Temperature 0 Generation:** We set the LLM temperature to 0 to eliminate randomness and force deterministic outputs. This makes our evaluation metrics reproducible and ensures the classifier behaves consistently on identical customer inputs.

7. **Local FAISS Index:** We implemented local, in-memory FAISS vector retrieval instead of relying on a managed cloud vector database. This choice drastically reduced operational latency to ~1ms and eliminated network dependency without sacrificing retrieval quality.

8. **RAG Instead of Fine-Tuning:** We chose Retrieval-Augmented Generation (RAG) over model fine-tuning because it grounds replies in actual historical actions and reduces hallucinations. It also allows the business to update support policies simply by changing the vector database, without needing to retrain a model.

9. **Recall-Biased Escalation Logic:** We calibrated our escalation threshold to favor false positives over false negatives, explicitly optimizing for recall. In enterprise triage, the business cost of missing a severe escalation (false negative) is exponentially higher than the manual review cost of an unnecessary escalation.

10. **Cohen's Kappa for Grading Agreement:** We used Cohen's Kappa to measure how reliably our automated LLM-as-a-judge aligned with human evaluators. This metric confirms that our automated reply-quality grading correlates with human preference beyond random chance, validating the evaluation pipeline.

11. **Force-Committing Caches:** We force-committed our API response caches to the repository to guarantee completely offline, deterministic reproducibility. This allows reviewers to run the full evaluation pipeline and verify our reported metrics without needing active API keys or incurring network latency.
