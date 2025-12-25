# Turn this project to:

A conversational system built on top of a pretrained language model

We use: Pretrained causal LM.

We build a chatbot around it by:

1. Implement conversation memory (history replay, summarization, truncation) - **DONE** (at least append and truncation methods, can add summarization easily)
2. Fine-tune on dialogue data - **TODO** (Cant find the finetuning code - rewrite, finetune and take loss curves etc., have metrics)
3. Build a serving system (Flask + Docker, decoding, session handling) - **DONE** (can work on scalable design more)
