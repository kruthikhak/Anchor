---
title: Anchor
emoji: 📘
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12.12"
app_file: serve.py
short_description: A domain-grounded AI learning assistant that uses RAG
pinned: false
license: mit
preload_from_hub:
  - BAAI/bge-base-en-v1.5 1_Pooling/config.json,config.json,config_sentence_transformers.json,model.safetensors,modules.json,sentence_bert_config.json,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.txt
  - BAAI/bge-reranker-base config.json,model.safetensors,sentencepiece.bpe.model,special_tokens_map.json,tokenizer.json,tokenizer_config.json
---

# Anchor

A study assistant for placement preparation that answers only from nine openly licensed
computer science textbooks, cites the page behind every claim, and says so when the books
don't cover a question.

Code, evaluation and write-up: https://github.com/kruthikhak/Anchor

The Space needs one secret, `GROQ_API_KEY`, set under Settings → Variables and secrets.
