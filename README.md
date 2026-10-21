# Transformer Decoding and Attention from Scratch
Assignment 3 of the NLP course at the University of Tehran (Spring 2026).
The notebook template and test cells were provided by the course; the implementations are mine.

Pure-Python implementation (no ML libraries) of the core pieces of a decoder-only LLM:
- Numerically stable softmax, cross-entropy and perplexity
- Greedy decoding, random sampling and temperature (entropy audit across temperatures)
- Multi-head self-attention: Q/K/V projections, scaled dot-product, causal mask, W_O
