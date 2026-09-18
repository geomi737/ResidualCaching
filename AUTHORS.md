# Credits and provenance

- **geomi737** — project author; proposed the residual-sum token-merging idea and directed the one-way causal architecture and its implementation.
- **Andrej Karpathy** — author of [nanoGPT](https://github.com/karpathy/nanoGPT), the upstream GPT implementation used here. Thank you for the code and educational work that made this experiment possible.
- **SmolLM2 authors and Hugging Face** — thank you for the compact model family, architecture/configuration, tokenizer, and open tooling. The current scratch experiment reuses the SmolLM2 configuration and tokenizer, but loads no pretrained weights. [Pinned model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md).
- **Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher, Salesforce Research, and Wikipedia contributors** — thank you for WikiText and its source articles, which make reproducible language-model comparisons possible. [Pinned dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md); [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843).
- **Gemini** — AI assistance during project development, acknowledged at the project author's request.
- **ChatGPT, including Codex** — AI assistance with implementation refinement, correctness checks, documentation, and publication preparation.

AI assistance does not imply endorsement by Google, OpenAI, Hugging Face, Salesforce, Wikipedia contributors, or Andrej Karpathy. This is an independent research project.

The public Git history records this project's implementation and publication. It does not establish a claim that no similar idea existed previously. No global novelty or priority claim is made in this release.
