# 🧠 Road to Stanford CS336

A self-study roadmap for learning AI from scratch — starting at high school math level and ending at the ability to understand [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu), a graduate-level course on building large language models from scratch.

**100% free. No credit cards. No free trials. No paywalls.**

***

## What Is This?

This repo tracks a personal learning journey from MYP5 math + basic Python to understanding how modern LLMs (like GPT) are built at a research level. Every resource linked here is completely free and openly accessible.

The target — Stanford CS336 — covers building a full LLM from scratch: tokenizers, transformers, distributed training, scaling laws, and RLHF. It is a graduate-level course. The roadmap here closes every gap between high school and that starting line.

***

## Roadmap Overview

| Phase | Content | Est. Time |
|---|---|---|
| **0** | Math: Linear Algebra, Calculus, Probability | 6–8 weeks |
| **1** | Python depth (CS50P + NumPy) | 3–4 weeks |
| **2** | ML Foundations (Google MLCC) | 2–3 weeks |
| **3** | Practical Deep Learning (fast.ai) | 8–10 weeks |
| **4** | Neural Networks from Scratch (Karpathy) | 12–16 weeks |
| **5** | PyTorch fundamentals | 3–4 weeks |
| **6** | Transformers & Tokenizers (Hugging Face) | 4–5 weeks |
| **7** | Stanford CS336 Lectures | 12+ weeks |

**Total: ~14–18 months of consistent work.**

***

## Phase 0 — Math

> Everything in deep learning is applied linear algebra and calculus. Do not skip or rush this phase.

- 📐 **Linear Algebra** — [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 📈 **Calculus** — [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- 🎲 **Probability & Statistics** — [Khan Academy: Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)
- 📝 **Practice Problems** — [MIT 18.06 Linear Algebra (OCW)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)

***

## Phase 1 — Python Depth

> CS336 warns its code volume is "an order of magnitude greater than a typical ML course." Solid Python is non-negotiable.

- 🐍 **CS50P** — [Harvard CS50's Introduction to Programming with Python](https://cs50.harvard.edu/python/)
- 🔢 **NumPy** — [NumPy Learn](https://numpy.org/learn/)

***

## Phase 2 — ML Foundations

> Learn what a loss function is, why gradient descent works, and what overfitting means — before touching neural networks.

- 🤖 **Google Machine Learning Crash Course** — [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)

***

## Phase 3 — Practical Deep Learning

> Build working models first, understand the theory as it becomes relevant. The best top-down deep learning course available for free.

- ⚡ **fast.ai: Practical Deep Learning for Coders** — [course.fast.ai](https://course.fast.ai)
  - Runs entirely in free Kaggle notebooks
  - No login required to watch lectures

***

## Phase 4 — Neural Networks from Scratch (Core Phase)

> This is the most important phase. Andrej Karpathy builds everything from backpropagation to a working GPT, from scratch, in Python.

- 🎥 **Karpathy: Neural Networks Zero to Hero** — [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)
- ▶️ **YouTube Playlist** — [youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- 💻 **Code (GitHub)** — [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

| Lecture | Topic | Direct Link |
|---|---|---|
| 1 | Backpropagation & micrograd | [Watch](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| 2 | Bigram language model | [Watch](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| 3 | MLP language model | [Watch](https://www.youtube.com/watch?v=TCH_1BHY58I) |
| 4 | Activations, gradients, BatchNorm | [Watch](https://www.youtube.com/watch?v=P6sfmUTpUmc) |
| 5 | Backprop from scratch (manual) | [Watch](https://www.youtube.com/watch?v=q8SA3rM6ckI) |
| 6 | WaveNet + `nn.Module` internals | [Watch](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) |
| 7 | **Build GPT from scratch** | [Watch](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 8 | GPT Tokenizer (BPE) from scratch | [Watch](https://www.youtube.com/watch?v=zduSFxRajkE) |

> **Rule:** Write every line of code yourself before watching Karpathy's solution. Watching without coding is nearly worthless.

***

## Phase 5 — PyTorch

> CS336 uses raw PyTorch — no wrappers. You need to be fluent in the API.

- 🔥 **Official PyTorch Tutorials** — [pytorch.org/tutorials/beginner/basics/intro.html](https://pytorch.org/tutorials/beginner/basics/intro.html)

Key topics to cover: tensor operations, autograd, `nn.Module`, optimizers, DataLoaders, device management (`.to(device)`).

***

## Phase 6 — Transformers & Tokenizers

> Bridge from "I built a GPT" to "I understand the modern LLM ecosystem."

- 🤗 **Hugging Face NLP Course** — [huggingface.co/learn/nlp-course/chapter1/1](https://huggingface.co/learn/nlp-course/chapter1/1)
  - Focus: Chapters 1–4 (transformers), Chapter 6 (tokenizers deep dive), Chapter 7 (causal language modeling)

***

## Phase 7 — Stanford CS336

> The destination. All lectures are publicly posted on YouTube and free to watch.

- 🎓 **Course Website** — [cs336.stanford.edu](https://cs336.stanford.edu)
- ▶️ **YouTube Playlist (Spring 2025)** — [youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)
- 💻 **Lecture Code** — [github.com/stanford-cs336/spring2025-lectures](https://github.com/stanford-cs336/spring2025-lectures)
- 📋 **Assignments** — [github.com/stanford-cs336/spring2025-assignments](https://github.com/stanford-cs336/spring2025-assignments)

> **Note:** The graded assignments require paid GPU compute (~$5–7/hr on AWS/GCP). Following the lectures and understanding the assignment code is fully achievable for free using Google Colab for smaller experiments.

***

## Full Detailed Guide

The complete roadmap with per-lecture breakdowns, specific warnings, checkpoints, and things to look out for at every step is in:

📄 [`ai_roadmap_to_cs336.md`](./ai_roadmap_to_cs336.md)

***

## Tools Used

| Tool | Purpose |
|---|---|
| [Google Colab](https://colab.research.google.com) | Free GPU notebook environment |
| [Kaggle Notebooks](https://www.kaggle.com/code) | Free GPU for fast.ai exercises |
| [GitHub](https://github.com) | Version control and portfolio |
| [VS Code](https://code.visualstudio.com) | Local development |

***

## Progress Tracking

- [ ] Phase 0 — Math complete
- [ ] Phase 1 — Python depth complete
- [ ] Phase 2 — MLCC complete
- [ ] Phase 3 — fast.ai complete
- [ ] Phase 4 — Karpathy Zero to Hero complete
- [ ] Phase 5 — PyTorch fundamentals complete
- [ ] Phase 6 — Hugging Face NLP course complete
- [ ] Phase 7 — CS336 lectures followed

***

*Started: June 2026*
