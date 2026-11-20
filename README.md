# 🧠 Road to Stanford CS336 (DP1 Edition)

A self-study roadmap for learning AI from scratch. It starts at the level of an IB Diploma Programme Year 1 student and ends at the ability to follow [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu), a graduate-level course on building large language models.

**100% free. No credit cards. No free trials. No paywalls.**

---

## What Is This?

This repo tracks a personal learning journey from DP1 math and basic Python to understanding how modern LLMs (like GPT) are built at a research level. Every resource linked here is free and openly accessible.

The target, Stanford CS336, covers building a full LLM from scratch: tokenizers, transformers, distributed training, scaling laws, and alignment. The roadmap closes every gap between DP1 and that starting line.

---

## What Changed From the MYP5 Version

DP1 math already covers functions, exponentials and logarithms, trigonometry, and an introduction to calculus and statistics. So this edition:

- Cuts the math phase from 6-8 weeks to about 3-5 weeks and focuses on what the DP syllabus skips (linear algebra, multivariable calculus, the chain rule applied to many variables)
- Replaces Khan Academy probability with a quick check against your DP stats notes
- Adds a pacing plan around DP workload
- Adds a section on using the IB core (Extended Essay, Internal Assessments, CAS) for your AI work

---

## Roadmap Overview

| Phase | Content                                     | Est. Time   |
| ----- | ------------------------------------------- | ----------- |
| **0** | Math gaps: Linear Algebra, Multivariable Calculus | 3-5 weeks   |
| **1** | Python depth (CS50P + NumPy)                | 3-4 weeks   |
| **2** | ML Foundations (Google MLCC)                | 2-3 weeks   |
| **3** | Practical Deep Learning (fast.ai)           | 8-10 weeks  |
| **4** | Neural Networks from Scratch (Karpathy)     | 12-16 weeks |
| **5** | PyTorch fundamentals                        | 3-4 weeks   |
| **6** | Transformers & Tokenizers (Hugging Face)    | 4-5 weeks   |
| **7** | Stanford CS336 Lectures                     | 12+ weeks   |

**Total: roughly 13-17 months at 6-8 hours per week.** These are estimates. Your pace will differ.

---

## Pacing Around the DP

DP2 is the heaviest year. A realistic plan:

- **DP1 (now to May 2027):** Phases 0, 1, and 2. Start Phase 3 if your workload allows.
- **Summer 2027:** Phase 3 and the start of Phase 4. Long free blocks suit Karpathy's videos.
- **DP2 (Sep 2027 to May 2028):** Light maintenance only. One Karpathy lecture or one coding session per week at most. Exams come first.
- **After exams (from May 2028):** Finish Phase 4, then Phases 5, 6, and 7.

Adjust this to your own schedule. Protect your grades.

---

## Phase 0 - Math Gaps

> Everything in deep learning is applied linear algebra and calculus. DP1 gives you the base. This phase fills the rest.

First, check what you already know:

- 📐 **Linear Algebra** (new for most DP students): [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 📈 **Calculus** (skim, you know the basics): [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- 📊 **Multivariable Calculus** (partial derivatives, gradients): [Khan Academy: Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
- 🎲 **Probability & Statistics:** review your DP notes first. Use [Khan Academy: Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) only for topics you do not know, such as expectation, variance, and conditional probability.
- 📝 **Practice Problems:** [MIT 18.06 Linear Algebra (OCW)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)

**Checkpoint:** You multiply two matrices by hand, explain what a gradient is, and apply the chain rule to a function of two variables.

---

## Phase 1 - Python Depth

> CS336 warns its code volume is "an order of magnitude greater than a typical ML course." Solid Python is non-negotiable.

- 🐍 **CS50P:** [Harvard CS50's Introduction to Programming with Python](https://cs50.harvard.edu/python/)
- 🔢 **NumPy:** [NumPy Learn](https://numpy.org/learn/)

If you take IB Computer Science, you already know the basics. Skip to the later CS50P weeks and spend your time on NumPy.

---

## Phase 2 - ML Foundations

> Learn what a loss function is, why gradient descent works, and what overfitting means before touching neural networks.

- 🤖 **Google Machine Learning Crash Course:** [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)

---

## Phase 3 - Practical Deep Learning

> Build working models first. Learn the theory as it becomes relevant.

- ⚡ **fast.ai: Practical Deep Learning for Coders:** [course.fast.ai](https://course.fast.ai)
  * Runs in free Kaggle notebooks
  * No login required to watch lectures

---

## Phase 4 - Neural Networks from Scratch (Core Phase)

> This is the most important phase. Andrej Karpathy builds everything from backpropagation to a working GPT, from scratch, in Python.

- 🎥 **Karpathy: Neural Networks Zero to Hero:** [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)
- ▶️ **YouTube Playlist:** [youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- 💻 **Code (GitHub):** [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

| Lecture | Topic                             | Direct Link                                          |
| ------- | --------------------------------- | ---------------------------------------------------- |
| 1       | Backpropagation & micrograd       | [Watch](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| 2       | Bigram language model             | [Watch](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| 3       | MLP language model                | [Watch](https://www.youtube.com/watch?v=TCH_1BHY58I) |
| 4       | Activations, gradients, BatchNorm | [Watch](https://www.youtube.com/watch?v=P6sfmUTpUmc) |
| 5       | Backprop from scratch (manual)    | [Watch](https://www.youtube.com/watch?v=q8SA3rM6ckI) |
| 6       | WaveNet + `nn.Module` internals   | [Watch](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) |
| 7       | **Build GPT from scratch**        | [Watch](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 8       | GPT Tokenizer (BPE) from scratch  | [Watch](https://www.youtube.com/watch?v=zduSFxRajkE) |

> **Rule:** Write every line of code yourself before watching Karpathy's solution. Watching without coding is nearly worthless.

---

## Phase 5 - PyTorch

> CS336 uses raw PyTorch with no wrappers. You need to be fluent in the API.

- 🔥 **Official PyTorch Tutorials:** [pytorch.org/tutorials/beginner/basics/intro.html](https://pytorch.org/tutorials/beginner/basics/intro.html)

Key topics: tensor operations, autograd, `nn.Module`, optimizers, DataLoaders, device management (`.to(device)`).

---

## Phase 6 - Transformers & Tokenizers

> Bridge from "I built a GPT" to "I understand the modern LLM ecosystem."

- 🤗 **Hugging Face NLP Course:** [huggingface.co/learn/nlp-course/chapter1/1](https://huggingface.co/learn/nlp-course/chapter1/1)
  * Focus: Chapters 1-4 (transformers), Chapter 6 (tokenizers deep dive), Chapter 7 (causal language modeling)

---

## Phase 7 - Stanford CS336

> The destination. All lectures are publicly posted on YouTube and free to watch.

- 🎓 **Course Website:** [cs336.stanford.edu](https://cs336.stanford.edu)
- ▶️ **YouTube Playlist (Spring 2025):** [youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)
- 💻 **Lecture Code:** [github.com/stanford-cs336/spring2025-lectures](https://github.com/stanford-cs336/spring2025-lectures)
- 📋 **Assignments:** [github.com/stanford-cs336/spring2025-assignments](https://github.com/stanford-cs336/spring2025-assignments)

> **Note:** The graded assignments need paid GPU compute (roughly $5-7 per hour on AWS or GCP). Following the lectures and reading the assignment code is fully doable for free. Use Google Colab for smaller experiments.

---

## Using the IB Core for This Work

Your AI learning can feed your DP requirements. Check every idea with your teacher or coordinator first. Your school and the IB set the rules.

- **Extended Essay:** A Math or Computer Science topic on tokenization, gradient descent, or how attention works. Pick a narrow research question you can answer with your own experiments.
- **Math Internal Assessment:** Explore the math behind backpropagation or gradient descent. Show your own calculations and a small worked example in code.
- **CAS:** Teach a younger student basic Python or ML ideas. Share this roadmap with your school's coding club.
- **Portfolio:** Keep your code in this repo. It shows sustained, independent work.

---

## Tools Used

| Tool                                              | Purpose                        |
| ------------------------------------------------- | ------------------------------ |
| [Google Colab](https://colab.research.google.com) | Free GPU notebook environment  |
| [Kaggle Notebooks](https://www.kaggle.com/code)   | Free GPU for fast.ai exercises |
| [GitHub](https://github.com)                      | Version control and portfolio  |
| [VS Code](https://code.visualstudio.com)          | Local development              |

---

## Progress Tracking

- [ ] Phase 0 - Math gaps complete
- [ ] Phase 1 - Python depth complete
- [ ] Phase 2 - MLCC complete
- [ ] Phase 3 - fast.ai complete
- [ ] Phase 4 - Karpathy Zero to Hero complete
- [ ] Phase 5 - PyTorch fundamentals complete
- [ ] Phase 6 - Hugging Face NLP course complete
- [ ] Phase 7 - CS336 lectures followed

---

*Started: June 2026. DP1 edition: November 2026.*
