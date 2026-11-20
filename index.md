# Road to Stanford CS336: A Complete Free Self-Study Guide (DP1 Edition)
*Built for Sarang | Goal: Understand Stanford CS336 — Language Modeling from Scratch*

---

## Table of Contents

- [Overview: What You're Working Toward](#overview-what-youre-working-toward)
- [What Changed From the MYP5 Edition](#what-changed-from-the-myp5-edition)
- [Before You Start: Mindset and Setup](#before-you-start-mindset-and-setup)
- [Phase 0 — Close the Math Gaps](#phase-0--close-the-math-gaps)
  - [0.1 Linear Algebra (3Blue1Brown)](#01-linear-algebra-3blue1brown)
  - [0.2 Calculus (3Blue1Brown)](#02-calculus-3blue1brown)
  - [0.3 Probability & Statistics (DP notes + Khan Academy)](#03-probability--statistics-dp-notes--khan-academy)
- [Phase 1 — Python Depth (CS50P)](#phase-1--python-depth-cs50p)
- [Phase 2 — First Look at ML (Google MLCC)](#phase-2--first-look-at-ml-google-mlcc)
- [Phase 3 — Practical Deep Learning (fast.ai)](#phase-3--practical-deep-learning-fastai)
- [Phase 4 — The Core Pipeline (Karpathy Zero to Hero)](#phase-4--the-core-pipeline-karpathy-zero-to-hero)
- [Phase 5 — PyTorch Fundamentals](#phase-5--pytorch-fundamentals)
- [Phase 6 — Transformers and Tokenizers (Hugging Face NLP Course)](#phase-6--transformers-and-tokenizers-hugging-face-nlp-course)
- [Phase 7 — Stanford CS336 Lectures](#phase-7--stanford-cs336-lectures)
- [Realistic Timeline](#realistic-timeline)
- [Pacing Around the DP](#pacing-around-the-dp)
- [Using the IB Core for This Work](#using-the-ib-core-for-this-work)
- [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)

---

## Overview: What You're Working Toward

**Stanford CS336 — Language Modeling from Scratch** ([cs336.stanford.edu](https://cs336.stanford.edu)) is a graduate-level course taught by Percy Liang and Tatsunori Hashimoto. The goal of the course is to build a large language model entirely from scratch — including the tokenizer, the transformer architecture, the training loop, distributed training across GPUs, and alignment techniques like RLHF.

The course officially requires:
- Fluent Python and PyTorch
- Multivariate calculus, linear algebra, probability at a college level
- At least one prior deep learning course
- Experience reading ML research papers

You are now in the IB Diploma Programme (DP1), taking Math AA HL, Physics HL, and Chemistry HL. You have solid Python from CS50x and JavaScript experience. This guide maps out every step between where you are now and where CS336 starts, using only 100% free resources — no credit cards, no free trials, no paywalls.

---

## What Changed From the MYP5 Edition

Math AA HL already covers a lot of what the old Phase 0 taught from scratch. Over the two years it gives you functions, exponentials and logarithms, trigonometry, differentiation and integration, Maclaurin series, vectors in 3D, complex numbers, and a full probability and statistics unit. Your DP1 term may not have reached all of it yet, so check the order your teacher follows and treat the later topics as "coming soon" and not "done". Physics HL adds practical experience with vectors and rates of change. Chemistry HL does not overlap directly, but it adds to your workload. So this edition:

- **Shortens Phase 0** from 6–8 weeks to about 3–4 weeks. Matrices and linear algebra stay almost in full, because Math AA does not teach matrices. Calculus becomes a skim plus a focused multivariable gap. Probability becomes a light review plus entropy.
- **Keeps Phases 1–7** with the same resources and links.
- **Adds a pacing plan** that works around three HL subjects and exams.
- **Adds a section on the IB core** (Extended Essay, Internal Assessments, CAS).
- **Updates the pitfalls list** with DP-specific advice.

---

## Before You Start: Mindset and Setup

**Code everything.** Reading about neural networks without coding them is like reading about swimming without getting in the water. Every phase in this guide requires you to run code, not just watch videos. If you skip this, you will not be ready for CS336.

**Use Google Colab** ([colab.research.google.com](https://colab.research.google.com)) as your coding environment throughout this entire journey. It gives you a free GPU in the browser, requires only a Google account (which you already have), and runs Jupyter notebooks — the same format used by CS336, Karpathy, fast.ai, and Hugging Face.

**Use GitHub** ([github.com](https://github.com)) to save your work. You already use VS Code, so install the GitHub extension and commit every notebook you write. This builds a portfolio that proves your work is real.

**Do not rush phases.** The most common mistake is moving forward before the previous phase is solid. If something in Phase 4 doesn't make sense, the problem is almost always Phase 0 — a math gap you skipped.

**Protect your DP grades first.** This roadmap is a long-term project. It is not worth a drop in your predicted grades. See [Pacing Around the DP](#pacing-around-the-dp).

---

## Phase 0 — Close the Math Gaps

Everything in deep learning — every weight update, every attention score, every loss function — is applied linear algebra and calculus. DP1 gives you a strong base, so this phase fills the specific gaps instead of rebuilding the base.

**Estimated time: 3–4 weeks**

---

### 0.1 Linear Algebra (3Blue1Brown)

**Link:** [youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

This is the Essence of Linear Algebra playlist by Grant Sanderson (3Blue1Brown). It is 16 videos, approximately 3–5 minutes to 20 minutes each. The visual approach is uniquely good — you will come away with intuition that people who learned through pure symbol manipulation never build.

Math AA HL covers vectors, including dot products, cross products, and lines and planes in 3D. Chapters 1, 2, and 9 will feel familiar, so watch them at a faster speed. Math AA does not teach matrices, so Chapters 3, 4, 6, and 14 are new. Slow down there. Physics HL also helps, because you already use vectors for forces and fields.

**What to watch and what to pay attention to:**

| Video | Key Concept | What to Look Out For |
|---|---|---|
| Ch. 1 — Vectors | Vectors as arrows in space | Understand both the geometric view (arrow) and the numerical view (list of numbers) — AI uses both simultaneously |
| Ch. 2 — Linear combinations, span, basis | What a basis is | The idea of a "basis" is everywhere in NLP — word embeddings are vectors in a basis |
| Ch. 3 — Linear transformations and matrices | Matrices as transformations | This is the single most important video — a matrix multiplication is not just arithmetic, it is a transformation of space |
| Ch. 4 — Matrix multiplication as composition | Chaining transformations | Every layer in a neural network is a matrix multiplication; stacking layers = composing transformations |
| Ch. 6 — The determinant | Scaling of space | Used in understanding probability distributions |
| Ch. 9 — Dot products | Inner products | Attention in a transformer is literally dot products — understand this deeply |
| Ch. 14 — Eigenvectors and eigenvalues | Directions that don't change under transformation | Used in PCA, optimization, understanding training dynamics |

**After watching:** Go to MIT OpenCourseWare 18.06 ([ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)) and do the first 5 problem sets. These are actual MIT assignments, free to access. They will show you whether you actually understood or just felt like you understood.

**Red flag:** If you can't answer "what does matrix multiplication mean geometrically?" without looking it up, you are not ready for the next video yet.

---

### 0.2 Calculus (3Blue1Brown)

**Link:** [youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

This is the Essence of Calculus playlist. It is 12 videos. Math AA HL covers derivatives, the chain rule, and integration, so treat this series as a skim. Watch it at a faster speed and stop on the chapters below, because they build the intuition underneath the symbol-pushing — which is what you need for understanding backpropagation.

**What you must deeply understand from this series:**

- **Chapter 2 (The Paradox of the Derivative):** Understand instantaneous rate of change. In neural networks, a derivative tells you how much a weight affects the loss — this is the entire basis of training.
- **Chapter 4 (Visualizing the Chain Rule):** The chain rule is the mathematical foundation of backpropagation. If you do not understand why the chain rule works, you will not understand why neural networks can be trained at all. Watch this video at least twice, even though you can already apply the rule.
- **Chapter 8 (Derivatives of exponentials):** The function e^x appears constantly — in softmax, in activation functions, in probability. Know why its derivative is itself.
- **Chapter 11 (Taylor Series):** Important for understanding approximations in optimization. Math AA HL covers Maclaurin series, so skim this chapter unless your course has not reached them yet. You will see this idea in CS336's discussion of training dynamics.

**The main gap: multivariable calculus.** Math AA HL calculus is single-variable. Deep learning needs many variables at once.

- **Gradient descent video:** Watch 3Blue1Brown's "Gradient descent, how neural networks learn" ([youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w)). It bridges single-variable calculus to the multi-variable gradient descent used in AI.
- **Partial derivatives and gradients:** Work through the first sections of Khan Academy's Multivariable Calculus course ([khanacademy.org/math/multivariable-calculus](https://www.khanacademy.org/math/multivariable-calculus)). Cover partial derivatives, the gradient, and the multivariable chain rule. You do not need the sections on integration, surfaces, or vector fields right now.
- **The gradient is just a vector of partial derivatives.** Internalize this. If you have a function with 1 million parameters (a small neural network), the gradient is a vector with 1 million numbers, each telling you which direction to nudge one weight to reduce the loss.

**Red flag:** If you cannot explain why the chain rule works in your own words, stop and rewatch Chapter 4 before moving on.

---

### 0.3 Probability & Statistics (Math AA HL notes + Khan Academy)

**Link (only for gaps):** [khanacademy.org/math/statistics-probability/probability-library](https://www.khanacademy.org/math/statistics-probability/probability-library)

Math AA HL has a full probability and statistics unit, including conditional probability, Bayes' theorem, distributions, and expected value. Start with your own notes and past exercises instead of Khan Academy. If your course has not reached the unit yet, use Khan Academy for those topics. Language models are probabilistic systems — a transformer outputs a probability distribution over vocabulary tokens. If you don't understand probability, you cannot understand what a model is actually doing.

**Check each topic against your Math AA notes first. Use Khan Academy only for what you cannot do cold:**

- **Probability basics:** Events, probability of an event, conditional probability (P(A|B) — probability of A given B). This is used constantly in language models where you compute the probability of the next word *given* all previous words.
- **Random variables and distributions:** Discrete and continuous probability distributions. The softmax output of a language model is a discrete probability distribution over vocabulary.
- **Expected value and variance:** Used when discussing training objectives and reward models.
- **The concept of entropy:** Not covered in Math AA and not deeply on Khan Academy, but you should understand intuitively that entropy measures "how uncertain" a distribution is. Search for "Shannon entropy explained simply" after finishing the review.

**What to skip for now:** Hypothesis testing and confidence intervals. They are less relevant at this stage. Come back later if needed.

**Red flag:** You must be able to answer — without looking it up — "what is a conditional probability and why is P(next word | all previous words) the right way to think about language modeling?"

---

## Phase 1 — Python Depth (CS50P)

**Link:** [cs50.harvard.edu/python](https://cs50.harvard.edu/python)

You already know Python from CS50x. CS50P goes deeper. It is 100% free with no account required to watch lectures, though you will want to create a free edX account to submit problem sets and get feedback.

**Why you still need this:**
CS336 is explicit that its code volume is "an order of magnitude greater than a typical ML course." The assignments involve writing optimized, production-style Python. CS50P will significantly strengthen your ability to write clean, modular, well-tested Python code.

**Estimated time: 3–4 weeks**

**Topics in CS50P and what to pay extra attention to:**

| Week | Topic | Why It Matters for AI |
|---|---|---|
| Week 2 — Loops | Generator functions and lazy evaluation | PyTorch DataLoaders use generators; understanding lazy evaluation is essential for handling large datasets |
| Week 4 — Libraries | NumPy and working with external packages | NumPy is the backbone of all numerical computation in Python; everything in AI uses it |
| Week 6 — File I/O | Reading and writing large files | Training data is massive; you need to handle file I/O efficiently |
| Week 8 — Object-Oriented Programming | Classes, inheritance, dunder methods | PyTorch's `nn.Module` is a class; you will subclass it for every neural network you build |
| Week 9 — Et Cetera | `*args`, `**kwargs`, type hints, decorators | Used extensively in PyTorch and modern AI codebases |

**After CS50P:** Spend one week getting comfortable with **NumPy** specifically. Go to [numpy.org/learn](https://numpy.org/learn) and work through the "NumPy fundamentals" section. Focus on array creation, broadcasting, and vectorized operations. The reason: before PyTorch exists, everything in ML is just NumPy. And even with PyTorch, understanding NumPy makes you understand what PyTorch tensors are actually doing.

**What to look out for:**
- Broadcasting rules in NumPy are confusing at first. A `(3,)` array times a `(3, 4)` matrix doesn't fail — it broadcasts. You need to understand why, because the same thing happens with tensors in PyTorch, and silent broadcasting bugs are a common source of incorrect model behavior.

---

## Phase 2 — First Look at ML (Google Machine Learning Crash Course)

**Link:** [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)

This is Google's free, fully reimagined (2024 version) Machine Learning Crash Course. It is approximately 15 hours, completely free, and requires no login. It covers fundamental ML concepts with interactive visualizations and code exercises using Python.

**Estimated time: 2–3 weeks**

**What this course covers that you need:**

- **Supervised learning:** The idea of mapping inputs to outputs using training data. This is the paradigm all language models use.
- **Loss functions:** The mathematical quantity that measures how wrong a model is. Training a neural network = minimizing a loss function.
- **Gradient descent:** The algorithm used to minimize the loss. You have the calculus intuition from Phase 0 — this is where you see it applied.
- **Overfitting and regularization:** A model can memorize training data instead of learning patterns. This is one of the most important practical problems in AI.
- **The 2024 update added:** Coverage of large language models, generative AI basics, and AutoML — directly relevant to your goal.

**What to look out for:**
- The course teaches gradient descent conceptually. The implementation behind the scenes is more complex (mini-batches, learning rate schedules, adaptive optimizers like Adam). Don't worry about those details yet — they come later. Focus on understanding *why* gradient descent works.
- The course uses TensorFlow for exercises. You will switch to PyTorch later. The concepts are identical — don't let the library difference confuse you.

**Checkpoint:** After this course, you should be able to answer: "What is a loss function, why do we compute gradients with respect to it, and what does gradient descent do with those gradients?" If you can explain this in plain language, you're ready for Phase 3.

---

## Phase 3 — Practical Deep Learning (fast.ai)

**Link:** [course.fast.ai](https://course.fast.ai)

This is fast.ai's "Practical Deep Learning for Coders" — approximately 60 hours of material, completely free, no login required to access lectures. All exercises run in Kaggle notebooks (free, no GPU cost). This is a legendary course in the AI community and is the best top-down introduction to deep learning available anywhere.

**Estimated time: 8–10 weeks**

**What makes this course unusual (and why it works):**
Most courses teach you theory first, then application. fast.ai does the opposite — you build a working image classifier in Lesson 1, and the theory is revealed gradually. This approach means you have working code from the start, which makes the theory much easier to understand when it appears.

**Lesson-by-lesson guide:**

- **Lesson 1 — Getting Started:** You train an image classifier. Don't just run the cells — understand what `learn.fit_one_cycle()` is doing. It is running gradient descent on a neural network. At this point you don't need to know the implementation details; just get comfortable with the workflow.
- **Lesson 2 — Deployment:** How to take a model to production. Practical skill, less critical for CS336, but good to know.
- **Lesson 3 — Neural Net Foundations:** This is where the theory starts. Pay very close attention to the manual implementation of gradient descent from scratch. This directly prefigures what Karpathy does in Phase 4.
- **Lesson 4 — NLP:** Your first look at text classification and language models. Pay attention to tokenization — how text becomes numbers. This is a core CS336 topic.
- **Lesson 5 — From Scratch:** fast.ai rebuilds neural networks from scratch. Critical viewing. Code alongside every cell.
- **Lesson 6 — Random Forests:** Less relevant to CS336; skim this one.
- **Lesson 7 — Collaborative Filtering:** Less relevant; skim.
- **Lessons 8–9 — Tabular and NLP from Scratch:** Return to NLP. These lessons introduce the ideas of embeddings and attention, which are the foundation of transformers.

**What to look out for:**
- The fast.ai library (`fastai`) is a high-level wrapper. In CS336, you will use raw PyTorch with no wrappers whatsoever. Use fast.ai to understand *what* these operations do, but get comfortable with the idea that later you will do the same things manually.
- Kaggle is free but requires account creation. It does not require a credit card and is appropriate for your age.

---

## Phase 4 — The Core Pipeline (Karpathy Zero to Hero)

**Link:** [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)  
**YouTube playlist:** [youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)  
**GitHub (code):** [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

This is the single most important resource in this entire guide for your goal. Andrej Karpathy (co-founder of OpenAI, former Director of AI at Tesla) built a series of videos that starts with calculus and ends with a working GPT. Every video involves building something from scratch in Python. This is as close as you can get to CS336's philosophy without being in the course itself.

**Estimated time: 12–16 weeks (do not rush this)**

---

**Lecture-by-lecture breakdown:**

### Lecture 1 — Micrograd: Backpropagation from Scratch
**Video:** [youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0)  
**Duration:** ~2.5 hours

Karpathy builds `micrograd` — a tiny autograd engine in ~150 lines of Python. Autograd is the system that PyTorch (and all deep learning frameworks) use to automatically compute gradients. You will implement:
- A `Value` class that wraps a number and tracks its gradient
- Forward pass computation
- Backward pass (backpropagation) through the chain rule

**What to look out for:**
- The `backward()` function in each operation node is just the chain rule applied to that specific operation. Addition: `d_loss/d_a = d_loss/d_output * 1`. Multiplication: `d_loss/d_a = d_loss/d_output * b`. If you don't understand why, go back to calculus Phase 0 and rewatch the chain rule video.
- Topological sort: the backward pass must happen in reverse order of the forward pass. Understand why.
- **Do not just watch.** Pause the video after every few minutes and write the code yourself before Karpathy reveals his version. This takes 3–4x longer but is 10x more effective.

**Checkpoint:** After this lecture, you should be able to implement a two-layer neural network that trains on a simple dataset using only your `micrograd` engine and plain Python. No PyTorch.

---

### Lecture 2 — Makemore Part 1: Bigram Language Model
**Video:** [youtube.com/watch?v=PaCmpygFfXo](https://www.youtube.com/watch?v=PaCmpygFfXo)  
**Duration:** ~1.5 hours

Karpathy introduces language modeling using character-level bigrams. A bigram model predicts the next character based only on the current character. This is the simplest possible language model — but the concepts introduced (token probabilities, loss functions for language models, the idea of "next token prediction") are identical to what GPT-4 does at a much larger scale.

**What to look out for:**
- **Negative log likelihood loss:** This is the loss function used in all language models, including CS336. Understand why it makes sense: you want to maximize the probability of the correct next token, which is equivalent to minimizing the negative log of that probability.
- The connection between the count matrix and the neural network version. Karpathy shows two ways to do the same thing — this is important for seeing that a neural network is learning to approximate a statistical distribution.

---

### Lecture 3 — Makemore Part 2: MLP
**Video:** [youtube.com/watch?v=TCH_1BHY58I](https://www.youtube.com/watch?v=TCH_1BHY58I)  
**Duration:** ~1.5 hours

Karpathy implements the Bengio 2003 MLP language model — the paper that launched neural language modeling. You build a multi-layer perceptron that uses learned character embeddings.

**What to look out for:**
- **Embeddings:** Each character is mapped to a learned vector. This is the same idea as word embeddings and, at a larger scale, token embeddings in GPT. Understand that embeddings are just a lookup table that gets trained.
- **The embedding space visualization:** When Karpathy plots the 2D embeddings, notice how similar characters cluster together. This is what "representation learning" means — the model learns geometry.

---

### Lecture 4 — Makemore Part 3: Activations, Gradients, BatchNorm
**Video:** [youtube.com/watch?v=P6sfmUTpUmc](https://www.youtube.com/watch?v=P6sfmUTpUmc)  
**Duration:** ~2 hours

This lecture goes deep into the practical difficulties of training neural networks — dead neurons, vanishing/exploding gradients, and BatchNorm.

**What to look out for:**
- **Saturated activations (dead neurons):** If the input to a `tanh` neuron is too large, the gradient is near zero — the neuron stops learning. This is called a "dead neuron." Karpathy shows how to diagnose this by looking at activation histograms.
- **Kaiming initialization:** A specific method for initializing weights that prevents gradients from vanishing or exploding at the start of training. CS336 expects you to know this.
- **BatchNorm:** Normalizes activations within a mini-batch. Understand what it does, why it helps, and why it is controversial (it has weird behavior at inference time vs. training time).

---

### Lecture 5 — Makemore Part 4: Backprop From Scratch (Manual)
**Video:** [youtube.com/watch?v=q8SA3rM6ckI](https://www.youtube.com/watch?v=q8SA3rM6ckI)  
**Duration:** ~2 hours

Karpathy manually implements the backward pass through BatchNorm without using PyTorch's autograd. This is the hardest lecture in the series.

**What to look out for:**
- Work through every gradient derivation on paper before watching Karpathy's solution. This is a graduate-level exercise.
- The goal is not to memorize the derivations. The goal is to get comfortable with the process of deriving gradients manually — because CS336's assignments require you to do exactly this.

---

### Lecture 6 — Makemore Part 5: WaveNet
**Video:** [youtube.com/watch?v=t3YJ5hKiMQ0](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)  
**Duration:** ~1 hour

Karpathy rebuilds the architecture using a tree-like CNN structure (WaveNet). He also dives into PyTorch internals — `nn.Module`, parameters, etc.

**What to look out for:**
- This is your first serious use of `nn.Module`. Understand how `__init__`, `forward`, and `parameters()` work — these are the three methods you will use in every model you build in CS336.

---

### Lecture 7 — Build GPT from Scratch
**Video:** [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
**Duration:** ~2 hours

This is the most important lecture in the series. Karpathy implements a character-level GPT (transformer) from scratch. Every component of the modern transformer is built and explained:
- Token and positional embeddings
- Self-attention (scaled dot-product attention)
- Multi-head attention
- Feed-forward layers
- Layer norm
- Residual connections
- The full training loop

**What to look out for:**
- **Self-attention mechanics:** The query, key, value matrix multiplication is the core of the transformer. Understand *why* Q·K^T gives attention weights: it is measuring similarity between every pair of tokens.
- **The causal mask:** Language models can only look at past tokens, not future ones. The mask implements this constraint. Understand how a triangular matrix achieves this.
- **Positional encoding:** Transformers have no notion of order by default (unlike RNNs). Positional encodings inject position information. Understand why this is necessary.
- **After this lecture:** You should be able to write a working GPT from scratch, from memory, with only PyTorch. If you can do this, you are genuinely prepared for CS336.

---

### Lecture 8 — GPT Tokenizer from Scratch
**Video:** [youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE)  
**Duration:** ~2 hours

Karpathy implements Byte Pair Encoding (BPE) — the tokenization algorithm used by GPT-2, GPT-4, and essentially all modern LLMs. CS336's first lecture is specifically on tokenization.

**What to look out for:**
- **Why do we need tokenizers at all?** Characters are too granular (too many steps). Words are too vocabulary-constrained (can't handle new words). BPE finds a middle ground.
- **Merge rules:** BPE works by iteratively merging the most frequent pair of tokens. Understand the algorithm step-by-step — you will implement your own tokenizer in CS336.
- **Why tokenization causes bugs:** Karpathy's infamous example — GPT-3 couldn't count the letters in "strawberry" because of how it was tokenized. CS336 discusses tokenization pitfalls in detail.

---

## Phase 5 — PyTorch Fundamentals

**Link (official tutorials):** [pytorch.org/tutorials](https://pytorch.org/tutorials)  
**Specific starting point:** [pytorch.org/tutorials/beginner/basics/intro.html](https://pytorch.org/tutorials/beginner/basics/intro.html)

After Karpathy, you know how PyTorch works conceptually. Now you need to be fluent in its API. CS336 uses raw PyTorch with no wrappers, and the assignments require using advanced PyTorch features.

**Estimated time: 3–4 weeks (run parallel with late Phase 4)**

**What to cover:**

- **Tensors:** Creation, shape manipulation (`view`, `reshape`, `permute`, `squeeze`, `unsqueeze`). Shape bugs are the most common bugs in ML code. You need to be able to reason about tensor shapes before writing a single line of a model.
- **Autograd:** `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`. Understand when gradients are computed and when they aren't (inference mode vs. training mode).
- **`nn.Module`:** Writing custom modules, the `forward()` method, `parameters()`, `state_dict()`.
- **Optimizers:** `torch.optim.Adam`, `torch.optim.SGD`. Know the difference between them. Adam is used in virtually all LLM training.
- **DataLoaders:** How to load and batch training data efficiently.
- **Device management:** Moving tensors between CPU and GPU with `.to(device)`. CS336 requires GPU awareness.

**What to look out for:**
- **In-place operations:** PyTorch sometimes silently breaks autograd if you use in-place operations (like `tensor += 1`) on tensors that require gradients. Know when to avoid them.
- **Gradient accumulation:** If you call `.backward()` twice without zeroing gradients with `optimizer.zero_grad()`, gradients accumulate. This is a common bug and also an intentional technique for large-batch training.

---

## Phase 6 — Transformers and Tokenizers (Hugging Face NLP Course)

**Link:** [huggingface.co/learn/nlp-course/chapter1/1](https://huggingface.co/learn/nlp-course/chapter1/1)

This course bridges from "I built a GPT from scratch" to "I understand the state-of-the-art transformer ecosystem." It is completely free, no account required for reading, and runs in Google Colab.

**Estimated time: 4–5 weeks**

**Chapter-by-chapter guide:**

- **Chapters 1–4 (Transformer Models):** How to use pre-trained models from Hugging Face Hub. Understand the pipeline API, tokenizers, and model outputs. This gives you context for what CS336 is trying to *build* from scratch.
- **Chapter 5 (Datasets):** How to work with large datasets. CS336 has a full lecture on data — sources, filtering, deduplication. This chapter is your primer.
- **Chapter 6 (Tokenizers Deep Dive):** Goes into BPE, WordPiece, and SentencePiece tokenizers in detail. This directly supports CS336 Lecture 1.
- **Chapter 7 (Main NLP Tasks):** Token classification, sequence-to-sequence, causal language modeling. Focus on the causal language modeling section — this is what GPT does.

**What to look out for:**
- **The difference between `AutoTokenizer` and writing your own:** Hugging Face makes it easy to use a pre-trained tokenizer in two lines. CS336 makes you write one yourself. Use this course to understand what the pre-built one is doing *under the hood*.
- **Attention masks in the Hugging Face API:** When you call a tokenizer on a batch of sequences of different lengths, it pads shorter sequences and gives you an attention mask (1 for real tokens, 0 for padding). Understand why the model needs this mask.

---

## Phase 7 — Stanford CS336 Lectures

**Course website:** [cs336.stanford.edu](https://cs336.stanford.edu)  
**YouTube playlist (Spring 2025):** [youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)  
**Lecture code/slides (Spring 2025 GitHub):** [github.com/stanford-cs336/spring2025-lectures](https://github.com/stanford-cs336/spring2025-lectures)

If you have completed Phases 0–6, you are ready to follow these lectures. They are posted publicly on YouTube and are completely free to watch.

The course website now shows a newer offering (Spring 2026). This guide links the Spring 2025 playlist and materials, which are still posted. Check the course website for newer videos when you reach this phase. Lecture topics and numbering may differ between years.

**Estimated time: 12+ weeks to follow seriously**

**Lecture overview (Spring 2025):**

| Lecture | Topic | What You Need from Prior Phases |
|---|---|---|
| Lecture 1 | Overview + Tokenization | Karpathy Lecture 8, HF Chapter 6 |
| Lecture 2 | PyTorch, Resource Accounting, FLOPs | Phase 5 (PyTorch), Phase 4 (Karpathy) |
| Lecture 3 | Transformer Architectures, Hyperparameters | Karpathy Lecture 7 |
| Lecture 4 | Attention Variants, MoE | Karpathy Lecture 7 + research paper reading |
| Lectures 5–6 | GPU Hardware, Triton Kernels, CUDA | Phase 5 + new material |
| Lectures 7–8 | Distributed Training (DDP, FSDP, pipeline) | Phase 5 + CS336 Lecture 2 |
| Lecture 9 | Scaling Laws (Chinchilla) | Phase 2 + probability intuition |
| Lecture 10 | Inference, KV-Cache, Flash Attention | Karpathy Lecture 7 deep understanding |
| Lecture 11 | Evaluation | Phase 2 intuition |
| Lecture 12 | Data — Sources, Filtering, Deduplication | HF Chapter 5 |
| Lecture 13 | Supervised Fine-Tuning (SFT) | All prior phases |
| Lectures 14–15 | RLHF, RLVR (how ChatGPT is aligned) | Hugging Face RL course bonus material |
| Lectures 16–17 | Multimodality, Mixture of Experts | Lecture 4 + CS336 lecture 3 |

**What to look out for in CS336:**
- **Triton kernels:** Lectures 5–6 introduce writing custom GPU kernels in Triton. This is genuinely difficult and requires comfort with GPU memory hierarchies. If this is unclear, watch the GPU section of the CS231n Stanford course as supplementary material ([youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)).
- **Scaling laws:** Lecture 9 is about the Chinchilla paper — the empirical finding that model size and data quantity need to be co-scaled. This is one of the most influential findings in LLM development.
- **The assignments require paid GPU compute** (AWS/GCP, ~$5–7/hour). For the goal of *understanding* the course, you can follow the lectures, read the assignment code, and use Google Colab's free tier for smaller experiments. The assignments are publicly posted at [github.com/stanford-cs336/spring2025-assignments](https://github.com/stanford-cs336/spring2025-assignments).

---

## Realistic Timeline

| Phase | Content | Duration |
|---|---|---|
| 0 — Math gaps (Linear Algebra, Calculus skim + multivariable, Probability review) | 3Blue1Brown + Khan Academy + MIT problem sets | 3–4 weeks |
| 1 — Python Depth | CS50P + NumPy fundamentals | 3–4 weeks |
| 2 — ML Foundations | Google MLCC | 2–3 weeks |
| 3 — Practical Deep Learning | fast.ai | 8–10 weeks |
| 4 — Karpathy Zero to Hero | All 8 lectures, fully coded | 12–16 weeks |
| 5 — PyTorch API | Official PyTorch tutorials | 3–4 weeks (parallel with late Phase 4) |
| 6 — Hugging Face NLP Course | Chapters 1–7 | 4–5 weeks |
| 7 — CS336 Lectures | Full Spring 2025 YouTube playlist | 12+ weeks |

**Total: roughly 13–17 months of work, spread over a longer calendar because of the DP.**

These are estimates for a pace of about 6–8 hours per week. Your own pace will differ, and that is fine. Treat this as a long-term project, not a sprint.

---

## Pacing Around the DP

Three HL subjects are a heavy load, and DP2 is the heaviest year, with the Extended Essay, Internal Assessments for each subject, and final exams. A realistic plan:

- **DP1 (now to the end of the school year):** Phases 0, 1, and 2, at 3–5 hours per week. Start Phase 3 only if your workload allows. Skip it in exam weeks.
- **Summer between DP1 and DP2:** Phase 3 and the start of Phase 4. Long free blocks suit Karpathy's videos.
- **DP2 until exams:** Light maintenance only. One Karpathy lecture or one coding session per week at most. Exams come first.
- **After exams:** Finish Phase 4, then Phases 5, 6, and 7.

Adjust this to your own timetable. If a deadline is close, pause the roadmap. Nothing here expires.

---

## Using the IB Core for This Work

Your AI learning can feed your DP requirements. Check every idea with your teacher or coordinator first. Your school and the IB set the rules, and an idea must fit the subject guide.

- **Extended Essay:** The EE subject must be one of your own DP subjects. You do not take Computer Science, so a Math EE is the natural fit. Choose a narrow question on the math behind tokenization, gradient descent, or attention, and answer it with your own calculations and small experiments.
- **Math Internal Assessment:** Explore the math behind backpropagation or gradient descent. Show your own calculations and a small worked example in code.
- **Physics and Chemistry Internal Assessments:** Python is a strong tool for data analysis and curve fitting in your own experiments. NumPy from Phase 1 is enough to start. Follow your school's rules on tools and on what counts as your own work.
- **CAS:** Teach a younger student basic Python or ML ideas. Share this roadmap with your school's coding club.
- **Portfolio:** Keep your code on GitHub. It shows sustained, independent work.

---

## Common Pitfalls to Avoid

1. **Watching without coding.** Watching Karpathy without writing the code yourself is almost worthless. You will feel like you understand and then be unable to implement anything. Code every lecture.

2. **Moving forward with unresolved confusion.** If you don't understand the chain rule in Phase 0, Lecture 1 of Karpathy will be mysterious. If Lecture 1 is mysterious, Lecture 7 will be incomprehensible. Every phase builds on the last. Stop and resolve confusion before advancing.

3. **Treating fast.ai as a library course.** fast.ai is teaching you *concepts*. The `fastai` library itself is not what you are learning — you are learning deep learning. The library is just the vehicle.

4. **Assuming Math AA HL is enough.** It gives you a strong base, but it does not teach matrices or multivariable calculus. Do not skip those parts of Phase 0, because Phase 4 depends on them.

5. **Expecting to do CS336's graded assignments for free.** The compute required for the actual assignments costs real money. The lectures, lecture code, and assignment specifications are all free. You can learn everything CS336 teaches without completing the assignments — but you cannot shortcut the compute cost if you want to actually run the training jobs.

6. **Letting this crowd out your DP work.** With three HL subjects, your grades and deadlines come first. A roadmap that costs you sleep or predicted grades is a bad trade. Use the pacing plan and slow down when school is busy.

7. **Comparing your pace to others online.** You are doing this on top of IB coursework. Someone who posts "I completed Karpathy in 2 weeks" is either experienced already, doing it poorly, or not in school full-time. Set your own pace.
