---
layout: post
title: What I read in June 2023
---

I originally saw this format of blog post on [kipply's blog](https://kipp.ly/blog/) and felt the overwhelming desire to steal it and make it my own. In June I, as well as seemingly every linkedIn-fluencer in existence, have been familiarising myself with a whole host of topics related to using and deploying LLMs.
\
\
I read Kobzol's guide to [writing python like it's rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html) - this is by far the best blog post I've read in a while. Rust changed the way I write python, and every suggestion felt like the author had been privy to my own _a-ha_ moments whilst comparing the two languages. I'm especially fond of the section on dataclasses
\
\
Kipply's guide on [estimating inference time with transformers](https://kipp.ly/blog/transformer-inference-arithmetic/) is a fantastic and detailed guide to all of the hardware and software considerations that surround the deployment of transformers. I'm especially fond of all of the cursory formulas for (im)precisely estimating time complexity with different hardware
\
\
Horace He's article on [speeding up models](https://horace.io/brrr_intro.html) provides a concise overview of the overhead and bottlenecks you can face then trying to make models go as fast as possible
\
\
Misha Laskin's post on [tensor parallelism](https://www.mishalaskin.com/posts/tensor_parallel) made model sharding 'click' for a simple first principles description and a bit of code to help along the way.
\
\
I listened to an [older podcast](https://signalsandthreads.com/python-ocaml-and-machine-learning/) hosted by Ron Minksy on the interplay between Python and OCaml at Jane Street - this got me interested in Jane Street's practise of using _expect tests_ - tests that essentially write themselves, given a smaller set of instructions. Find out more [here](https://blog.janestreet.com/the-joy-of-expect-tests/).
\
\
Huyen Chip's [introduction to machine learning compilers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html) is a great overview of compilers for ml and how they work.
\
\
I also read the [LoRA paper](https://arxiv.org/abs/2106.09685) for the first time and was taken aback by the overall simplicity and elegance of the idea.