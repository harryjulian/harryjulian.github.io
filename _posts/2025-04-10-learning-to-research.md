---
layout: post
title: Learning how to do machine learning research
---


I started my first research role at the end of last summer. It’s been a whirlwind, and I think I’ve probably learned more in the past eight months than I have in the rest of my professional experience combined. This has also served me yet another reminder that the process of learning is rarely painless. I thought it might be nice to document my experience thus far learning how to do research and run through some of my failures, summarised into a collection of aphorisms. 

---

## Don’t trust every(any)thing you see on Arxiv

The ML space is drowning in hype. People inflate their numbers and cherry-pick results. Having run evals on countless models / frameworks claiming to beat the SOTA at something, more often than not I’m surprised to find that a simple test case can end up being an adversarial example. Robustness is often ignored in favour of shiny results, and when you’re working on models that are going to end up as a product, all you think about all day is robustness. It's important to not take most academic results you see at face-value. Even if you can believe the results, they're still usually only ran on common benchmark datasets. Learn to distinguish grifting from the real deal!

---

## Attention (to detail) is all you need

I don’t have any sort of formal academic research background. I’m completely self-taught as a programmer and have learned everything I know about ML either on the job or knee-deep in textbooks (thanks [Kevin Murphy](https://probml.github.io/pml-book/book1.html)!). This has led to be to develop the something akin to the ‘hacker’ mindset; to move fast and break things, and instinctively initially only dig into the lowest level of information necessary to accomplish the side-quest at hand.

This doesn’t always work when you’re doing research, unsurprisingly. It can be incredibly useful sometimes to just churn through an implementation, but pace and attention to detail are usually inversely proportional to one another (moderated by experience, which allows you to know where to focus your attention to detail…). Missing out on a minute detail can waste a week of misplaced effort in the wrong direction.

ML problems can often be solved through different lenses; some (if not most) problems are related to the data distribution, sometimes your model architecture could be the limiting factor. Sometimes a problem might just require pure engineering effort, be it on the software or hardware side. Having ample a priori visibility of the problem and knowing how your tools work (and their limitations) is a cheat code to choosing the right approach.

---

## Simple isn't always simple at scale

Modern ML models are massive and they're trained on on huge amounts of data. In my domain, TTS, most open-source models are now trained on at least ~50k hours of audio, with notable exceptions of course. Processing these datasets, even for simple operations like resampling, becomes time-consuming. Say you want to transcribe everything or apply other neural models? You probably need 4 days and a couple of GPUs. Planning ahead with data processing has actually become an essential part of my day job - I've become pretty good at estimating how long these jobs can take. Experiments can't run without their requisite datasets. 

It's important to note, that though it might take an extra hour implementing gritty optimisations in your processing pipelines upfront, the speed gains you realise down the line can end up adding up to dozens of hours of compute capacity spared.

---

## Sometimes you need to start from scratch

Implementing ideas or ideas that exist without the accompanying code is costly in terms of time. Though sometimes, it’s necessary. 

Where ideas are already implemented, it’s about making the decision as to how much of a codebase you’re willing to trust. Open source is fantastic, but smaller open source projects (particular model implementations, rather than PyTorch) have been written by someone equally as fallible as yourself, and you won’t know the sharp edges of a codebase until they’ve bitten you. 

When they aren’t implemented, you really need to weigh up how good the idea actually might end up being vs how long it’s going to take. If it’s the full model + training code from a paper, expect to spend time wrestling *lots* of bugs (unless you're truly cracked, then carry on). If it's too promising not to do, get your hands dirty with the impl and share the burden with as many teammates as possible - it'll be worth it!