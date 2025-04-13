---
layout: post
title: Learning how to do machine learning research
---


I started my first research role at the end of last summer. It’s been a whirlwind, and I think I’ve probably learned more in the past eight months than I have in the rest of my professional experience combined. Though it's been fun, it's has also served me yet another slightly brutal reminder that learning is rarely painless. I thought it might be nice to document some of the most important lessons I've learned (usually the hard way) thus far.

---

## Don’t trust every(any)thing you see on Arxiv

It's quite apparent that everything in the ML space has been drowning in hype since the ChatGPT moment. But even if we ignore linkedInfluencer posts and Sam Altman's snake oil, I was still quite surprised to find that the level of rigour in a lot of can be academic research quite poor and that inflating numbers and cherry-picking results seems rife. I think this might be especially true in TTS, where authors can pretty easily massage a model into producing some good samples; I've more often that not been under-whelmed by new releases and more often that not it's surprisingly easy to find a simple test case can actually be a good enough adversarial example to create genuinely demonic noices. Robustness is often ignored in favour of shiny results, and failure modes are rarely discussed. I've found it's very important to validate interesting results yourself; your own evals should be your source of truth, most papers should be very much just a source of inspiration. 

---

## Attention (to detail) is all you need

I don’t have any sort of formal academic research background. I’m completely self-taught as a programmer and have learned everything I know about ML either on the job or knee-deep in textbooks (thanks [Kevin Murphy](https://probml.github.io/pml-book/book1.html)!). This led me to develop a somewhat scrappy mentality; I'm definitely good at learning by doing,  failing first and failing fast. But, as I was always interested in implementing and getting my hands dirty, I'd instinctively only dig into the lowest level of information necessary to accomplish the side-quest presently at hand. This is a great way to independently learn on your own, I think there's only so much time in a given day I can churn thorugh a textbook or course lectures. But I've found this approach doesn’t always work when you’re doing research, unsurprisingly. 

It's obviously incredibly useful sometimes to have the chops to just churn through an implementation, but pace and attention to detail are usually inversely proportional to one another (moderated by experience, which allows you to know *where* to focus your attention to detail…), and I've painfully discovered that missing out on a minute detail can waste a week of misplaced effort in the wrong direction.

To counter this I'm mainly trying an approach where I make no assumptions about any of my code, and force myself to write tedious and obvious `assert` statements, mainly in training scripts, but also with anything else where correctness is paramount. Little things like `did i properly shuffle my dataset?` are easy to assume you did, but let's say you make a mistake writing that line 1% of the time; if your training code is 600 lines long, then you might have 6 mistakes in there - and who knows how subtle they might be. Obvious mistakes are great! Subtle mistakes? The one that take weeks to discover and days to debug? Disasterous!

---

## Simple isn't always simple at scale

Modern ML models are massive and they're trained on on huge amounts of data. In my domain, TTS, most open-source models are now trained on at least ~50k hours (usually ~3tb) of audio. Processing these datasets, even for simple operations like resampling, becomes time-consuming. Say you want to transcribe everything to make sure the text aligns with the audio, or apply other neural models like upsamplers? You probably need 4 days and a couple of GPUs. Planning ahead with data processing has actually become an essential part of my day job - I've become pretty good at estimating how long these jobs can take. Experiments can't run without their requisite datasets! It's important to note, that though it might take an extra hour implementing gritty optimisations in your processing pipelines upfront, and this usually feels very frustrating as you want to get the pipleline running ASAP because it takes so long, the speed gains you realise down the line can end up adding up to dozens of hours of compute capacity spared. It's worth doing the optimisation up front! 

---

## Sometimes you need to start from scratch

Implementing ideas or ideas that exist without the accompanying code is costly in terms of time. Though sometimes, it’s necessary. 

Where ideas are already implemented, it’s about making the decision as to how much of a codebase you’re willing to trust. Open source is fantastic, but smaller open source projects (particular model implementations, rather than PyTorch) have been written by someone equally as fallible as yourself, and you won’t know the sharp edges of a codebase until they’ve bitten you. 

When they aren’t implemented, you really need to weigh up how good the idea actually might end up being vs how long it’s going to take. If it’s the full model + training code from a paper, expect to spend time wrestling *lots* of bugs (unless you're truly cracked, then carry on). If it's too promising not to do, get your hands dirty with the impl and share the burden with as many teammates as possible - it'll be worth it!