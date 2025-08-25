---
layout: post
title: Releasing Neucodec
---

This week at Neuphonic we’re releasing our first open-source model to the community - [neucodec](https://huggingface.co/neuphonic/neucodec) - the neural audio codec that powers our TTS models. 

Having spent the majority of the last 6 months heading up the training and experimentation with this framework (which has apparently culminated in 166 training runs, according to my log…) I’m really excited we’re releasing some of the work. 

Props to our whole team for putting our heads together to get this wrapped up in a nice little bow in just a couple of days last week!

Anyway, we’re releasing two compatible versions of the model, [neucodec](https://huggingface.co/neuphonic/neucodec) and [distill-neucodec](https://huggingface.co/neuphonic/distill-neucodec). The TL;DR highlights are:
- FSQ quantisation resulting in a single codebook, making it ideal for downstream modeling with Speech Language Models.
- Trained with CC data such that there are no Non-Commercial data restrictions.
- At 50 tokens/sec and 16 bits per token, the overall bit-rate is 0.8kbps.
- The codec takes in 16kHz input and outputs 24kHz using an upsampling decoder.
- The FSQ encoding scheme allows for bit-level error resistance suitable for unreliable and noisy channels.
- We have a distilled-encoder (as the original encoder was SLOW) to facilitate training downstream models in encoding-heavy regimes (like ASR).

We're currently writing a paper about about some interesting properties of the framework, so stay tuned.

Thanks to the authors of [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2), [SQCodec](https://arxiv.org/abs/2504.04949) and [Vocos](https://arxiv.org/abs/2306.00814) who's work underpins and inspires a lot of the work we've done with this. 