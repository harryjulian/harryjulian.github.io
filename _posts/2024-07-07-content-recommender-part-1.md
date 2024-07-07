---
layout: post
title: Building a technical content recommender, part 1
---

I hate trawling the internet to try and stay up to date with the breakneck speed at which AI has been moving for the past couple of years - there's simply too much content. X amount of arxiv papers published everyday. It's exhasuting and information overload.

My old go-to for this sort of thing was Twitter (now X), where I'd be presented with a nicely written thread on some sort of new or obscure topic that I was interested in - but now I'm only presented with new flavours of increasingly obscure propoganda. 

I've recently decided to try and minimize doom-scrolling...

Creating my own recommender seemed appropraite

# Aims

### v0.1
- Pull recommendations from latest content whenever I run the system
- Send these recommendations to a service I have access to
- Be free to run and thus should probably run on my laptop (courtesy of MLX)
- Be able to integrate a written prompt of my preferences for technical content
- Have a way of collecting preference data for recommendations for eventual fine-tuning

### v0.2 and beyond
- Integrate my database of previously read pieces of content with
- Fine-tune a model on previously read content
- Fine-tune on preference data

# Building the engine

My initial idea for how the system should work.

## Pulling latest content 

## Rating and Ranking with an LLM

## Serving Recommendations

## TODO: