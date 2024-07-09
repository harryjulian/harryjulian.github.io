---
layout: post
title: Building a Technical Content Recommender, Part 1 - The System
---

I enjoy reading well-written technical content whenever and wherever it presents itself to me. However, keeping up with the breakneck speed at which machine learning has been moving for the past two years can be little short of exhausting sometimes (for example: the amount of arXiv papers published daily about ML/AI [roughly doubles every two years)](https://x.com/MarioKrenn6240/status/1577102743927652354). 

Keeping up can be a chore, and it can be hard to distinguish the signal from the noise when choosing something to read. Old twitter used to be fantastic for this sort of thing: I was often recommended nice, concise threads about fascinating machine learning subtopics I'd never heard of. Unfortunately, now, I'm only presented with new flavours of increasingly obscure propoganda. Thanks Elon!

This problem inspired me to set out on trying to create a personalised technical content recommender, just for me.


# The Plan

First things first, I thought I'd get cracking on building out the mechanical elements of the system before I really consider how I'd be rating and ranking recommendations - mainly because I hadn't yet had a chance to read any literature about text recommendation systems.

To start with, I laid out some basic requirements:

- The system should be efficient enough to run from my M1 Macbook (using MLX and my GPUs?).
- It should pull recommendations from latest content, rate and rank them, then send the top N rated articles to my email.
- When rating new content, it will consider: i) the articles I've previously read and ii) a written prompt describing the type of articles I want to read going forwards.
- Have a way for me to collect preference data about the recommended articles - and then a means of ingesting this back into the system to update datasets and improve models. 

The first 3 requirements are somewhat straightforward, encompassing writing some web crawlers for candidate generation, devising a quick model to rate the content given my preferences and previously read content, and then finding a way to push them into an external system. 

When it comes to serving the recommendations, I chose to ensure they're sent to an **external system** so that I can access them on the go - as this is when I do the bulk of my reading. I'd initially considered embedding a form into the email I was going to send to my main email address, but discovered this is a pain for a few reasons. I did consider using google forms, but the Python API appeared to be needlessly complicated and it didn't really matter what service I chose, as long as it did what I wanted it to do.

I ended up going with Airtable as the free tier seemed to cover what I wanted from a service. I guess we'll find out if that was a good decision or not.


<style>
  .image-container {
    text-align: center;
  }
  .image-container figure {
    margin: 0;  /* Remove default figure margin */
  }
  .image-container img {
    max-width: 50%;
    height: auto;
  }
  .image-container figcaption {
    font-style: italic;
    color: #666;  /* Gray color for the caption */
  }
</style>

<div class="image-container">
  <figure>
    <img src="/assets/images/recommender-service.png" alt="Description of the image">
    <figcaption>A hacked together system diagram.</figcaption>
  </figure>
</div>

## Pulling latest content

I thought I'd start with writing crawlers for `arXiv` and `hackernews`, as these seemed like low-hanging fruit.

I wrote a single-threaded crawler at first, which was (unsuprisingly) slow. Adding multithreading was as easy as using `concurrent.futures.ThreadPoolExecutor`, and my issue was solved. Here's the basic crawling function for any arbitrary page - I specialised the functions for `arXiv` and `hackernews` to use their public APIs for a ease of use, but haven't included them to keep this concise. I defined an `ABSTRACT_SIZE` constant which determined how many characters would be saved as 'content', for efficiency when passing all of the articles through the model - this value could be too short at the moment, but it's just a placeholder for development. There's also a distinct lack of retry logic and error-handling, which I should probably sort out...

```python
from typing import Optional
import concurrent
import requests
import json
from bs4 import BeautifulSoup
from dataclasses import dataclass


ABSTRACT_SIZE = 512 # Determines how much of the content we're saving into the object to be passed to our model


@dataclass(slots=True)
class IndexItem:
    title: str
    url: str
    content: str
    rating: Optional[float] = None


def item_from_soup(soup: BeautifulSoup, url: str, abstract_size: int = ABSTRACT_SIZE) -> IndexItem:
    title = soup.title.string if soup.title.string else "untitled article"
    page_content = " ".join(soup.stripped_strings)[:abstract_size]
    item = IndexItem(
        title=title,
        url=url,
        content=page_content
    )
    return item


def crawl_pages(urls: list[str], max_workers: int = 10) -> list[IndexItem]:
    """Given a list of urls, retrieve and parse content."""

    def _crawl_page(url):
        try:
            page_response = requests.get(url) # retry logic or error handling pls
            soup = BeautifulSoup(page_response.content, "html.parser")
            item = item_from_soup(soup, url)
            return item
        except:
            print(f"Failed for {url}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        articles = list(executor.map(_crawl_page, urls))
        
    return articles
```

## Rating and Ranking with an LLM

I had a few ideas when it came to a first approach for rating articles.

- A: start with `sentence_transformers`, embed each previously read article and then compute a weighted rating for new articles based on i) similarity to my written preferences and ii) similarity with my previously read articles.
- B: try zero-shotting a quantised LLM with constrained generation, courtesy of `outlines`, it's support for `MLX`, and the handy GPU in my M1 Mac.

I played around with embeddings for a little while, trundled through a load of VectorDB documentation and eventually threw it all in the bin for this simple zero-shot LLM rating mechanism. That was a little bit of a mistake. I failed to get a few different models to solely generate either constrained probabilities or ratings between 0 and 100, even with regex constrained generation (working theory: something to do with the tokenization, or lack of, of individual numbers?). I moved to the `generate.json` method, which works a little better, but only with a model as large as a 4-bit quantized `Phi-3-mini`. I'll be replacing rating mechanism in the next article, but oh well, I wanted to crack through writing the engine and this does provide us with a proxy of a rating for now. I'm keen to keep this setup to see if I can improve a smaller model on this task with some fine-tuning.


```python
from pydantic import BaseModel, conint
from outlines import prompt, generate


# We use this to slightly improve the LLM performance
class RelevanceRating(BaseModel):
    rating: conint(ge=0, le=100)


@prompt
def _format_rating_prompt(user_preferences: str, content_abstract: str):
    """Your role is to recommend curated technical content to users based on their explicit preferences.
    You will be provided with the user's request, outlining the type of technical content they are interested
    in receiving and a piece of technical content scraped from the internet. Leave a relevance rating for the piece of content, based on how likely the user is to want to read it. Though the rating should reflect overlap between their topics of interest, don't be afraid to recommend great pieces slightly outside of these preferences if the style of the piece overlaps. The rating value should be between 0 and 100.
    
    User Preferences: {% raw %}{{ user_preferences }}{% endraw %}
    Content Abstract: {% raw %}{{ content_abstract }}{% endraw %}

    Answer with JSON, where the JSON should be a dictionary with the key "rating" between 0 and 100.
    """


def rate_content(user_preferences: str, content: str, model):
    """Provides LLM-generated rating given user preferences and piece of content."""
    try:
        return generate.json(model, RelevanceRating)(_format_rating_prompt(user_preferences, content))
    except:
        return None
```

For the time being I've kept the ranking component simple -- selecting the top N articles with the highest ratings.

```python
# Naive ranking based on scores, pull top n
sorted_index = sorted(index, key = lambda item: item.rating)
items_to_recommend = sorted_index[-RECOMMEND_TOP_N:]
```

## Serving Recommendations

Once we've rated the articles and selected those that will be recommended, we need to devise a way of getting them into some sort of table or form that we can pull preference data from when we need to. I defined a simple airtable schema which creates a table with the columns `title`, `url` and a `preference` column where I can leave a 👍 or 👎 preference rating. Everytime the system runs, we then create a new airtable labelled as `Recommendations {{ date }}`, push the data to the table, and then grab the `base` and `table` ids needed to generate a link to the table, of which we'll embed in the email.

```python
import pandas as pd
from pyairtable import Api


TABLE_SCHEMA = {
    "name": "recommendations",
    "fields": [
        {'name': 'title', 'type': 'singleLineText'},
        {'name': 'url', 'type': 'url'},
        {'name': 'preference', 'type': 'singleSelect', 'options': 
            {'choices': [
                    {'name': '👍', "color": "greenDark1"},
                    {'name': '👎', "color": "redDark1"},
                ]
            }
        }
    ]
}


def create_recommendation_airtable(items_to_recommend, date: str, api: Api) -> str:
    """Creates airtable to send to user and collect data from."""
    table_name = f"recommendations {date}"
    base = api.create_base(os.getenv("AIRTABLE_WORKSPACE"), table_name, [TABLE_SCHEMA])
    table = base.tables()[0]
    table = api.table(base.id, table.name)
    table.batch_create(
        [
            item.for_airtable() for item in items_to_recommend
        ]
    )
    return base.id, table.id
```

Fantastic. All we need to do is write a bit of html to embed our link into (thanks Claude) and send the email over SMTP.

```python
import os
import smtplib
from string import Template
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


AIRTABLE_BASE_URL = "https://airtable.com/"


def format_email(base_id: str, table_id: str, date: str) -> str:
    airtable_url = os.path.join(AIRTABLE_BASE_URL, base_id, table_id)
    template = Template("""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Today's Technical Content Recommendations, ${date}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; text-align: center; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #2c3e50;">Recommendations🚀</h1>
        
        <p>
            Our language model has curated a list of top recommendations, just for you.
        </p>
        
        <p>
            <a href=${airtable_url} style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                View Recommendations in Airtable
            </a>
        </p>
        
        <p>All the best,<br>Your Recommendation Engine</p>
    </body>
    </html>
    """
    )
    return template.substitute(date=date, airtable_url=airtable_url)


def send_email(content: str, date: str, sender_email: str, receiver_email: str, app_password: str) -> bool:
    
    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"Today's Technical Content Recommendations"

    # Add body to email
    message.attach(MIMEText(content, "html"))

    # Create SMTP session
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtpserver:
            smtpserver.ehlo()
            smtpserver.login(sender_email, app_password)
            # Convert the message to a string and send it
            smtpserver.send_message(message)
    except: 
        return False

    return True
```

And there we have an email, sent directly into my inbox.

<style>
  .image-container {
    text-align: center;
  }
  .image-container figure {
    margin: 0;  /* Remove default figure margin */
  }
  .image-container img {
    max-width: 50%;
    height: auto;
  }
</style>

<div class="image-container">
  <figure>
    <img src="/assets/images/recommendation-email.png" alt="Recommendation email.">
  </figure>
</div>

# Next Steps

I've got a nice functional recommendation system - which is failing to make m/any decent recommendations. We'll fix this in part 2.