---
layout: post
title: Building a Technical Content Recommender, Part 1 - The System
---

There's not much I enjoy more than finding well-written and interesting technical content to read. However, keeping up with the breakneck speed at which machine learning has been moving for the past two years can be nothing short of exhausting (The amount of arxiv papers published daily about ml [roughly doubles every two years)](https://x.com/MarioKrenn6240/status/1577102743927652354). 

When you're choosing something to read it's hard to distinguish the signal from the noise. Old twitter used to be fantastic for this sort of thing, where I was often recommedned nice concise threads about fascinating machine learning subtopics I'd never heard of. Unfortunately, I'm now I'm only presented with new flavours of increasingly obscure propoganda. Thanks Elon!


# The Plan

First things first, I thought I'd get cracking on building out the mechanical elements of a personal recommendation system before I considered explicitly HOW I should be making recommendations - this was largely driven by the fact I'd never built a recommender before, so gave me a bit of time to do some reading and let my ideas ruminate whilst still making progress. 

To start with, I laid out some basic requirements, where the system should:

- Be efficient enough to run from my M1 Macbook.
- Pull recommendations from latest content, rate and rank them, then send top N recommendations to my email.
- When rating new content, it will consider: i) the articles I've previously read and ii) a written prompt describing the type of articles I want to read going forwards.
- Have a way for me to enter preference data about recommended articles - and then a means of ingesting this back into the system to update datasets and improve models. 

The first 3 requirements are somewhat straightforward: I need to write some web crawlers for candidate generation, initially for websites of interest or that I frequent, devise a model to rate the content given my preferences and previously read content, and then find a way to serve them. 

With regards to serving the recommendations, I wanted to send them to an external system that I can access on the go, as this is when I do the bulk of my reading. I'd initially considered embedding a form into the email I was going to send to my main email address, but found that embedding forms in emails isn't very easy or at all recommended. I did consider using google forms but the Python API seemed dreadfuul so I ended up making a bit of a snap decision to serve my recommendations in Airtable. 


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

At the moment if I'm looking to keep up to date, I'll probably surf `hackernews` or `arxiv` to see if there's anything that catches my eye - thus, I decided to write crawlers for these sites first. Unsurprisingly, the first single-threaded iterations of the crawlers were incredibly slow, having to wait for the respective APIs to respond sequentially - adding a little bit of multithreading with `concurrent.futures` worked a dream to speed this up, as expected. Here's an excerpt - I also wrote some other functions for hackernews and arxiv in particular as they have helpful APIs/packages associated with them.

```python
from typing import Optional
import itertools
import concurrent
import requests
import json
from bs4 import BeautifulSoup
from dataclasses import dataclass


ABSTRACT_SIZE = 512


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

I played around with embeddings for a little while, trundled through a load of VectorDB documentation and eventually threw it all in the bin for this simple zero-shot LLM rating mechanism. That was a little bit of a mistake. I failed to get a range of models to solely generate either constrained probabilities or ratings between 0 and 100 - even with structured generation (working theory: something to do with the way the tokens aren't trained for generating individual numbers?). I moved to this `generate.json` method, which works a bit better, but still not without a model as large as `Phi-3-mini`. I'll be replacing this in the next article, but oh well, I wanted to crack through writing the engine mostly and this does provide us with a proxy of a rating.


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
    
    User Preferences: {{ user_preferences }}
    Content Abstract: {{ content_abstract }}

    Answer with JSON, where the JSON should be a dictionary with the key "rating" between 0 and 100.
    """


def rate_content(user_preferences: str, content: str, model):
    """Provides LLM-generated rating given user preferences and piece of content."""
    try:
        return generate.json(model, RelevanceRating)(_format_rating_prompt(user_preferences, content))
    except:
        return None
```

For the time being I've left ranking incredibly simple -- selecting the top N articles with the highest ratings.

```python
# Naive ranking based on scores, pull top n
sorted_index = sorted(index, key = lambda item: item.rating)
items_to_recommend = sorted_index[-RECOMMEND_TOP_N:]
```

## Serving Recommendations

Once we've rated the articles and selected those that will be recommended, we need to devise a way of getting them into some sort of table or form that we can pull preference data from at some point. I defined a simple airtable schema which creates a table with the columns `title`, `url` and a `preference` column where I can leave a 👍 or 👎 preference rating. Everytime the system runs, we then create a new airtable labelled as `f"recommendations {date}"`, push the data to the table, and then grab the `base` and `table` ids needed to generate a link to the table of which we'll embed in the email.

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
def send_email(content: str, date: str, sender_email: str, receiver_email: str, app_password: str):
    
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

I've got a nice functional recommendation system - but currently failing to make many decent recommendations. Next, I think I'm going do some more reading and have a go at augmenting my existing dataset of previously read articles with some negative examples, and then think about doing some supervised fine-tuning.