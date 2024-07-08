---
layout: post
title: Building a technical content recommender, part 1
---

I really enjoy reading well-written, interesting technical content; in order to keep up-to-date with the breakneck speed at which AI has been moving for the past couple of years, it's an essential task. However, distinguishing signal from noise can be difficult when you're deciding to spend your most valuable resource (time) consuming a piece of content, and occasionally wasting it.

There is simply too much choice. The amount of arxiv papers published daily about AI [roughly doubles every two years](https://x.com/MarioKrenn6240/status/1577102743927652354). My old go-to forcutting through the noise was Twitter, where I'd be presented with a nicely written thread on some sort of new or obscure topic that I was interested in - but now I'm only presented with new flavours of increasingly obscure propoganda. Thanks Elon!

Anyway, to the point: I decided to keep my content-fix fed and minimize my exposure to opportunites for doom-scrolling, I should build a recommender for technical content based on my own tastes. I want to design a basic system that gives me a bit of room to play with embeddings and langauge models to make recommendations, as well as potentially integrating some agentic candidate discovery down the line...

# Aims

I thought I'd get cracking on building out whatever mechanical code I needed to give me a chance to think about how I wanted to approach any training or modelling. I put together a little list of essentials for the first stage of the project:

- The service must be efficient to run from my M1 Macbook.
- The service should pull recommendations from latest content, rate and rank them, then send top N recommendations to my email.
- The service should to integrate a prompt describing the type of content I want to read.
- The service should be built to accomodate a way of recording preference data for future fine-tuning.
- The service should integrate information about previously read articles into it's recommendations.

# Building the engine

This was my intial idea for how the system was going to work. I'd originally thought I might be able to embed a form for rating preference data in the email but this seemed to be a bit troublesome (and generally a bad idea), so I went with Airtable instead route. I did consider a google form but I took a look at the python API and...refused.

![Rough Architecture Diagram](/assets/images/recommender-service.png)

## Pulling latest content

At the moment if I'm looking to keep up to date, I'll probably surf `hackernews` or `arxiv` to see if there's anything that catches my eye. I decided to write crawlers for these sites first. Unsurprisingly, the first single-threaded iteration of the crawlers were incredibly slow, having to wait for the respective APIs to repeatedly respond - adding a little bit of multithreading with `concurrent.futures` worked a dream as expected. Here's an excerpt, - I also wrote some other functions for hackernews and arxiv in particular as they have helpful APIs/packages associated with them.

```python
ABSTRACT_SIZE = 512

dataclass(slots=True)
class IndexItem:
    title: str
    url: str
    content: str
    rating: Optional[float] = None

    def for_airtable(self):
        """Convenience func to get necessary info from obj for airtable push."""
        return {
            "url": self.url
        }
    

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

I had a few ideas when it came to an initial approach for rating articles.

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

## Serving Recommendations

Once we've rated the articles and selected those that will be recommended, we need to devise a way of getting them into some sort of table or form that we can pull preference data from at some point. ...

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

And there we have an email, sent directly into our inbox!

![Rough Architecture Diagram](/assets/images/recommender-email.png)