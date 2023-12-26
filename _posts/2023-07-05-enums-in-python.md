---
layout: post
title: Why you should use Enums in Python
---

As [previously mentioned](https://harryjulian.github.io/04/07/2023/what-i-read-in-june-2023), I recently came across an excellent article by Kobzol called [writing python like it’s rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html) which details a range of ways in which we can use the rich, strongly-typed syntax of Rust as inspiration to write better Python.
\
\
Something that wasn’t mentioned in this article but I thought were worth mentioning are Python [enums](https://docs.python.org/3/library/enum.html). Enums, simply put, are enumerations of constants.
\
\
Enums don’t really add any additional functionality to Python, but they’re useful to enforce _correctness_ when writing code. When writing Rust, the compiler ensures that only valid states of the programme can be represented. Using enums can help enforce these valid programmatic states by forcing us to be precise about the possible values of our type.
\
\
For example, let’s consider the example of using compass direction and their related values in a programme. We could write out our constants as individual variables, but this is cumbersome. If we want to store them in one file and import them to another, we’d have to import each variable individually. There’s also no means of determining that the variables are related to one another. Blah blah blah, this is a shit approach.
```python
north = 0
east = 90
south = 180
west = 270 
```

A second approach might be to use a dictionary. This might be the standard ‘pythonic’ approach. It keeps all the related values accessible in an identifiable, relatable way. This is better, but i) we’re still using a standard dictionary that we can add arbitrary keys to and ii) I personally dislike the square-bracket string access method for dictionaries. It feels like I’m slotting in a rather precise key each time to access a value and auto-complete often feels slower (placebo? not sure).
```python
directions = {
  "north": 0,
  "east": 90,
  "south": 180,
  "west": 270,
}
```

The optimal approach here, in my opinion, is replacing this with an Enum. Here, we define all the possible values of our type and nothing more. The type can only be of this value.
```python
from enum import Enum

class Direction(Enum):
  North = 0
  East = 90
  South = 180
  West = 270

print(f"East is at {Direction.East.value}")
```

When someone else comes along to use your code and they see the function signature, they'll know exactly what the possible values for the direction argument are.
```python
def can_move_step(metres: int, direction: Direction) -> bool:
    ...
```

I played around with the [alpaca-py](https://github.com/alpacahq/alpaca-py) package a little bit recently and found the authors use of enums incredibly useful.