---
layout: home
title: Home
---

Welcome to my personal website! Check out my blog posts below.

## Quick Links

- [About Me]({{ '/about/' | relative_url }}) - My background and what drives this journey
- [Notes]({{ '/notes/' | relative_url }}) - My research and project notes

## Recent Notes

<ul>
  {% assign sorted_notes = site.notes | sort: "title" %}
  {% for note in sorted_notes limit:5 %}
    <li>
      <a href="{{ note.url | relative_url }}">{{ note.title | default: note.basename }}</a>
    </li>
  {% endfor %}
</ul>