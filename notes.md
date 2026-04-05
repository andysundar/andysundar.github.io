---
layout: page
title: Notes
permalink: /notes/
---

{% assign grouped_notes = site.notes | group_by: "topic" %}

{% for group in grouped_notes %}
  <h2>{{ group.name | default: "Other" }}</h2>
  <ul>
    {% assign sorted_items = group.items | sort: "title" %}
    {% for note in sorted_items %}
      <li>
        <a href="{{ note.url | relative_url }}">{{ note.title | default: note.basename }}</a>
        {% if note.description %} — {{ note.description }}{% endif %}
      </li>
    {% endfor %}
  </ul>
{% endfor %}