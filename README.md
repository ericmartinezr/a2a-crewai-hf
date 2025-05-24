## A2A Learning Project

### A2A

https://google.github.io/A2A/

Mostly based on CrewAI example

https://github.com/google/A2A/tree/main/samples/python/agents/crewai

### CrewAI

https://docs.crewai.com/introduction

The LLM model used is Gemma3 (4b) with Ollama

### Huggingface

Used Stable Difussion 3.5 Large model
https://huggingface.co/stabilityai/stable-diffusion-3.5-large

This model was used to generate the image in the tool called by the CrewAI Agent.

---

## Execution

### Server

```python
source .venv/bin/activate

uv run . --host localhost --port 9999
```

### Client

```python
source .venv/bin/activate

uv run test_client.py
```

---

## Result

![Cute Cat](images/Purrfect%20Companion.png)

---

The purpose of this project is to learn about all these new technologies.
