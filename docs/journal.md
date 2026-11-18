# journal.md: the human's own notes

This file is **yours**. `CLAUDE.md` forbids the agent from writing your reasoning
here, so nothing below is filled in for you.

Use it for:
- **Predictions before running** anything that prints a number (bandwidth,
  latency, whether Triton beats torch). Write the number and one sentence of
  reasoning *first*, then run. A wrong prediction that you can explain is worth
  more than a right one you guessed.
- **Explanations in your own words** after each chunk: what the kernel does, why
  it maps threads to data the way it does, what the bottleneck is.
- **Questions you still cannot answer**, which become the next chunk.

Suggested entry shape:

```
## YYYY-MM-DD - <chunk name>
Prediction:
Result:
Why (my words):
Still unclear:
```
