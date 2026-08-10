---
name: msmarco-passage-experiments
description: This skills shows how to conduct experiments the MSMARCO passage corpus on PyTerrier. This also includes the TREC Deep Learning track topics sets 2019 and 2020 (often abbreviated as DL'19 and DL'20).

---

Use PyTerrier's `pt.Experiment` formulation for conducting experiments. Assuming your have two systems/pipelines, `systemA` (which may be your baseline) and `systemB`:

```python

from pyterrier.measures import *
pt.Experiment(
    [
        systemA,
        systemB,
    ]
    pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged').get_topics(),
    pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged').get_qrels(),
    [ MAP(rel=2), nDCG@10 ]
)
```

The relevant datasets and corresponding measures are
 * Dev set: 'irds:msmarco-passage/dev/judged', using MRR measure
 * TREC DL'19: 'irds:msmarco-passage/trec-dl-2019/judged', using nDCG@10 and MAP(rel=2) 
 * TREC DL'20: 'irds:msmarco-passage/trec-dl-2019/judged', using nDCG@10 and MAP(rel=2) 


TREC Deep Learning conventions:

- **nDCG@10** uses the **graded** labels (0–3).
- **MAP and MRR** binarise at **rel ≥ 2**. Using rel ≥ 1 inflates both, and is a common silent error.
- MAP is usually reported at depth 1000 and is recall-sensitive — a truncated candidate pool caps it even
  when nDCG@10 looks fine.

Match the paper's metrics and cutoffs exactly before comparing anything.
