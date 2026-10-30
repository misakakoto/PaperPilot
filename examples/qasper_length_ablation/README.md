# Bounded research-loop acceptance case

This approved executable compares BM25 b=0 and b=0.75, fixing k1=1.2,
tokenization, IDF log(N/df), candidate paragraphs, tie order and top-k=3.
It samples 12 papers from sorted QASPER-dev IDs with each registered seed.
Samples can overlap: they are diagnostic slices, not independent training runs.
No training, hyperparameter search or official QASPER evaluation is performed.

Scoring: best-annotator exact-paragraph text Recall@3, macro-averaged over
questions with at least one text evidence annotation. Figure-only and
unanswerable-without-text questions are excluded and counted. Missing gold
paragraphs stay in the denominator; ranking failures count zero. No gold enters
the ranking function. Every scored question and ranking is retained in metrics.

The dataset must be copied into an isolated source staging directory alongside
experiment.py. Do not put the dataset in Git. Docker runs only that copy.

Scientific motivation (not evidence of improvement on this dataset):
[Introduction to Information Retrieval, BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)
describes term-frequency saturation and document-length scaling; b=0 disables
length normalization. These mechanisms motivate testing a length ablation,
but do not establish which setting is better on QASPER.
[QASPER](https://aclanthology.org/2021.naacl-main.365/) collects questions over
research papers and separate answer/evidence annotations. The local dev set has
already been used in project development and must not be called held out.
