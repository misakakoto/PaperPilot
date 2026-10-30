"""Bounded overlapping text windows; score original paragraphs, not artificial gold units."""
import math
import re


class WindowReranker:
    """Wrap a reranker without changing paragraph IDs or downstream top-k.

    Word windows are intentionally independent of a private tokenizer. BERT may
    still split words into several pieces; this is NOT a no-truncation guarantee.
    """
    def __init__(self, ranker, *, words=180, overlap=40, max_windows=8):
        if not 1 <= overlap < words or max_windows < 1:
            raise ValueError('invalid window bounds')
        self.ranker = ranker
        self.words, self.overlap, self.max_windows = words, overlap, max_windows

    @property
    def name(self):
        return f'{self.ranker.name}:word-window-{self.words}-{self.overlap}-{self.max_windows}'

    def score(self, query, passages):
        windows, owners = [], []
        for index, passage in enumerate(passages):
            tokens = list(re.finditer(r'\S+', passage))
            if len(tokens) <= self.words:
                parts = [passage]
            else:
                starts = list(range(0, len(tokens), self.words-self.overlap))
                # For very long paragraphs, preserve head and tail instead of
                # pretending an unbounded scan was performed.
                if len(starts) > self.max_windows:
                    starts = starts[:self.max_windows-1] + [max(0, len(tokens)-self.words)]
                parts = [passage[tokens[s].start():tokens[min(s+self.words,len(tokens))-1].end()] for s in starts]
            windows.extend(parts)
            owners.extend([index]*len(parts))
        if not windows:
            return []
        scores = self.ranker.score(query, windows)
        if len(scores) != len(windows) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError('invalid window reranker scores')
        result = [-math.inf]*len(passages)
        for owner, score in zip(owners, scores, strict=True):
            result[owner] = max(result[owner], float(score))
        return result
