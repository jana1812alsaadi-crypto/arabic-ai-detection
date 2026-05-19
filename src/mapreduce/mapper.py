#!/usr/bin/env python3
# Hadoop Streaming mapper for word count over the cleaned Arabic corpus.
# Reads CSV from stdin and emits (word, 1) pairs for each token.

import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

try:
    text_idx = header.index("text_clean")
except (ValueError, AttributeError):
    text_idx = 4

for row in reader:
    if len(row) <= text_idx:
        continue
    text = row[text_idx]
    for word in text.split():
        word = word.strip()
        if word:
            print(f"{word}	1")
