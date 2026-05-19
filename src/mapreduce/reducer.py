#!/usr/bin/env python3
# Hadoop Streaming reducer for word count.
# Receives sorted (word, 1) pairs from mappers and sums counts per word.

import sys

current_word = None
current_count = 0

for line in sys.stdin:
    try:
        word, count = line.strip().split("	", 1)
        count = int(count)
    except ValueError:
        continue

    if word == current_word:
        current_count += count
    else:
        if current_word is not None:
            print(f"{current_word}	{current_count}")
        current_word = word
        current_count = count

if current_word is not None:
    print(f"{current_word}	{current_count}")
