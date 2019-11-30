#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys
import math

unique_count = 0
nan_count = 0
word = None
current_word = None

for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)
    try:
        word = float(word)
        count = int(count)
    except ValueError:
        continue

    if word != current_word:
        current_word = word
        print(current_word)
        unique_count += count
print('Total no. of unique values is %s' % (unique_count))

