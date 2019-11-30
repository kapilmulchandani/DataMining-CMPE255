#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

limit = 5000
lower_limit = 10000
upper_limit = 20000
count1 = 0
count2 = 0
word = None
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)
    try:
        word = float(word)
        count = int(count)
    except ValueError:
        continue
    if lower_limit < word < upper_limit:
        count2 += count
    if limit < word:
        count1 += count
print('Total no. of rows in between %s and %s is %s' % (lower_limit, upper_limit, count2))
print('Total no. of rows greater than %s is %s' % (limit, count1))
