import nltk
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
#
# from urllib.request import Request, urlopen
#
# req = Request('http://www.cmegroup.com/trading/products/#sortField=oi&sortAsc=false&venues=3&page=1&cleared=1&group=1', headers={'User-Agent': 'Mozilla/5.0'})
# webpage = urlopen(req).read()

url = Request('https://towardsdatascience.com/predicting-premier-league-odds-from-ea-player-bfdb52597392', headers={'User-Agent': 'Mozilla/5.0'})
html = urlopen(url).read()
print(html)

soup = BeautifulSoup(html)
print(soup)
# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

print(soup)

text = soup.get_text()
print(text)
# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = 'n'.join(chunk for chunk in chunks if chunk)

print(text)

from nltk.corpus import stopwords
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

#tokenise the data set
from nltk.tokenize import sent_tokenize, word_tokenize
words = word_tokenize(text)
print(words)

# removes punctuation and numbers
wordsFiltered = [word.lower() for word in words if word.isalpha()]
print(wordsFiltered)

# remove stop words from tokenised data set
filtered_words = [word for word in wordsFiltered if word not in stopwords.words('english')]
print(filtered_words)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud(max_words=1000, margin=10, background_color='white',
scale=3, relative_scaling = 0.5, width=500, height=400,
random_state=1).generate(' '.join(filtered_words))
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis("off")
plt.show()
#wc.to_file("/wordcloud.png")