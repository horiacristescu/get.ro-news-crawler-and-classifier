# get.ro - A news crawler and classifier

This project aims to replicate some of the functionality of Google News. We crawl thousands of sources, collect the new articles and extract the text and images. But by doing that, we end up with a huge mass of tens of thousands of news items every day which is impossible to browse by hand. So we must build a news classifier to split the firehose of news into manageable streams. And since it's possible, why classify just in 10-20 topics when we could have thousands? 

## News crawler

The first stage of our system is the crawler. The crawler has to start somewhere so we build a list of newspaper homepage URLs. In order to do that we shamelessly exploit other projects who index the Romanian newssites. Another method is to query Google with keywords that only appear in very recent news (such as names of people mentioned for the first time in the last 24 hours), identifying newspapers.

We crawl the homepages every hour and extract all the links, including from the RSS feed. We compare the links with a list of known older articles and skip those. Since the news homepages change all the time, in a few days we are able to filter out old news from recent ones.

Now that we have the HTML of news articles, we need to extract the title, text and main image. In order to do that we keep a database of all phrases ever met on the same newspaper, and filter out paragraphs that have been seen before. This method works well, but more recently there have been other methods that rank text inside HTML pages based on heuristics, they work well too and don't require the large database of past phrases.

## Classifying news

The main problem we have when we get to classifying news is that we don't actually have a labeled dataset. We don't even have a complete topic list. So we choose word embeddings as our main workhorse by learning unsupervised representations. This works great because we can now have rich representations of topics as well. We just need to select one single keyword or a few for each topic, no need to have thousands of hand-labeled examples.

We built word embeddings using many available tools - starting with word2vec, glove, fasttext and finally doc2vecC. The best embeddings for our purposes were the ones generated by doc2vecC because it models words in document context.

In order to improve the efficiency of this method we treat collocations and names like single tokens and learn vectors for them like we learn all the rest of the vocabulary. Wikipedia was of invaluable help to us here. We simply matched the list of article names to ngrams in a huge corpus of news and only retain those that appear above a certain treshold. This way we have entity names like "Albert Einstein" and multi-word units like "deep learning". We also used entropy based filtering to identify collocations from the corpus.

We have built a web tool to visualize word clouds by similarity and help us define topics in an interactive way. Browsing the embedding space was a unique experience. From each concept many directions spread out, it's as if you're walking a city where each intersection has 20 roads and they connect in hyperspace, permitting shortcuts. Every word is close to every other word, just a few hops away. Usually when the similarity score of two words is above 0.35 (cosinus distance) then they are related.


## Topic definitions

Defining topics is a difficult problem. One way to go about it is to have a dataset with 100..1000 articles classified by hand to each topic. Unfortunately such a dataset is hard to come by. Supervised learning is not easy to apply in this situation. So we turn to unsupervised learning instead. We build word vectors on a large corpus of text (about 1 billion words). Training takes a few hours and results vary depending on window size, algorithm used and corpus size. The best results so far come from doc2vecC (Document to vector through corruption). In doc2vecC the word is modeled in the context of the document (and local window). The corruption part refers to applying dropout when averaging vectors.

If we rely on word vectors we can specify a topic by a single word. This works well for most topics, but for some it is necessary to average the embeddings of a few terms (<50) related to the topic. I have used Wikipedia to extact lists of topics, but some topics might not be on Wiki, so we use another method: we cluster the word vectors and inspect clusters instead of individual words. While there are 1 million words in our vocabulary, we can reduce that to 1000-10000 clusters and go by hand and name them one by one. By doing this we make sure there are no missed topics.

## Document embeddings

In order to compute the embedding vector of a document there are many techniques. We start from word vectors and make a weighted sum of their embeddings. It is essential at this point to maintain a list of stopwords which are not topically informative. In order to build such a list we measure which words appear all over the topic space (compute entropy of the distribution of probability of word to all the topics). Words with low entropy are stopwords, so we select them at a cutoff point.

Another way to wheigh words is to use tf-idf. It works well on average but is surpassed by better methods. We have experimented with many "attention" schemes. One of the best is to compute the similarity of each pair of words in a document, treshold at a certain level and then average. The words that have more related words in the same document come on top. The tf-idf method by contrast is not topic specific, it just boosts rare words.

## Ranking news

We want to emphasize topics that are unusually strong in the last 24 hours, so we build an average distribution of keywords over a large timespan (a year) and compare it to the distribution of keywords in the last 24 hours. Some keywords appear more often than average, other less. Then we cluster the words that appear more often and identify a short list of clusters. We rank each article by similarity to the interest clusters. This would make sure than on an election day we boost election news and on an holliday we boost holliday news.

## Improving topic classification with "strong words"

We observe that sometimes word vectors can be fuzzy and not discriminate well between close topics. We tried to use large embeddins size but it doesn't help. Instead, we select a few "strong words" for each topic, including the topic name, and compute a simple word based topic score. In order to compute strong words related to a topic we use Label Prop where we feed in a bunch of related topics and words and it assigns each word to a topic.

## Rendering the website

In order to render the website we use a python framework (Klein) and ReactJS-like organisation for the HTML. We divide the news into individual indexes by hour we we can roll out old news without having to delete them from a central database. The hourly news databases are built on an incremental basis so we only need to parse the last hour and update it regularly (every minute). The backend also creates an inverted index of the text so we can do keyword search.
