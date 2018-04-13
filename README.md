# get.ro - A news crawler and classifier

## News spider
    - building a list of news sources
    - crawling homepages for new articles
    - article extractor
        - method based on filtering
        - method based on scoring

## Classifying news
    - main problem: don't have a dataset, using unsupervised methods and one-shot learning
    - word embeddings
        - my word embeddings tool
        - word2vec, fasttext, doc2vecC
        - collocations and entity names
            - counting ngrams with count-min-sketch
        - exploring word vectors
            - online tool
    - topic definitions
        - how I built the topic list -> clustering
        - tool to edit topics
            - problems with topic overlap
        - finding "strong words"
    - document embeddings
        - attention schemes - tf-idf, word similarity based
        - combining vocabulary check with document embeddings
    - ranking
        - finding trends by comparing word distributions

## Rendering the website
    - inverted index & document vectors
        - divided by hour, rebuiding only the last period
    - generating the website
