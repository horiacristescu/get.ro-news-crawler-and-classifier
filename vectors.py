import sys
from gensim.models.wrappers import FastText

class Vectors:

    def __init__(self, fname=None):
        self.id2label = []
        self.labels = {}
        self.annoy = None
        self.vecs = np.array([])
        self.base_fname = None
        if fname != None:
            self.load(fname)

    def text(self, id):
        if id in self.labels:
            return self.labels[id].get("text", "")
        else:
            return ""

    def freq(self, id, min_freq=1):
        if id in self.labels:
            freq = self.labels[id].get("freq", min_freq)
            if freq == None:
                freq = min_freq
            return freq
        else:
            return min_freq

    def to_vecs(self, txts):
        if type(txts) == str:
            txts = [ txts ]
        return np.array([ self.linear_word_combination(txt) for txt in txts ])

    def sort_by_vec(self, txts, vec, showScores=False):
        vecs = self.to_vecs(txts)
        rez = zip(txts, np.dot(vecs, vec))
        rez = LOL_sort(rez)
        if showScores:
            return LOL_round(rez)
        else:
            return LOL_keys(rez)

    def add(self, label=None, vec=None, **kvargs):
        '''
            kwargs = freq, text, topic_pos, topic_neg, ...
        '''
        if label in self.labels:
            self.labels.update(kvargs)
            if type(vec) == np.ndarray:
                id = self.labels[label]["id"]
                self.vecs[id] = vec
        else:
            id = len(self.id2label)  # last+1
            self.id2label.append(label)
            kvargs["id"] = id
            self.labels[label] = kvargs
            if vec is not None:
                if self.vecs.shape[0] == 0:
                    self.vecs = np.array([vec])
                else:
                    self.vecs = np.append(self.vecs, np.array([vec]), axis=0)

    def from_vecs_labels(self, vecs=None, labels=None, meta_information=None):
        self.vecs = vecs
        self.id2label = list(labels)
        for id, lbl in enumerate(self.id2label):
            ref = {}
            if meta_information != None:
                ref = dict(meta_information.labels[lbl])
            else:
                ref["freq"] = 1
            ref["id"] = id
            self.labels[lbl] = ref
        self.build()
        return self

    def from_concat(self, v1=None, v2=None, k1=1., k2=1.):
        labels = list(set(v1.id2label) & set(v2.id2label))
        vecs = np.hstack((v1[labels]*k1, v2[labels]*k2))
        vecs = normalize(vecs)
        self.from_vecs_labels(vecs=vecs, labels=labels, meta_information=v1)
        return self

    def save(self, *fname):
        if len(fname)==0:
            fname = self.base_fname
        else:
            fname = fname[0]
            self.base_fname = fname
        if fname == None:
            raise Exception("missing file name")
        # save labels as memoized shelve
        d = shelve.open("%s.labels" % fname, writeback=True)
        labels_flat = {}
        for kw in self.labels:
            labels_flat[kw.encode("utf-8")] = self.labels[kw]
        d.update(labels_flat)
        # save label list as text
        with open("%s.id2label" % fname, "w") as f:
            f.write("\n".join(self.id2label).encode("utf-8"))
        # save vecs as numpy mmap
        np.save("%s.vecs" % fname, self.vecs)
        if self.annoy != None:
            self.annoy.save(self.base_fname + ".annoy")

    def load(self, *fname):
        if len(fname)==0:
            fname = self.base_fname
        else:
            fname = fname[0]
            self.base_fname = fname
        if fname == None:
            raise Exception("missing file name")
        # load labels as memoized shelve
        self.labels = shelve.open("%s.labels" % fname, writeback=True)
        # load label list as text
        with open("%s.id2label" % fname, "r") as f:
            self.id2label = f.read().split("\n")
        # load vecs as numpy mmap
        self.vecs = np.load("%s.vecs.npy" % fname, mmap_mode="r+")
        # loading annoy
        if os.path.isfile(fname + ".annoy"):
            self.annoy = AnnoyIndex(self.vecs.shape[1])
            self.annoy.load(fname + ".annoy")

    def sync(self):
        if hasattr(self.labels, "sync"):
            self.labels.sync()

    def build(self):
        if self.base_fname != None:
            print >>sys.stderr, "Building annoy index", self.base_fname + ".annoy"
        else:
            print >>sys.stderr, "Building annoy index"
        self.annoy = AnnoyIndex(self.vecs.shape[1])
        for i, v in enumerate(self.vecs):
             self.annoy.add_item(i, v)
        self.annoy.build(100)
        if self.base_fname != None:
            self.annoy.save(self.base_fname + ".annoy")

    def __getitem__(self, label):
        if type(label)==list:
            return np.array( [ self[l] for l in label if l in self.labels ] )
        elif type(label) in [str, unicode] and label in self.labels:
            return self.vecs[self.labels[label]["id"]]
        return np.zeros_like(self.vecs[0])

    def __repr__(self):
        count = self.vecs.shape[0]
        if len(self.vecs.shape)>1:
            size = self.vecs.shape[1]
        else:
            size = 0
        if self.base_fname != None:
            return "Vectors(count=%d, width=%d, fname='%s')" % (count, size, self.base_fname)
        else:
            return "Vectors(count=%d, width=%d)" % (count, size)

    def linear_word_combination(self, txt):
        if hasattr(self, "word_vectors"):
            wv = self.word_vectors
        else:
            wv = self
        # merge si cu lista si cu string
        if type(txt)==list:
            words = txt

            ids = []
            factors = []
            for word, score in words:
                if word not in wv.labels:
                    continue
                ids.append(wv.labels[word]["id"])
                factors.append(score)

            if len(ids)==0:
               return np.zeros_like(wv.vecs[0])

            #ids  = [ wv.labels[word]["id"] for (word, score) in words if word in wv.labels ]
            #factors = np.array([ score for (word, score) in words if word in wv.labels ])
            txt_vecs = np.sum( self.vecs[ids].transpose() * factors, axis = 1)
        else:
            ids = [ wv.labels[word]["id"] for word in txt.split() if word in wv.labels ]
            if len(ids)==0:
                return np.zeros_like(wv.vecs[0])
            txt_vecs = np.sum(wv.vecs[ids], axis=0)
        return normalize(txt_vecs)

    def linear_word_combination_posneg(self, pos, neg):
        vpos = self.linear_word_combination(pos)
        vneg = self.linear_word_combination(neg)
        return normalize(vpos - 0.2 * vneg)

    def similar(self, vec, topn=20, showScores=False):
        v_norm = normalize(vec)
        ids, scores = self.annoy.get_nns_by_vector(v_norm, topn, include_distances=True)
        rez = zip(ids, scores)
        if showScores:
            # ALTERNATIVE
            # max(1 - max(score - 0.8, 0), 0) / 2.
            # max(1. - (s - 0.75) * 2., 0)
            return [ ( self.id2label[id], round( min(0.36/(score+0.01), 1.), 3) )
                        for (id, score) in rez ]
        else:
            return [ self.id2label[id]
                        for (id, score) in rez ]

    def similar_vec(self, vec, topn=20, min_score=0.3):
        sims = self.similar(vec, topn=topn, showScores=True)
        ids = LOL_get(sims, 0)
        scores = LOL_get(sims, 1)
        scores = [ max(s - min_score, 0.) for s in scores ]
        vecs = np.array(self[ids])
        scores = np.array(scores)
        rez = vecs.transpose() * scores
        rez = np.sum(rez, axis=1)
        return rez

    def tfidf(self, txt, cutoff=0.2, showScores=False, boostFirstKws=0., useSense=False):
        freq_max = 249197145.
        freq_min = 10.
        words = [ word for word in txt.split() if word in self.labels ];
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        words = []
        rez = []
        poz = 0
        for word in word_count:
            if word not in self.labels:
                continue
            word_fr = self.labels[word].get("freq", freq_min)
            if word_fr == None or word_fr < freq_min:
                word_fr = freq_min
            idf = math.log(freq_max / word_fr)
            tf = 1 + math.log(word_fr)

            poz += 1
            positionFactor = 1. + boostFirstKws/(boostFirstKws+poz)

            if useSense:
                senseFactor = math.log(self.labels[word].get("sense", 0) + 1.) / 9.
            else:
                senseFactor = 1.

            rez.append(tf*idf*positionFactor*senseFactor)
            words.append(word)
        if len(rez)==0:
            return rez
        rez_max = max(rez)
        if rez_max>0:
            rez = [ round( (x / rez_max) ** 2, 3) for x in rez ]
        rez = sorted( zip(words, rez), key=lambda(w,s): -s )
        if not showScores:
            rez = [ w for w,s in rez ]
        if cutoff > 0:
            nr_cut = int(len(rez) * cutoff)
            return rez[:-nr_cut]
        return rez

    def inner_product_rank(self, txt, prag=0.5):
        cloud = [ kw for kw in txt.split() if kw in self.labels ]
        cloud_count = defaultdict(int)
        for kw in cloud:
            cloud_count[kw] += 1
        cloud = cloud_count.keys()
        vecs = self[cloud]
        sims_mat = np.dot(vecs, vecs.transpose())
        sims_mat = sims_mat - prag
        sims_mat[sims_mat<0] = 0
        rez = zip(cloud, np.sum(sims_mat, axis=0) - (1-prag))
        rez = [ (k,round(cloud_count[k]*s,3)) for k,s in rez if s>0.01]
        return LOL_sort(rez)

    def from_gensim(self, fname, normalizeVectors=True):
        import gensim
        print >>sys.stderr, "Loading embeddings file", fname
        if re.search(r'bin$', fname):
            model = FastText.load_fasttext_format(fname)
        elif re.search(r'bin$', fname):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                fname,
                binary=True,
                encoding='utf-8',
                unicode_errors='ignore',
                fvocab=fvocab
                #,unicode_errors='replace'
            )
        elif re.search(r'vec$', fname):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                fname,
                binary=False,
                fvocab=fvocab
            )
        else:
            model = gensim.models.KeyedVectors.load(fname)

        if "wv" in model:
            model = model.wv

        try:
            self.vecs = model.vectors
        except:
            self.vecs = model.syn0

        self.id2label = [ ] + model.index2word
        for w in model.vocab:
            self.labels[w] = {
                "id":   model.vocab[w].index,
                "freq": model.vocab[w].count}

        if normalizeVectors:
            self.vecs = normalize(self.vecs)
        self.build()
        return self

    def from_topics(self, word_vectors=None, posNeg=False):
        self.word_vectors = word_vectors
        self.topic_cloud, self.topic_avoid = load_topics()
        for topic in self.topic_cloud:
            pos = self.topic_cloud[topic]
            neg = self.topic_avoid[topic]
            if pos == "":
                continue
            if posNeg:
                vec = self.word_vectors.linear_word_combination_posneg(pos, neg)
                self.add(label=topic, vec=vec, pos=pos, neg=neg)
            else:
                vec = self.word_vectors.linear_word_combination(pos)
                self.add(label=topic, vec=vec, pos=pos, neg=neg)
        self.build()
        return self

    def from_documents(self, word_vectors=None, fname=None, nr_lines=10, extractor=None):
        self.word_vectors = word_vectors
        wv = self.word_vectors
        nrl = 0
        new_vecs = []
        with open(fname, "r") as f:
            for line in f:
                vline = line.strip().split("|")
                txt = vline.pop(-1)
                if extractor!=None:
                    vtxt = wv.linear_word_combination(extractor(txt))
                else:
                    vtxt = wv.linear_word_combination(wv.tfidf(txt, cutoff=0.5, showScores=True))
                new_vecs.append(vtxt)
                self.add(label=str(nrl), vec=None, text=txt)
                nrl += 1
                if nrl % 100 == 0:
                    print >>sys.stderr, nrl
                if nrl > nr_lines:
                    break
        self.vecs = np.array(new_vecs)
        self.build()
        return self

    def from_selection(self, select=None, word_vectors=None):
        self.word_vectors = word_vectors
        wv = self.word_vectors
        ids = [ wv.labels[label]["id"] for label in select if label in wv.labels ]
        self.vecs = wv.vecs[ids]
        for label in select:
            if label not in wv.labels:
                continue
            self.add(label=label, vec=None, **wv.labels[label])
        self.build()
        return self

    def similar_second_order(self, word_vec_or_list, topn=20, showScores=False):
        rez = defaultdict(float)
        if type(word_vec_or_list)==str:
            frontier = self.similar(self.linear_word_combination(word_vec_or_list), topn=topn)
        elif type(word_vec_or_list)==list:
            frontier = word_vec_or_list
        else:
            frontier = self.similar(word_vec_or_list, topn=topn, showScores=True)
        for word1, score1 in frontier:
            for word2, score2 in self.similar(self[word1], topn=topn, showScores=True):
                rez[word2] += score1 * score2
        sc_max = max(rez.values())
        rez = [ (w, round(s/sc_max, 3)) for (w,s) in rez.items() ]
        rez = sorted(rez, key=lambda(w,s): -s)
        if not showScores:
            rez = [ w for w,s in rez ]
        return rez[:topn]

    def similar_nth_order(self, word_vec_or_list, topn=20, showScores=False, order=2):
        rez = word_vec_or_list
        for i in range(order):
            rez = self.similar_second_order(rez, topn=topn, showScores=True)
        if not showScores:
            rez = [ w for w,s in rez ]
        return rez

    def from_classified_vectors(self, topics, words, build=True):
        '''
        clasifica un set de vectori si creaza vectori noi din topicuri
        '''
        prag = 0.28
        vecs = []
        n_topics = len(topics.id2label)
        for i, word in enumerate(words.id2label):

            vec = np.dot(topics.vecs, words[word])
            vec = vec - 0.5
            vec[vec<0] /= 100
            vecs.append(vec)

            # vec = np.zeros(n_topics)
            # for topic, sim in topics.similar(words[word], topn=len(topics.id2label), showScores=True):
            #     topic_id = topics.labels[topic]["id"]
            #     if sim - prag > 0:
            #         vec[topic_id] = sim - prag
            #     else:
            #         vec[topic_id] = (sim - prag)/100.
            # vecs.append(vec)

            self.add(label=word)
            if i%100==0:
                print >>sys.stderr, i
        self.vecs = np.array(vecs)
        if build:
            self.build()
        return self

    def from_average_neighbor_embeddings(self, words, topics, word_list):
        nr = 0
        new_vecs = []
        for word in word_list:
            cloud = words.similar(words[word], topn=20)
            counts = count_strong_words(" ".join(cloud))
            if len(counts)>0:
                direction = topics[counts[0][0]]
            else:
                direction = words[word]
            sims = np.dot(words[cloud], direction) - 0.25
            sims[sims<0] = 0.
            sims_sum = np.sum(sims)
            if sims_sum>0:
                sims = sims / np.sum(sims)
            vec = words.linear_word_combination(zip(cloud, sims))
            self.add(label=word, vec=None, **words.labels[word])
            new_vecs.append(vec)
            nr += 1
            #print word, counts[0][0], LOL_sort(zip(cloud, sims))
            if nr%1000==0:
                print >>sys.stderr, nr
        self.vecs = np.array(new_vecs)
        self.build()

    def similar_as_texts(self, vec, topn=20, plainText=False, tfidf=True, showScores=False):
        rez = []
        for id, score in self.similar(vec, topn=topn, showScores=True):
            txt = self.labels[id]["text"]
            if tfidf:
                txt = " ".join(words_d.tfidf(txt, cutoff=0.3))
            if showScores:
                rez.append((txt, round(score, 3) ))
            else:
                rez.append(txt)
        if plainText and not showScores:
            rez = "\n\n".join(rez)
        return rez

    def pair_similarity_rank(self, txt):
        cloud = defaultdict(int)
        for kw in txt.split():
            if kw in self.labels:
                cloud[kw] += 1
        cloud_kw = cloud.keys()
        if len(cloud_kw) == 0:
            return []
        vecs = self[cloud_kw]
        scores = np.sum(np.dot(vecs, vecs.transpose()), axis=0)
        rez = []
        for kw, score in zip(cloud_kw, scores):
            rez.append((kw, score*cloud[kw]))
        return LOL_norm(LOL_sort(rez))

    def bigram_rank(self, txt, cutoff=0.5, showScores=False, temp=1, includeWordSim=False):
        '''
        Ca tfidf, dar bazat pe bigram counts
        '''
        txt = self.tfidf(txt, cutoff=cutoff)
        tscor = [ 0. ] * len(txt)
        for i in range(len(txt)):
            for j in range(i+1, len(txt)):
                if (txt[i] not in self.labels) or (txt[j] not in self.labels):
                    continue
                if includeWordSim:
                    sim = np.dot(words[txt[i]], words[txt[j]])
                    if sim < 0.3:
                        sim = 0
                    sim = 1. + sim * 20.
                else:
                    sim = 1.
                sc = sim * self.bigram_count(txt[i], txt[j], temp=temp)
                tscor[i] += sc
                tscor[j] += sc
        # tscor = [ s**0.5 for s in tscor ]
        smax = max(tscor)
        if smax==0:
            smax = 1
        tscor = [ round(s/smax, 3) for s in tscor ]
        rez = sorted(zip(txt, tscor), key=lambda (w,s): -s)
        if not showScores:
            rez = [ w for w, s in rez ]
        return rez

    def bigram_count(self, w1, w2, temp=1.):
        if w2 < w1:
            w1, w2 = w2, w1
        if w1 in stop or w2 in stop:
            return 0
        try:
            w1 = lfhf[w1]
            w2 = lfhf[w2]
            b = w1 + " " + w2
            f1 = self.labels[w1]["freq"]
            f2 = self.labels[w2]["freq"]
            f1 = max(f1 - 1000, 1)
            f2 = max(f2 - 1000, 1)
            rez = float(max(bigr.get(b, 0)-100, 1)) ** temp / ((f1 + f2) + 1)
        except:
            rez = 0
        rez /= 0.00485
        rez = rez ** 0.4
        if rez < 0.03:
            rez = 0.
        return rez

    def sim_rank(self, txt, cutoff=0.4, showScores=False):
        '''
        Ca un fel de tfidf, bazat pe sim kw - kw
        '''
        vtxt = []
        tfidf = []
        for (kw, sc) in self.tfidf(txt, cutoff=cutoff, showScores=True):
            vtxt.append(kw)
            tfidf.append(sc)
        txt = vtxt
        vecs = self[txt]
        sim_mat = np.dot(vecs, vecs.transpose())
        sim_mat[sim_mat<0.2] = 0
        sims = np.sum(sim_mat, axis=0) - 1.
        rez = [ (w, s1*(s2**2)) for (w,s1,s2) in zip(txt, sims, tfidf) ]
        smax = max([s for w,s in rez])
        rez = [ (w, round(s/smax,3)) for (w,s) in rez ]
        rez = sorted(rez, key=lambda(w,s): -s)
        if not showScores:
            rez = [ w for w, s in rez ]
        return rez
