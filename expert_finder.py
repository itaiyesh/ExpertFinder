from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TitleSearchFaissHandler:
    def __init__(self,
                 model_save_path,
                 paper2data,
                 author2idx,
                 idx2author,
                 idx2paper,
                 author_idx2paper_idxs_file,
                 index_file,
                 paper_embs_file=None,
                 author_list=None,
                 flat_index_threshold=100000):
        self.sentence_transformer = SentenceTransformer(model_save_path).to(device)
        self.idx2author = idx2author
        self.paper_embs_file = paper_embs_file
        self.idx2paper = idx2paper
        self.paper2idx = {paper: idx for idx, paper in enumerate(self.idx2paper)}
        self.paper2data = paper2data  # DEBUG

        self.author2idx = author2idx
        self.index_file = index_file
        self.author_idx2paper_idxs_file = author_idx2paper_idxs_file
        self.flat_index_threshold = flat_index_threshold
        self.set_authors(author_list)

    def set_authors(self, author_list):
        self.author_list = author_list if author_list is not None else list(self.author2idx.keys())
        relevant_paper_idxs = None

        if author_list is not None:
            author_idx2paper_idxs = load(self.author_idx2paper_idxs_file)
            relevant_paper_idxs = set()
            for author in author_list:
                author_idx = self.author2idx[author]
                if not author_idx in author_idx2paper_idxs:
                    print(author_idx, " not in")
                    continue  # This means the authors has no papers in title embs
                for paper_idx in author_idx2paper_idxs[author_idx]:
                    relevant_paper_idxs.add(paper_idx)
            relevant_paper_idxs = sorted(list(relevant_paper_idxs))

        if relevant_paper_idxs is not None:
            self.paper_idx2data = {paper_idx: self.paper2data[self.idx2paper[paper_idx]] for paper_idx in
                                   relevant_paper_idxs}
            print("Restricting title search to {} papers (for {} authors)".format(len(relevant_paper_idxs),
                                                                                  len(author_list)))
        else:
            self.paper_idx2data = {paper_idx: self.paper2data[paper_id] for paper_idx, paper_id in
                                   enumerate(self.idx2paper)}

        relevant_papers = set([self.idx2paper[paper_idx] for paper_idx in self.paper_idx2data.keys()])

        if len(relevant_papers) < self.flat_index_threshold:
            print("Using flat indexing since papers < {}".format(self.flat_index_threshold))
            self.use_flat_index = True
        else:
            self.use_flat_index = False
        if self.use_flat_index:
            paper_embs, paper_ids = self.get_embs(relevant_papers)

            self.paper_vectors = keyedvectors.WordEmbeddingsKeyedVectors(len(paper_embs[0]))
            self.paper_vectors.add(paper_ids, paper_embs)
        else:
            self.index = faiss.read_index(self.index_file)
            # That should be enough
            self.index.nprobe = 100
            remove_papers = [paper for paper in self.idx2paper if paper not in relevant_papers]
            if len(remove_papers) > 0: self.index.remove_ids(np.array(remove_papers))
            print(self.index.ntotal, " papers are in index.")

    def get_embs(self, relevant_papers):
        # If paper embs are given, don't calculate them again!
        if self.paper_embs_file is not None:
            print("Using existing embeddings file.")
            relevant_paper_idxs = [self.paper2idx[paper] for paper in relevant_papers]
            relevant_paper_idxs = sorted(relevant_paper_idxs)
            relevant_paper_ids_sorted = [self.idx2paper[idx] for idx in relevant_paper_idxs]
            with h5py.File(self.paper_embs_file, 'r') as f:
                paper_embs = f['embs'][relevant_paper_idxs]

            return paper_embs, relevant_paper_ids_sorted

        # Otherwise, run language model to get new embs
        chunk_size = 1048
        chunk = []
        paper_embs = []
        paper_ids = []
        pbar = tqdm(total=len(relevant_papers))
        #     for paper, data in self.paper2data.items():
        for paper_id in list(relevant_papers):
            paper_data = self.paper2data[paper_id]
            chunk.append(paper_data['title'])
            paper_ids.append(paper_id)

            if len(chunk) > chunk_size:
                embs = self.sentence_transformer.encode(chunk, show_progress_bar=False)
                chunk.clear()
                paper_embs += embs
                pbar.update(len(chunk))
        pbar.close()
        if len(chunk) > 0:
            embs = self.sentence_transformer.encode(chunk, show_progress_bar=False)
            paper_embs += embs
        return paper_embs, paper_ids

    def recommend_topn(self, title, radius, k, recency_decay, author_weight):
        assert recency_decay < 1 and recency_decay >= 0, 'Lambda must be [0, 1)'
        em = np.array(self.sentence_transformer.encode([title]))
        faiss.normalize_L2(em)

        if self.use_flat_index:
            similar_papers = self.paper_vectors.similar_by_vector(self.sentence_transformer.encode([title])[0], topn=k)
            ids, similarities = zip(*similar_papers)
            distances = [1 - sim for sim in similarities]

        else:
            distances, ids = self.index.search(em, k)  # *100)
            ids = ids[0]
            distances = distances[0]

        author2score = Counter()
        author2best_papers = {}

        for paper_id, distance in zip(ids, distances):
            if paper_id == -1: break
            if distance > 1: break
            similarity = 1 - distance
            #       similarity = similarity**l
            paper_data = self.paper2data[paper_id]
            delta_t = 2019 - int(paper_data['year'])
            recency_score = np.exp(-(recency_decay / (1 - recency_decay)) * delta_t)
            # print(recency_score, similarity,paper_data['title'], "Score: {}".format(similarity*recency_score))
            for author_idx, weight in zip(paper_data['citing_authors_idxs'], paper_data['citing_authors_weights']):
                author = self.idx2author[author_idx]

                weight = author_weight if author in paper_data['authors'] else 1 - author_weight
                score = weight * similarity * recency_score
                author2score[author] += score
                if author not in author2best_papers:
                    author2best_papers[author] = []
                author2best_papers[author].append((score, paper_id))

        possible_authors_set = set(self.author_list)
        recommended_author_ids = []
        recommended_author_scores = []
        collected_authors = 0
        for author_id, value in author2score.most_common():
            if (radius is not None and collected_authors > radius) or len(possible_authors_set) == 0:
                break
            if author_id in possible_authors_set:
                recommended_author_ids.append(author_id)
                recommended_author_scores.append(value)
                collected_authors += 1
                possible_authors_set.remove(author_id)

        if len(recommended_author_scores) > 0:
            return recommended_author_ids, norm(np.array(recommended_author_scores).reshape(1, -1))[
                0], author2best_papers  # {author:data for author,data in author2best_paper.items() if author in recommended_author_ids}#[author2best_paper[author] for author in recommended_author_ids]
        else:
            return [], [], {}  # []


class CollaborativeFilteringHandler:
    def __init__(self, auto_encoder_path, auto_encoder_metadata_path, author2idx, idx2author, idx2paper,
                 author_list=None):

        self.author2idx = author2idx
        self.idx2author = idx2author
        metadata = torch.load(auto_encoder_metadata_path)
        self.auto_encoder = MultiVAE(metadata['p_dims']).to(device)
        self.auto_encoder.load_state_dict(torch.load(auto_encoder_path, map_location=device))
        self.auto_encoder.eval()

        self.set_authors(author_list)

    def set_authors(self, author_list):
        self.author_list = author_list if author_list is not None else list(author2idx.keys())

    def recommend_topn(self, authors, radius=None):
        author_idxs = [self.author2idx[int(author)] for author in authors]
        data = sparse.csr_matrix((np.ones_like(author_idxs), (np.zeros_like(author_idxs), author_idxs)),
                                 dtype='float32', shape=(1, len(self.author2idx)))
        data = sparse2torch_sparse(data).to(device)
        recon_batch, mu, logvar = self.auto_encoder(data)
        flattened_batch = recon_batch.cpu().detach().numpy().flatten()
        possible_authors_set = set(self.author_list)
        collected_authors = 0
        recommended_author_ids = []
        recommended_author_probs = []
        recommended_author_idxs = []  # DEBUG
        for author_idx in flattened_batch.argsort()[::-1]:
            if (radius is not None and collected_authors > radius) or len(possible_authors_set) == 0:
                break
            author_id = self.idx2author[author_idx]
            if author_id in possible_authors_set:
                recommended_author_ids.append(author_id)
                recommended_author_probs.append(flattened_batch[author_idx])
                collected_authors += 1
                possible_authors_set.remove(author_id)
                recommended_author_idxs.append(author_idx)

        recommended_author_probs = norm(np.array(recommended_author_probs).reshape(1, -1))[0]
        return recommended_author_ids, recommended_author_probs


class ExpertRecommendationTool:
    def __init__(self, data_file_id, author_names, flat_index_threshold = 100000):
        if not os.path.exists('data.tar'):
            download_file_from_google_drive(data_file_id, 'data.tar')

        subprocess.Popen(['tar', '-xvf', 'data.tar']).communicate()
        bert_model_path = 'gdrive/My Drive/ExpertFinal/model_2019-10-23_23_38'
        cf_model_path = 'gdrive/My Drive/ExpertFinal/model_path'
        cf_model_metadata_path = 'gdrive/My Drive/ExpertFinal/model_metadata_path'
        filtered_paper2data_file = 'gdrive/My Drive/ExpertFinal/filtered_paper2data.pkl'
        author2name_file = 'gdrive/My Drive/ExpertFinal/author2name.pkl'
        author2idx_file = 'gdrive/My Drive/ExpertFinal/author2idx.pkl'
        idx2paper_file = 'gdrive/My Drive/ExpertFinal/idx2paper.pkl'
        author_idx2paper_idxs_file = 'gdrive/My Drive/ExpertFinal/author_idx2paper_idxs.pkl'
        paper_embs_file = 'gdrive/My Drive/ExpertFinal/title_embs.h5'
        index_file = 'gdrive/My Drive/index_pcr128'

        self.ef = ExpertFinder(bert_model_path,
                               cf_model_path,
                               cf_model_metadata_path,
                               filtered_paper2data_file,
                               author2name_file,
                               author2idx_file,
                               idx2paper_file,
                               author_idx2paper_idxs_file,
                               index_file,
                               paper_embs_file,
                               authors_names=author_names,
                               flat_index_threshold=flat_index_threshold)

    def recommend_topn(self, title, author_names, topn=10, k=100, recency_decay=0.25, author_weight=0.7, cf_weight=0.5,
                       clip_n=4,
                       max_text_len=None, font_size=14, fig_height=None, fig_width=4):
        print("Title: {} Authors: {}".format(title, author_names))
        return self.ef.recommend_topn(title, author_names, topn, k, recency_decay, author_weight, cf_weight, clip_n,
                                      max_text_len, font_size, fig_height, fig_width)

    def set_authors(self, authors_names):
        return self.ef.set_authors(authors_names)


class ExpertFinder:
    def __init__(self,
                 bert_model_path,
                 vae_model_path,
                 vae_metadata_path,
                 paper2data_file,
                 author2name_file,
                 author2idx_file,
                 idx2paper_file,
                 author_idx2paper_idxs_file,
                 index_file=None,
                 paper_embs_file=None,
                 authors_names=None,
                 flat_index_threshold=100000):

        self.paper2data = load(paper2data_file)
        self.author2idx = load(author2idx_file)
        self.idx2author = {idx: author for author, idx in self.author2idx.items()}
        self.author2name = load(author2name_file)
        self.name2author = {name: author for author, name in self.author2name.items()}
        self.idx2paper = load(idx2paper_file)

        self.authors_list = self.get_authors_from_names(authors_names) if authors_names else None
        self.title_search_handler = TitleSearchFaissHandler(bert_model_path,
                                                            self.paper2data,
                                                            self.author2idx,
                                                            self.idx2author,
                                                            self.idx2paper,
                                                            author_idx2paper_idxs_file,
                                                            index_file,
                                                            paper_embs_file,
                                                            author_list=self.authors_list,
                                                            flat_index_threshold=flat_index_threshold)

        self.collaborative_filter_handler = CollaborativeFilteringHandler(
            vae_model_path,
            vae_metadata_path,
            self.author2idx,
            self.idx2author,
            self.idx2paper,
            self.authors_list)

    def set_authors(self, authors_names):
        self.authors_list = self.get_authors_from_names(authors_names) if authors_names else None
        assert self.authors_list is None or len(self.authors_list) > 1, 'Must use at least 2 authors'
        self.title_search_handler.set_authors(self.authors_list)
        self.collaborative_filter_handler.set_authors(self.authors_list)

    def get_close_matches_icase(self, word, possibilities, *args, **kwargs):
        """ Case-insensitive version of difflib.get_close_matches """
        lword = word.lower()
        lpos = {p.lower(): p for p in possibilities}
        lmatches = difflib.get_close_matches(lword, lpos.keys(), *args, **kwargs)
        return [lpos[m] for m in lmatches]

    def get_authors_from_names(self, author_names):
        names = []
        missing_names = []
        auto = False
        for name in author_names:
            if name not in self.name2author.keys():
                # close_matches = difflib.get_close_matches(name, self.name2author.keys())
                close_matches = self.get_close_matches_icase(name, self.name2author.keys())

                if len(close_matches) == 0:
                    print("{} does not exist in database.".format(name))
                    missing_names.append(name)
                    continue

                print("{} does not exist in database. Did you mean:".format(name))
                for i, close_match in enumerate(close_matches):
                    print("[{}] {}".format(i, close_match))
                if not auto: selection = input("[enter] to skip. [a] to auto-select all.")
                if selection == 'a':
                    auto = True
                    selected_match = close_matches[0]
                    names.append(selected_match)
                    print("{} (original) -> {} (matched)".format(name, selected_match))
                    continue
                if selection.isdigit() and int(selection) < len(close_matches):
                    selected_match = close_matches[int(selection)]
                    names.append(selected_match)
                    print("Selected: {}".format(selected_match))
            else:
                names.append(name)

        print("Matched authors ({}):".format(len(names)))
        for name in names:
            print(name)
        if len(missing_names) > 0:
            print("Missing ({}):".format(len(missing_names)))
            for name in missing_names:
                print(name)

        authors = [int(self.name2author[name]) for name in names]

        return authors

    def recommend_topn(self, title, author_names, topn=10, k=100, recency_decay=0.25, author_weight=0.7, cf_weight=0.5,
                       clip_n=4,
                       max_text_len=None, font_size=14, fig_height=None, fig_width=9):
        radius = max(topn * 10, 1000)  # This dictates the strength of CF
        #     radius = None
        authors = self.get_authors_from_names(author_names)

        recommended_by_title = []
        author2best_papers = {}
        if (title is not None) and title != '':
            recommended_by_title, _ , author2best_papers = self.title_search_handler.recommend_topn(
                title, radius, k, recency_decay, author_weight)
        recommended_by_authors = []
        recommended_by_authors_probs = []
        if len(authors) > 0:
            recommended_by_authors, recommended_by_authors_probs = self.collaborative_filter_handler.recommend_topn(
                authors, radius=radius)

        predicted_author2idx = {author: idx for idx, author in
                                enumerate(set(recommended_by_title).union(set(recommended_by_authors)))}
        predicted_idx2author = {idx: author for author, idx in predicted_author2idx.items()}
        author2info = {}
        for author, best_papers in author2best_papers.items():
            best_papers = sorted(best_papers, key=lambda x: x[0], reverse=True)
            if len(best_papers) > clip_n:
                others_sum = sum([x[0] for x in best_papers[clip_n:]])
                best_papers = best_papers[:clip_n] + [(others_sum, -1)]

            author2info[author] = [(score * (1 - cf_weight), id) for score, id in best_papers]

        for author, prob in zip(recommended_by_authors, recommended_by_authors_probs):
            if author not in author2info: author2info[author] = []
            author2info[author].append((prob * cf_weight, -2))

        author2score = Counter()
        for author, info in author2info.items():
            author2score[author] = sum([score for score, _ in info])

        # print(author2score)
        sorted_authors = []
        possible_authors = set(self.authors_list)
        for author, score in author2score.most_common():
            if author in possible_authors:
                sorted_authors.append(author)
            if len(sorted_authors) >= topn: break
        # print(sorted_authors)

        if len(sorted_authors) == 0: return [], []
        with PdfPages('results.pdf') as pdf:
            max_percent = 0
            for author in sorted_authors:
                info = author2info[author]
                for score, _ in info:
                    if score > max_percent:
                        max_percent = score

            # print("Max percent: {}".format(max_percent))
            fig_height_ = fig_height if fig_height is not None else clip_n

            fig, axs = plt.subplots(nrows=len(sorted_authors), figsize=(fig_width, fig_height_ * len(sorted_authors)),
                                    squeeze=False)
            if len(sorted_authors) > 1:
                plt.subplots_adjust(left=None, bottom=None, right=None, top=0.7, wspace=1.0, hspace=1.0)

            for i, author in enumerate(sorted_authors):
                author_name = self.author2name[str(author)]
                info = author2info[author]
                total_score = author2score[author]
                # print('{}: (Author total score:{})'.format(author_name,total_score))
                graph_labels = []
                graph_scores = []
                for score, element in info:
                    if element == -2:
                        # print('**CF (Contribution: {})'.format(score))
                        graph_labels.append('Collaborative filtering')
                        graph_scores.append(score)
                    elif element == -1:  # Others
                        graph_labels.append('Other papers')
                        # print("Other score: ",score)
                        graph_scores.append(score)
                    else:
                        paper_id = element
                        paper_data = self.paper2data[paper_id]
                        paper_title = paper_data['title']
                        if max_text_len is not None:
                            if len(paper_title) > max_text_len:
                                new_title = paper_title[:max_text_len] + '-\n'
                                if len(paper_title[max_text_len:]) > max_text_len:
                                    new_title += paper_title[max_text_len:max_text_len * 2] + '-\n'
                                    new_title += paper_title[max_text_len * 2:]
                                else:
                                    new_title += paper_title[max_text_len:]
                                paper_title = new_title
                        paper_year = paper_data['year']
                        is_paper_author = int(author) in paper_data['authors']

                        # print('**{} ({}) (Contribution: {})'.format(paper_title,paper_year,score))
                        graph_labels.append(
                            '{} ({}){}'.format(paper_title, paper_year, ' - Author' if is_paper_author else ''))
                        graph_scores.append(score)

                axs[i][0].set_title('{} (Score: {:.2f})'.format(author_name, total_score), y=1.05,
                                    fontsize=font_size)  # TODO: Remove for paper
                df = pd.DataFrame({'Score': graph_scores, 'Papers': graph_labels})
                df = df.sort_values(['Score'], ascending=False).reset_index(drop=True)
                pal = sns.color_palette("Blues_r", len(df))
                rank = df['Score'].argsort().argsort()
                g = sns.barplot(x='Score', y='Papers', data=df, ax=axs[i][0], orient='h',
                                palette=np.array(pal[::-1])[rank])  # palette = sns.color_palette("Blues_r"))
                g.set_xlim(0.0, max_percent)
                g.tick_params(labelsize=font_size)
                axs[i][0].set_ylabel('', fontsize=font_size)
            if len(sorted_authors) <= 1:
                pdf.savefig(bbox_inches='tight')
            else:
                pdf.savefig(pad_inches=0.5, bbox_inches='tight')

        return author_names, author2info
