#Expert Finder

### Installation and usage:
https://colab.research.google.com/drive/1TLCr46f-Mn2oOpegoMT6lTim2d4uBVLO
## ExpertRecommendationTool Class (API):
> constructor
* data_file_id - ID of the data file in google drive.
* author_names - Names of the candidate experts
* flat_index_threshold (optional) -below this threshold,
     title search will be done using lossless representation 
     of the title embeddings (defaults to 100000)
     
> recommend_topn

Returns top-n experts for the given paper.

* title - queried paper title 
* author_names - queried paper authors
* topn (default: 10)- number of queries to retrieve
* k (default: 100) - knn value for title search. 
* recency_decay [0,1] (default: 0.25)- lambda parameter. How much importance to give to recent
papers vs. old.
* author_weight [0,1] (default 0.7)- weight given to direct relationships during 
title similarity search
* cf_weight [0,1] (default 0.5)- weight given to collaborative filtering vs. title search 
in final ranking of candidates
* clip_n (default: 4)- how many paper results to show in histogram per each recommended expert.
  The rest will be aggregated under 'Other papers'.
* max_text_len (optional) - maximum characters of histogram labels before line break.
* font_size (default: 14)- histogram labels font size.
* fig_height (optional)- height of each histogram
* fig_width (default: 9) - width of each histogram