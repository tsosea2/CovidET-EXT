# CovidET-EXT (ACL 2023)

This repo contains the dataset and code for our ACL 2023 paper. If you use this dataset, please cite our paper.

Title: <a href="https://aclanthology.org/2023.acl-long.531/">Unsupervised Extractive Summarization of Emotion Triggers</a>

Authors: <a href="https://www.tsosea.com/">Tiberiu Sosea</a>, <a href="https://honglizhan.github.io/">Hongli Zhan</a>, <a href="https://jessyli.com/">Junyi Jessy Li</a>, <a href="https://www.cs.uic.edu/~cornelia/">Cornelia Caragea</a>

```bibtex
@inproceedings{sosea-etal-2023-unsupervised,
    title = "Unsupervised Extractive Summarization of Emotion Triggers",
    author = "Sosea, Tiberiu  and
      Zhan, Hongli  and
      Li, Junyi Jessy  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.531",
    pages = "9550--9569",
    abstract = "Understanding what leads to emotions during large-scale crises is important as it can provide groundings for expressed emotions and subsequently improve the understanding of ongoing disasters. Recent approaches trained supervised models to both detect emotions and explain emotion triggers (events and appraisals) via abstractive summarization. However, obtaining timely and qualitative abstractive summaries is expensive and extremely time-consuming, requiring highly-trained expert annotators. In time-sensitive, high-stake contexts, this can block necessary responses. We instead pursue unsupervised systems that extract triggers from text. First, we introduce CovidET-EXT, augmenting (Zhan et al., 2022){'}s abstractive dataset (in the context of the COVID-19 crisis) with extractive triggers. Second, we develop new unsupervised learning models that can jointly detect emotions and summarize their triggers. Our best approach, entitled Emotion-Aware Pagerank, incorporates emotion information from external sources combined with a language understanding module, and outperforms strong baselines. We release our data and code at https://github.com/tsosea2/CovidET-EXT.",
}
```

This work is inspired by the EMNLP 2022 paper <a href="https://aclanthology.org/2022.emnlp-main.642.pdf">Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts</a>. Please cite it as well:

```bibtex
@inproceedings{zhan-etal-2022-feel,
    title = "Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts",
    author = "Zhan, Hongli  and
      Sosea, Tiberiu  and
      Caragea, Cornelia  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.642",
    pages = "9436--9453",
    abstract = "Crises such as the COVID-19 pandemic continuously threaten our world and emotionally affect billions of people worldwide in distinct ways. Understanding the triggers leading to people{'}s emotions is of crucial importance. Social media posts can be a good source of such analysis, yet these texts tend to be charged with multiple emotions, with triggers scattering across multiple sentences. This paper takes a novel angle, namely, emotion detection and trigger summarization, aiming to both detect perceived emotions in text, and summarize events and their appraisals that trigger each emotion. To support this goal, we introduce CovidET (Emotions and their Triggers during Covid-19), a dataset of {\textasciitilde}1,900 English Reddit posts related to COVID-19, which contains manual annotations of perceived emotions and abstractive summaries of their triggers described in the post. We develop strong baselines to jointly detect emotions and summarize emotion triggers. Our analyses show that CovidET presents new challenges in emotion-specific summarization, as well as multi-emotion detection in long social media posts.",
}
```

# Abstract
Understanding what leads to emotions during large-scale crises is important as it can provide groundings for expressed emotions and subsequently improve the understanding of ongoing disasters. Recent approaches trained supervised models to both detect emotions and explain emotion triggers (events and appraisals) via abstractive summarization. However, obtaining timely and qualitative abstractive summaries is expensive and extremely time-consuming, requiring highly-trained expert annotators. In time-sensitive, high-stake contexts, this can block necessary responses. We instead pursue unsupervised systems that extract triggers from text. First, we introduce CovidET-EXT, augmenting (Zhan et al., 2022)’s abstractive dataset (in the context of the COVID-19 crisis) with extractive triggers. Second, we develop new unsupervised learning models that can jointly detect emotions and summarize their triggers. Our best approach, entitled Emotion-Aware Pagerank, incorporates emotion information from external sources combined with a language understanding module, and outperforms strong baselines. We release our data and code at https://github.com/tsosea2/CovidET-EXT.

# Data
We release the original Reddit posts together with the annotations in CovidET-EXT. The dataset is under the "data" folder.

# Code
To create summaries using Emotion-Aware Pagerank, run:

```
python eap.py  --emotion <emotion> --training_data_path <path_to_posts_train_csv> --test_data_json <path_to_test_proc_anon_json> --validation_data_json <path_to_validation_proc_anon_json> --force_embeddings_computation --embedding_directory <directory_path>
```

[![RevolverMaps Live Traffic Map](http://rf.revolvermaps.com/w/3/s/a/7/0/0/ffffff/010020/aa0000/5mtpsf8i5p6.png)](https://www.revolvermaps.com/livestats/5mtpsf8i5p6/)

[![ClustrMaps Tracker](https://www.clustrmaps.com/map_v2.png?d=UGBgp9pq2WpPEFTNNkQzZuv0dgvOOMXQ1-gflu9WZFk&cl=ffffff)](https://clustrmaps.com/site/1btj3)

(Traffic since March 10th, 2023)
