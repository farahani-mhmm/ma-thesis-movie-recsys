# Fuzzy logic and uncertainty modelling in recommender systems (cold start, fuzzy + LLM/neural, explainability, soft gating), 2025-2026 only

Search date: 2026-10-30. Scope: papers first published 2025-01-01 or later. Coverage is NOT exhaustive (about 35 tool calls; SciSpace semantic search, Wiley full-text search, WebSearch, WebFetch of arXiv/AAAI/Europe PMC). Verification labels used below:
- FULL = I read the full text / HTML / PDF (or substantial full-text passages).
- ABSTRACT = I read the abstract on a publisher/arXiv/Europe PMC page, or the abstract as returned by SciSpace/Wiley. Numbers are the authors' abstract claims only.
- SNIPPET = only a search-result snippet or aggregator text; treat as unconfirmed.

Headline: the 2025-2026 literature on *fuzzy + cold start* is thin, mostly low-visibility venues, and almost never benchmarked against strong baselines (popularity, SVD/BPR-MF, LightGCN, LightGBM, SASRec). The one paper I could check in full with a strong CF baseline (arXiv 2505.06000) lost to SVD on MovieLens-1M. I found NO 2025-2026 paper that fuzzifies "how cold a user is", NO paper that fuzzily gates between content-based and collaborative models, and only ONE (snippet-level) paper that combines fuzzy sets with an LLM inside a recommender.

---

## Key Question 1: What fuzzy-based cold-start / sparsity methods exist in 2025-2026, and do any beat strong non-fuzzy baselines on public data?

### Takeaway
Only a handful of 2025-2026 fuzzy recommenders even target cold start or sparsity, and none that I could verify shows a win over strong non-fuzzy baselines (SVD, BPR-MF, LightGCN, LightGBM, neural models) on public data under a proper cold-start protocol. The best-documented fuzzy-neural paper (arXiv 2505.06000) explicitly reports that SVD is significantly better on MovieLens-1M and has no dedicated cold-start experiment.

### Cited Findings

#### A. Fuzzy neural / differentiable-fuzzy models (read in full)

**A1. Differentiable Fuzzy Neural Networks for Recommender Systems** (FULL: arXiv HTML)
- Authors: Stephan Bartl, Kevin Innerebner, Elisabeth Lex. Submitted 9 May 2025, arXiv:2505.06000 (cs.LG), work-in-progress. URL: https://arxiv.org/abs/2505.06000 (HTML: https://arxiv.org/html/2505.06000v1). Code: https://github.com/stephba/diff-fnn (as stated in the paper; I did not open the repo).
- What is fuzzified / how: predefined human-readable "atoms" (fuzzy truth values of user/item features); the network learns weighted fuzzy AND/OR rules over them, so each rule is a fuzzy-logic expression. Differentiable, so it can be plugged into other neural models.
- Datasets: a synthetic dataset and MovieLens-1M (binary label: rating >= 4.0).
- Baselines: BaselineOnly, KNNBasic, SVD, LightGCN (CF); JRip and Decision Tree (on raw features and on the atoms); NCF.
- Protocol: ML-1M sorted by time, 0.7/0.1/0.2 train/val/test, so the test set contains cold-start users and items. Synthetic: random 0.7/0.1/0.2. 10 repetitions, two-sided Wilcoxon signed-rank test (p<0.01). Metrics: Precision/Recall/NDCG/MAP at k=5 and 10.
- Headline numbers: Synthetic: the model recovers the ground-truth rules, P@5 = NDCG@5 = MAP@5 = 1.000 (JRip and Decision Tree also perfect; NCF about 0.997 P@5). ML-1M: FNN P@5 0.218, R@5 0.225, NDCG@5 0.232, MAP@5 0.210; **SVD is best (P@5 0.235, NDCG@5 0.251), BaselineOnly second (P@5 0.232)**.
- Honest negatives (authors' own): SVD significantly beats the FNN on all ML-1M metrics; learned fuzzy weights on ML-1M are much lower than on synthetic data and only partly explained; hyperparameter tuning was exploratory; not optimised for ranking; rules depend on predefined atoms and random initialisation (different runs can give different rules); no user study; only two datasets.
- Cold start: no dedicated cold-start experiment and no separate numbers. For unseen users/items the atom values are set to dataset means; authors say cold-start performance is limited.
- This is the paper the caller flagged. Verified: the "not better than SVD" claim is true, but it is a general ML-1M ranking result, not a cold-start-only result.
- Usefulness for ML + TMDB project: HIGH as a reference baseline and as a template for "rules as explanations" with atoms built from TMDB features (genre, decade, runtime, popularity bucket, director). Code is public. Do not expect it to beat SVD/LightGBM.

**A2. Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks** (FULL: arXiv HTML)
- Authors: Kevin Innerebner, Stephan Bartl, Markus Reiter-Haas, Elisabeth Lex. arXiv:2601.04019, 7 Jan 2026 (date from SciSpace). URL: https://arxiv.org/abs/2601.04019. Code: anonymous link in the paper, https://anonymous.4open.science/r/FNN4NewsRecommendation/ (may expire).
- What is fuzzified: same FNN idea; learned rules over human-readable atoms (e.g., article age, user/article features) to predict clicks; rules extracted at configurable thresholds to control complexity (RC@t = number of logical operators).
- Datasets: MIND-small and EB-NeRD-small, provided temporal train/val splits; 4 sampled non-clicked articles per impression for training.
- Baselines: Random, User-KNN, Item-KNN, SVD++ (implicit), Decision Tree (same crisp atoms), NRMS (neural two-tower, the only black-box baseline). Metrics: AUC, MRR, nDCG@5/10 per impression; mean of 5 runs.
- Numbers (AUC / nDCG@10): EB-NeRD: FNN-and .6925 / .5664, FNN-or .6915 / .5662, NRMS .5548 / .4648, Decision Tree .6039 / .4840, so FNN beats every baseline there. MIND: FNN-and .5806 / .3355, NRMS .6548 / .4043, Decision Tree .5812 / .3532, so NRMS is clearly best and the Decision Tree matches/beats the FNN. Authors note better EB-NeRD results exist elsewhere (up to 79.10 nDCG@10 on test), so not SOTA.
- Cold start: no dedicated mechanism. KNN and SVD++ do badly on ranking because many test articles are absent from training (cold items under temporal split); the FNN's article-age atoms help (removing them drops EB-NeRD AUC from .692 to .600). This is indirect evidence that content/context atoms handle cold items better than ID-based CF.
- Limitations: depends on feature richness (MIND has few features; over 95% of EB-NeRD gender/age/postcode are "unknown"); offline only; rule complexity sensitive to regularisation.
- Usefulness: MEDIUM-HIGH. Good evidence for the honest framing "fuzzy rules are competitive only when informative atoms exist"; baseline set is still weak on the strong side (no LightGBM/LightGCN).

**A3. Fuzzy Collaborative Reasoning (FuzzCR)** (FULL: AAAI PDF)
- Authors: Huanhuan Yuan, Pengpeng Zhao, Jiaqing Fan, Junhua Fang, Guanfeng Liu, Victor S. Sheng. AAAI-25, vol. 39 no. 12, published 11 Apr 2025, DOI 10.1609/aaai.v39i12.33432. URL: https://ojs.aaai.org/index.php/AAAI/article/view/33432
- What is fuzzified: recommendation cast as first-order-logic query answering; learning-free fuzzy logical operators (t-norm style AND/OR/NOT) replace the parameterised neural logic modules of NCR/NLR.
- Datasets: ML-100k, Amazon Beauty, Amazon Sports. Ratings > 3 positive; last two positives per user = validation/test (NCR preprocessing). Eval: 100 sampled negatives per positive, N@5/10 and HR@5/10, mean of 10 runs.
- Baselines: BPRMF, SVD++, NeuMF, GRU4Rec, NARM, STAMP, NLR, NCR, CBox4CR (no SASRec/BERT4Rec/LightGCN).
- Numbers (ML-100k): FuzzCR N@5 0.4253, N@10 0.4719, HR@5 0.5915, HR@10 0.7385 vs best baseline CBox4CR 0.4103 / 0.4624 / 0.5741 / 0.7353. Relative gains over CBox4CR are small: +3.65%, +2.05%, +3.03%, **+0.43%** (HR@10) on ML-100k; +0.95% to +2.75% on Beauty; +1.6% to +2.2% on Sports.
- Evidence strength: weak-to-moderate. Sampled-metric evaluation (100 negatives) is known to be unreliable for ranking comparison; the baseline list omits modern strong baselines; single fixed split protocol, no significance test described; no cold-start or sparsity experiment; I found no code link.
- Usefulness: MEDIUM as a source of fuzzy-operator design (learning-free t-norms); LOW for cold start.

#### B. Fuzzy + graph / logic-rule models (abstract level)

**B1. Neural Recommendation Reasoning with Logic Rules (Recolor)** (ABSTRACT)
- Authors: Jing Yao, Xiting Wang, Jianxun Lian, Xiaoyuan Yi, Xing Xie. ACM Trans. Information Systems 43(5), 2025, DOI 10.1145/3742856, published 25 Jun 2025 per SciSpace. https://doi.org/10.1145/3742856
- Fuzzified: a "fuzzy logic validity module" estimates rule validity on highly sparse data; a neural logic generation module decouples representation learning from rule construction; BPR loss for ranking. Any neural recommender can be upgraded to a self-explainable version.
- Claims (abstract): four public datasets with different sparsity, numerical experiments plus user studies, "high-validity rule explanations, generality, and high recommendation accuracy". Datasets, baselines and numbers not verified. I found no arXiv preprint or repo in search.
- Usefulness: HIGH conceptually (rule explanations with a fuzzy validity score, sparsity-aware, user study), but verify full paper first.

**B2. Enhancing Graph-Based Recommendations via Fuzzy Clustering and Gated Multi-View Fusion (FuzzGCN)** (ABSTRACT)
- Authors: Yuanming Huang, Jie Lu, Keqiuyin Li, Guangquan Zhang. FUZZ-IEEE 2025 (6 Jul 2025 per SciSpace), DOI 10.1109/FUZZ62266.2025.11152097.
- Fuzzified: Fuzzy C-Means on item embeddings gives global item groupings; a gated fusion combines them with local graph propagation; contrastive learning aligns views. Targets sparsity. Claims to beat "several competitive baselines" on three public datasets (not named). Not read beyond abstract; web search could not locate the paper elsewhere.
- Note: the gate here is a learned feature-level gate, not a function of the number of ratings.

**B3. Fuzzy-enhanced variable weight graph convolutional networks (FVW-GCN)** (ABSTRACT)
- Authors: Wanna Cui, Hak-Keung Lam. Computers & Electrical Engineering, 8 Jan 2026 (SciSpace date), DOI 10.1016/j.compeleceng.2026.110970.
- Fuzzified: fuzzy C-means on semantic embeddings from pretrained language models to enrich the adjacency for sparse/long-tail items, plus a variable-weight GCN. Claims consistent wins in recall/NDCG/hit ratio on benchmark datasets (unnamed). Relevant because it fuses fuzzy clustering with LM embeddings. Not read beyond abstract.

**B4. FMCL (fuzzy prototypes for multi-behaviour recommendation)** (SNIPPET, low confidence)
- Only an aggregator page (synapsesocial.com) describes it: adaptive fuzzy prototypes from view-balanced embeddings, contrastive learning. No title/venue/arXiv ID confirmed. Do not cite without verification.

#### C. Uncertainty-aware fuzzy-family CF (type-2, neutrosophic, signature, classical fuzzy CF)

**C1. A neutrosophic clustering approach to handle recommendation uncertainty for gray sheep users** (ABSTRACT, via Europe PMC; full text blocked)
- Authors: Dina Samir, Eman Abd El Reheem, Saad M. Darwish, Magda M. Madbouly. Scientific Reports 16(1), first published online 2026-03-21, DOI 10.1038/s41598-026-41651-8, PMCID PMC13009167, open access. https://www.nature.com/articles/s41598-026-41651-8 (SciSpace mislabels the journal "Dental science reports"; the DOI is Scientific Reports).
- Fuzzified: neutrosophic k-means gives each user truth / indeterminacy / falsity memberships; users in the high-indeterminacy cluster ("gray sheep") get item-based CF, mainstream users get standard IBCF. Evaluation focuses only on the gray-sheep cluster.
- Numbers (abstract): ML-100K precision 88.70%, recall 90.90%, F1 89.79%, accuracy 84.07%, MAE 0.534, RMSE 0.719; also tested on Book-Crossing and Last.fm.
- Evidence strength: weak. The abstract does not name any matrix-factorization/neural baseline, precision of about 89% on a sub-cluster is not comparable to standard top-k metrics, and the protocol is unknown (could not read methods). This is gray-sheep, not cold start, but "indeterminacy as per-user uncertainty" is conceptually close to a "how uncertain/cold is this user" membership.

**C2. A Novel Neutrosophic Recommender System Model Using Entropy Theory and Collaborative Filtering Algorithm** (ABSTRACT + partial FULL passages via Wiley)
- Authors: Cu Nguyen Giap, Thi-Lich Nghiem, Le-Kim-Danh Tran, R. Murray (author list per Wiley metadata). Int. J. Intelligent Systems vol. 2026 no. 1, published 2026-06-28, DOI 10.1155/int/5244922.
- Fuzzified: crisp ratings to neutrosophic (T, I, F) via entropy-optimised fuzzification; neutrosophic operations in CF; a new ranking-accuracy metric for neutrosophic and crisp data; hyperparameter learning for fuzzification parameters. "Experiments on a public dataset" (unnamed in what I saw). I did not retrieve results tables, baselines or split. Its own related-work text claims the neutrosophic + CF combination is "fragmented and underexplored".
- Evidence: unknown. Proposes its own metric, which weakens comparability.

**C3. Modeling Rating Uncertainty with Interval Type-2 Fuzzy Decision Trees in Multi-Criteria Recommendation** (ABSTRACT)
- Authors: Heersh Azeez Khorsheed, Sadegh Abdollah Aminifar. Int. J. on Smart Sensing and Intelligent Systems, 2026, DOI 10.2478/ijssis-2026-0074.
- Fuzzified: histogram-based interval type-2 membership functions on five Yahoo! Movies criteria (overall, story, acting, direction, visuals); HDBSCAN then k-means subclusters as targets for IT2 fuzzy ID3/C4.5 trees; cosine similarity within subcluster.
- Claims consistent gains in MAE, precision, recall, F1 "over crisp and type-1 models" on Yahoo! Movies. The page I could access shows no numbers, no split, no SVD/MF/neural baselines, no code. Evidence: weak (only compared to its own crisp/type-1 variants as far as the abstract shows). Yahoo! Movies multi-criteria data is niche; not usable for ML+TMDB beyond the idea of per-criterion uncertainty bounds.

**C4. Rental Property Recommendation System Based on Fuzzy Decision Tree** (ABSTRACT + conclusion passage FULL via Wiley)
- Authors: C. Liu, B. Shao, W. Zhao, H. Zeng, N. Tian, X. Zhao, X. Liu. Computational Intelligence 42(2), 16 Mar 2026, DOI 10.1111/coin.70220.
- The only 2025-2026 paper I found that explicitly frames a fuzzy method around the cold-start problem (scarce rental history; new users/new items). Fuzzy rules preprocess user/house attributes; a "Decision Jungle" predicts desired house attributes; a matching module maps to inventory.
- Claims: Recall 99.6%, Precision 88.6%, F1 93.7%; validated with real testers (89% of recommendations rated accurate), compared against Beike/Ke.com's "House Finder" module.
- Evidence: weak for our purposes. Proprietary/private rental data, tester-based evaluation, compared with a commercial system rather than standard strong baselines, no cold-start vs warm split on public data. Recall of 99.6% suggests a very small or easy candidate set.
- Usefulness: LOW for ML+TMDB; only shows the "fuzzy attribute matching for new users" idea.

**C5. A Recommendation System Based on Fuzzy Signature** (SNIPPET/ABSTRACT via SciSpace)
- Authors: Giuseppe D'Aniello, Mario Della Corte, Matteo Gaeta. IEEE Access, 2026, DOI 10.1109/ACCESS.2026.3656087.
- Fuzzified: a "Fuzzy Signature" (a fuzzy relation representing user interests) and a "user kindredness" similarity for neighbourhood CF. Evaluated on MovieLens against "collaborative filtering, probabilistic methods and fuzzy genetic algorithms" with "good improvements in accuracy and error rates". Baselines look classical/neighbourhood-type; no MF/neural baselines named. Evidence: weak.

**C6. Ukrainian-language fuzzy CF papers by Ivohin (and Shelyakin)** (ABSTRACT only)
- "Approaches to Integration of Fuzzy Logic in User-Based Collaborative Filtering Methods", Shtuchnyi Intelekt (Artificial Intelligence), 30 Sep 2025, DOI 10.15407/jai2025.03.063: review plus proposed Gaussian-like membership functions for genre profiles of users and films; "further testing" needed, i.e. no experimental validation in the abstract.
- "Recommender model for data prediction based on fuzzy logic and collaborative filtering method" (Ivohin, Shelyakin), Nauka, tekhnolohii, innovatsii, 4 Apr 2026, DOI 10.35668/2520-6524-2026-1-06: personalised triangular membership functions, Mamdani inference for user similarity; tested only on an **artificially generated dataset** compared with basic CF on MSE/RMSE/SSE. Evidence: very weak.

#### D. Non-fuzzy but directly relevant 2026 papers on cold-user gating (for contrast)

**D1. Adaptive Cluster-Smoothed SVD: a hybrid recommender system with dynamic weight optimization** (ABSTRACT, SciSpace)
- Authors: Nayan Roy, Priyanjana Mitra, Isha Chakraborty, Avijit Saha, Karobi Bachhar, Asmita Banerjee, Krishna Kumar Jha. Journal article dated 2026-06-01, DOI 10.58532/nbennuramlab9p1c8.
- A HARD switch: cluster smoothing is applied only to users with fewer than five ratings; predictions are blended with SVD at fixed weight alpha = 0.8. ML-100K, ML-1M, Book-Crossing, Jester. Reported RMSE equal to SVD on dense data (ML-100K RMSE 0.877) and a ~0.8% RMSE *increase* on very sparse Book-Crossing; i.e., an honest "no loss for active users, slight/no gain overall" result, and RMSE is the metric (not top-k).
- This is the closest 2026 item to "switch strategy by number of ratings", but it is crisp (threshold at 5), not fuzzy. It shows the niche is open for a fuzzy version, and that gains are small versus SVD.

**D2. Cold-Start Recommendation via Hybrid Matrix Factorization and Cluster-Based Association Rules** (ABSTRACT)
- Sumitha T et al., Apr 2026 (IEEE conf., DOI 10.1109/RMKMATE69073.2026.11518964). K-means + association rules + SVD + ridge regression; with k=1 seed item per user: Precision@10 19.3%, Recall@10 4.5%. Crisp, not fuzzy; compared only with "the existing one".

#### E. Reviews

**E1. A Review of Deep Learning and Large Language Models for Cold Start Problem in Recommender Systems** (ABSTRACT, Wiley)
- Authors: C. Liu, D. Jiang, Y. Cai, H. Li. WIREs Data Mining and Knowledge Discovery 16(1), 4 Feb 2026, DOI 10.1002/widm.70068. Taxonomy by information source (content, structure, transfer, generation); states that no universal solution exists and that standardised evaluation metrics and integration strategies are research gaps. Does not centre on fuzzy methods (abstract). Useful as the context against which fuzzy methods must compete (LLM/DL cold-start methods).

### Inferences
- Fuzzy methods for MovieLens-style data have not been shown, in anything I could verify, to beat strong CF baselines; in the one case with a strong baseline and full text (A1), the fuzzy model is below SVD by about 7% relative on P@5 (0.218 vs 0.235).
- The only places where fuzzy-rule models beat baselines clearly (A2 on EB-NeRD) are those with rich content/context atoms and a temporal split where ID-based CF fails on cold items. That supports using fuzzy rules over metadata features for cold items, and not expecting gains in warm users.
- Most 2025-2026 fuzzy-CF papers in lower-tier venues use classical neighbourhood baselines, synthetic data, self-defined metrics, or in-house testers. Treat their headline numbers (e.g., F1 89.79%, Recall 99.6%) as not comparable to standard top-k metrics.

### Gaps
- Full text of C2, C3, C5, B1, B2, B3 was not read (paywalled or blocked), so datasets/baselines/numbers there are abstract-level only.
- I could not confirm any code repository for FuzzCR, Recolor, FuzzGCN, FVW-GCN, the neutrosophic papers or C3-C5.
- I did not find any 2025-2026 fuzzy paper evaluated under an explicit new-user protocol (e.g., users with 0/5/20/50 ratings) with SVD/LightGBM baselines. If it exists, it is outside what my search tools surfaced.

---

## Key Question 2: Is there work combining fuzzy logic with LLMs for recommendation, for uncertainty of LLM outputs, or for explanation?

### Takeaway
Fuzzy + LLM exists mainly for explanation outside recommendation (medical, audit video, clustering), and I found exactly one fuzzy-family + LLM recommender (neutrosophic hypergraph, snippet-level and probably education-oriented). Uncertainty of LLM recommendation outputs is being studied in 2025-2026, but with evidential/Dirichlet/entropy methods, not fuzzy sets.

### Cited Findings

#### A. Fuzzy + LLM inside a recommender
- **LLM-Semantic Enrichment of Neutrosophic Fuzzy Hypergraphs for Indeterminacy-Driven Reinforcement Learning and Explainable Path-Based Recommendations** (ABSTRACT via SciSpace). Authors: Mehbooba P. Shareef, Babita Roslind Jose, Jimson Mathew. IEEE Access, 2026, DOI 10.1109/ACCESS.2026.3715708. Users are hyperedges, items hypernodes; neutrosophic fuzzy hypergraph captures uncertain multi-faceted interactions; LLM plays three roles (infer item-item prerequisite/dependency relations, shape the RL reward through feedback, generate natural-language explanations from paths); R-GCN refines preference vectors; a DQN does indeterminacy-aware exploration/exploitation. Claims better NDCG, hit ratio, precision, recall on "real-world datasets" (unnamed in abstract). The words "prerequisite" and "pedagogical alignment" suggest an e-learning setting (my inference). Evidence: unverified, snippet-level; baselines unknown. Closest thing found to the target idea but not MovieLens-oriented.
- **FVW-GCN** (see Q1-B3) uses fuzzy C-means over pretrained language-model embeddings of reviews/descriptions: a fuzzy + LM-embedding recommender, not an LLM-generation one.
- Web search for "LLM + fuzzy linguistic terms + recommendation" (arXiv/Scholar-style queries) returned nothing matching; the search tool itself flagged this as an apparent open niche.

#### B. Fuzzy rule-based systems + LLM for explanation (not recommender, but transferable method)
- **Generating trustworthy explanations with a language model enriched by fuzzy rule-based systems.** P. M. Perez-Ferreiro, A. Catala, A. Bugarin, J. M. Alonso. FUZZ-IEEE 2025 (6 Jul 2025), DOI 10.1109/FUZZ62266.2025.11152139. Fuzzy rule firing strengths are given to an LLM to verbalise; open-source modular architecture; validated in the medical domain; reports significantly fewer hallucinations (Wilcoxon signed-rank, p<0.001) and more compact narratives without accuracy loss. ABSTRACT.
- **Fusion of fuzzy rule-based factual and counterfactual explanations with large language models.** Same group, Int. J. Approximate Reasoning, 22 Sep 2026, DOI 10.1016/j.ijar.2026.109828. Adding counterfactuals neither increases hallucination nor lowers accuracy; **Chain-of-Thought prompting did not help: it increased verbosity and harmed readability, "with no apparent benefits"** (honest negative). Medical use case. ABSTRACT.
- **A Human-Centered Explainability Framework for Latent Clustering Based on NMF, Fuzzy Logic, and LLMs.** Casalino, Castellano, Kaczmarek-Majer et al., FUZZ-IEEE 2026 (June 2026), DOI 10.1109/FUZZ69877.2026.11626367. Fuzzy logic maps NMF latent factors to linguistic concepts that structure LLM prompts; six prompting strategies, eleven open-source LLMs, acoustic bipolar-disorder task. NMF latent factors are also what matrix-factorization recommenders have, so the pattern (latent factor -> fuzzy linguistic concept -> LLM text) is directly transferable. ABSTRACT.
- Also noted (ABSTRACT, not recsys): Mady, Hagras et al., "A fuzzy logic based generative explainable AI for automated video auditing", J. Ambient Intell. Humaniz. Comput., 9 Oct 2026, fuzzy + deep learning + RAG; V. Figueiredo, "A Fuzzy Logic Prompting Framework for Large Language Models in Adaptive and Uncertain Tasks", arXiv:2508.06754 (8 Aug 2025), tutoring; a 2026 RusAutoCon paper on fuzzy inference + ML + RAG for log events (Mamdani, criticality score as filter for retrieval).
- Caution on quality: a Preprints.org 2025 "Fuzzy Logic and Deep Learning for a Knowledge-Driven Modeling-Based Recommendation System" (DOI 10.20944/preprints202502.1674.v1) describes attack detection and financial goals with unclear experiments; I rate it unreliable. A Nov 2024 "FuzzyRec" paper reporting "mean RMSE of 89.7%" is dated 2024 (out of scope) and its numbers are implausible.

#### C. Uncertainty of LLM recommendation outputs (non-fuzzy, evidential/entropy)
- **Uncertainty Quantification and Decomposition for LLM-based Recommendation.** Wonbin Kweon, Sanghwan Jang, SeongKu Kang, Hwanjo Yu. WWW 2025, arXiv:2501.17630 (v2 12 Feb 2025). Predictive uncertainty of LLM recommendations decomposed into recommendation uncertainty and prompt uncertainty; shows uncertainty indicates reliability; proposes uncertainty-aware prompting. Code and weights: https://github.com/WonbinKweon/UNC_LLM_REC_WWW2025. ABSTRACT (datasets/baselines/numbers not retrieved).
- **GUIDER: Uncertainty Guided Dynamic Re-ranking for LLM-based Recommender Systems.** Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao, Meng Yan. AAAI-26, vol. 40 no. 19, 14 Mar 2026, DOI 10.1609/aaai.v40i19.38639. LLM logits used as evidence for a Dirichlet distribution; splits data uncertainty vs model uncertainty in one pass; four-quadrant policy adjusts output diversity; zero-shot setting. Evidential, not fuzzy. ABSTRACT.
- **EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking.** Meng Yan, Cai Xv, Xujing Wang, Ziyu Guan, Wei Zhao. arXiv:2606.04727, 3 Jun 2026. Three evidences from one forward pass, "reliable opinion aggregation" (the page does not name subjective logic; that link is my inference), position-aware calibration, calibrated confidence guides ranking optimisation; three datasets. ABSTRACT; no code link found.
- **Trustworthy LLM-based Recommenders: Uncertainty, Conformal Prediction, and Hallucination Mitigation** (survey), S. Zammali, IC3IT 2026 (26 Mar 2026), DOI 10.1109/IC3IT69043.2026.11567549. ABSTRACT.
- Other 2025-2026 explainable/cold-start LLM recommenders without fuzzy logic (for context): CogRec (arXiv:2512.24113, Dec 2025; LLM + Soar production rules, symbolic not fuzzy); M-LLM3REC (CIKM 2025, DOI 10.1145/3746252.3761214; LLM motivation extraction for cold start); a 2025 Frontiers review "On explaining recommendations with LLMs" (Alan Said) found only 6 of 232 articles directly on the topic.

### Inferences
- Nobody, in what I surfaced, has used a fuzzy/linguistic layer to quantify or calibrate the uncertainty of an LLM recommender; all LLM-rec uncertainty work uses Dirichlet/evidential or entropy/logit-based confidence. A fuzzy "confidence membership" over those scores would be a new angle, but it would need to be argued as more than relabelling a calibrated probability.
- The most reusable 2025-2026 fuzzy+LLM pattern is the verbalisation pipeline (fuzzy rule activations -> structured prompt -> narrative; measure hallucination/faithfulness). It was validated only in medicine; no recommender version found. The IJAR 2026 negative result (CoT prompting hurts readability) is worth carrying into a design.

### Gaps
- No peer-reviewed 2025-2026 paper found applying fuzzy sets to LLM *ranking/recommendation outputs* or to LLM-generated user profiles.
- The neutrosophic-hypergraph + LLM paper's datasets, baselines and numbers could not be verified.
- Datasets/baselines of Kweon et al., GUIDER and EviRank not retrieved (fetch returned abstract pages only).

---

## Key Question 3: Which fuzzy ideas are most defensible and novel enough for a small MovieLens + TMDB practice project (e.g., a fuzzy "how cold a user is" membership, fuzzy rule explanations)?

### Takeaway
A fuzzy membership of user coldness used as a soft gate between content-based/LLM and collaborative components is open ground in 2025-2026 (closest found: a crisp "<5 ratings" switch in a 2026 hybrid and indeterminacy-based gray-sheep clustering), and fuzzy rule explanations over human-readable atoms have a public codebase to compare against. Both must be benchmarked against strong baselines and a learned gate; small effects should be reported as small.

### Cited Findings
- No 2025-2026 paper found that defines a membership function over the number of a user's ratings (very cold / cold / warm) and blends content-based and CF scores by membership degree. The nearest items: a hard switch at fewer than 5 ratings with fixed alpha 0.8 blend and essentially no gain over SVD in aggregate (D1 above), neutrosophic indeterminacy as a per-user uncertainty signal (C1), and learned feature-level gating in FuzzGCN (B2).
- Fuzzy rule explanations with a public implementation: Bartl/Innerebner/Lex FNN (https://github.com/stephba/diff-fnn); it lost to SVD on ML-1M but produced readable rules, and in news recommendation beat baselines when informative atoms existed (A1, A2).
- Fuzzy-rule-to-LLM verbalisation with measurable faithfulness (hallucination counts, Wilcoxon tests) exists in medicine (Q2-B); no recommender instance found.
- Honest-reporting precedent: the 2025-2026 papers worth imitating methodologically are those that report negative or small results: arXiv 2505.06000 (SVD wins), arXiv 2601.04019 (NRMS wins on MIND), ACSS 2026 (slightly worse on sparsest data), IJAR 2026 (CoT hurts).

### Inferences (my own reasoning, not from sources)
1. **Fuzzy coldness gate (most novel, most defensible).** mu_cold(n) from the user's rating count (e.g., smooth membership in {none, few, some, many}, optionally also from rating variance or recency), final score = mu * content/LLM score + (1 - mu) * CF score. Defensibility depends on comparing against: (a) hard thresholds at several n, (b) a learned gate or a GBM (LightGBM/hist-gradient-boosting) given n_ratings as a feature, which already performs implicit soft switching and is the strong baseline to beat, (c) popularity and SVD/BPR, evaluated at N = 0/5/20/50 ratings with a leave-latest or time-based split. Expect small gains at best; report them as such.
2. **Fuzzy genre/taste atoms as explanations.** Use TMDB-derived atoms (genre membership degrees, decade, runtime class, popularity class, director fame) in the FNN of A1; explain with learned rules; evaluate faithfulness and accuracy loss versus LightGBM. Novelty is modest (reuses A1) but the TMDB-enriched atoms and a cold-user focus are a legitimate extension; the honest hypothesis is "no accuracy gain, comparable interpretability".
3. **Fuzzy linguistic layer + LLM verbalisation.** Convert rule activations or latent factors into linguistic labels ("strong match on sci-fi, weak on runtime"), feed to an LLM, measure hallucination against rule truth (as in Perez-Ferreiro et al.). Novel for recsys; evaluation must be a faithfulness metric, not accuracy.
4. **Neutrosophic/IT2 "indeterminacy" as user-uncertainty** (C1-C3): conceptually interesting, but the three papers lack strong baselines; use only as a feature or as a justification for the gate, not as the headline contribution.
5. **Avoid** claiming fuzzy superiority from classic fuzzy-CF papers with synthetic data or self-defined metrics (C4, C6).

### Gaps
- No evidence about whether any of these ideas beats a GBM with an n_ratings feature; that is exactly what a project would need to test.
- I did not find 2025-2026 fuzzy work using TMDB-style metadata.

---

## Evidence-strength summary (strong vs weak)

| Paper | Strong baselines (pop / MF / LightGBM / neural / LLM)? | Proper split? | Cold-start tested? | Read |
|---|---|---|---|---|
| Bartl et al. 2505.06000 | SVD, LightGCN, NCF, KNN, trees: yes, no LightGBM, no popularity | Temporal 0.7/0.1/0.2 + 10 runs + Wilcoxon | No dedicated; limited, no numbers | FULL |
| Innerebner et al. 2601.04019 | NRMS, SVD++, KNN, DT: reasonable for news | Provided temporal splits, 5 runs | Indirect | FULL |
| FuzzCR AAAI-25 | BPRMF, SVD++, NeuMF, seq. models, CR models; no SASRec/LightGCN | Leave-last-two, but 100 sampled negatives | No | FULL |
| Recolor TOIS 2025 | Unknown | Unknown | Sparsity levels (claimed) | ABSTRACT |
| FuzzGCN / FVW-GCN | Unknown ("competitive baselines") | Unknown | Sparsity (claimed) | ABSTRACT |
| Neutrosophic gray sheep (Sci Rep 2026) | None named in abstract | Unknown | Gray-sheep, not cold | ABSTRACT |
| Neutrosophic entropy CF (IJIS 2026) | Unknown, own metric | Unknown | Sparse/noisy (claimed) | ABSTRACT |
| IT2 fuzzy trees (2026) | Only crisp/type-1 variants | Unknown | No | ABSTRACT |
| Rental fuzzy tree (2026) | Commercial module + "similar methods" | Real testers | Claimed cold start | ABSTRACT + conclusion |
| Fuzzy signature (IEEE Access 2026) | Classical CF, probabilistic, fuzzy GA | Unknown | No | SNIPPET |
| Ivohin 2025 / 2026 (Ukrainian) | Basic CF only; synthetic data | n/a | No | ABSTRACT |
| ACSS hybrid 2026 (non-fuzzy) | SVD (is the baseline) | Unknown | Users < 5 ratings | ABSTRACT |

Overall: weak. Of 2025-2026 fuzzy recommender papers surfaced, only three (A1, A2, A3) were checked against mainstream CF/neural baselines in full text, and gains are either negative (A1), mixed (A2), or small with sampled metrics (A3). None compares to LightGBM or LLM-based recommenders.

---

## Older work mentioned (pre-2025, listed only because 2025-2026 papers build on it or for orientation)
- Neural Collaborative Reasoning (Chen et al., WWW 2021, arXiv:2005.08129): the logic-neural baseline FuzzCR (AAAI-25) replaces with learning-free fuzzy operators.
- Cbox4CR (Liang et al. 2023), NLR (Shi et al. 2020): baselines inside FuzzCR.
- Zokaei, Shahrokhzadeh, Farahani, Hosseinalizdeh, "Mitigation of the cold start problem in movie recommender systems using fuzzification of users' social network information", QICAR 2024, DOI 10.1109/QICAR61538.2024.10496631: the user's own earlier paper (fuzzy rating measure per user-item-context on the CSP dataset); returned by SciSpace; it is 2024 so out of scope for the catalogue but is the base the 2025-2026 work should be compared with.
- Fuzzy C-means for user cold start (Gupta and Goel, 2018) and type-2 fuzzy linguistic CF (Darwish et al., 2014): classic references that recent Egyptian/Indian fuzzy-CF papers still follow.
- Kweon et al. build on prior LLM-rec uncertainty (e.g., "To Believe or Not to Believe Your LLM", 2024).

---

## Tool notes
- **SciSpace search-papers** worked best: natural-language questions returned relevant 2025-2026 items (it indexes up to at least Oct 2026, including FUZZ-IEEE 2025/2026, IEEE Access, AAAI-26, arXiv 26xx IDs) with abstracts. Limits: every call returns about 15-25k tokens (8-10 papers with long abstracts), so budget context; many hits are pre-2025 and must be date-filtered; metadata errors (Scientific Reports labelled "Dental science reports"; dates sometimes index dates rather than publication dates; author lists incomplete); no full text for most items; it cannot filter by year.
- **Wiley full-text search** (with start_year=2025): useful for 3 relevant Wiley/Hindawi items (Computational Intelligence rental paper, IJIS neutrosophic CF, WIREs cold-start review) with real full-text passages and exact dates, but only 8 unique articles per query and the response (89k characters) was too large for inline return; I parsed the saved JSON file with Python. Fuzzy-systems journals outside Wiley (IEEE TFS, FSS, IJAR, Information Sciences, Applied Soft Computing) are not covered by it.
- **WebSearch**: weak for niche fuzzy-recsys topics; returns generic summaries and sometimes irrelevant older papers. Useful for finding the Recolor details, Europe PMC and FuzzCR AAAI pages, and for confirming absence ("no 2025 fuzzy cold-start LLM paper surfaced").
- **WebFetch**: arXiv abs and HTML pages are excellent (full tables, protocols); AAAI abstract pages work; AAAI PDF came back as binary but is saved locally and can be read with PyMuPDF (`import pymupdf`; note `python -I` hides user-site packages, so run without -I from a clean directory). Nature.com redirects to an auth page; PMC shows a reCAPTCHA; Europe PMC REST API gives abstracts of open-access papers; reference-global (Sciendo) shows abstract only; IEEE Xplore and ScienceDirect were not fetched and typically block bots, so IEEE Access/FUZZ-IEEE/Elsevier items remain abstract-level.
- **Coverage limits**: I did not query Google Scholar, DBLP, Semantic Scholar API, OpenReview, or IEEE Xplore directly; I did not run a systematic arXiv listing search by category and month. Exceeded the suggested ~15 tool calls (about 35) because the brief asked for a dense catalogue. Absence claims ("no paper found") mean "not surfaced by these tools", not proof of non-existence.
