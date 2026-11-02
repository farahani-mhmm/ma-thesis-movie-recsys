# Cold-start recommendation: benchmarks, datasets, evaluation protocols (2025-2026) and social-media/profile-based new-user cold start

Scope: only papers first published in 2025 or 2026 (today = 2026-10-30). Verification labels used below:
- **[FT/AbsV]** = I fetched the arXiv abstract page and/or HTML/PDF text in this session and the facts below come from it.
- **[Snippet]** = search-result snippet or aggregator only, not opened at the primary source.
Where a fetched page did not state a detail (dataset version, split, code link) I say "not stated on the page I read" rather than guess. arXiv IDs in the 2603-2608 range are as returned by the tools; I did not independently check arXiv's calendar for them.

---

## Q1. Which cold-start protocol, dataset version and metric set is most common in 2025-2026 papers on MovieLens, so a new project can compare against published numbers?

### Takeaway
There is no single standard. In the 2025-2026 papers I could open, MovieLens appears as ML-1M or ML-20M (not 32M), and metrics are Recall/NDCG or HR/NDCG at K in {5, 10, 20, 50}; item cold-start splits range from an 80/20 item split to a time-based 90/10 split. The closest thing to a shared reference is the ColdRec toolbox (unified division of data, cold user / cold item / warm tasks), plus the recent explicit protocols listed below; a practice project should copy one of them exactly and state it, rather than expect to match "the" published number.

### Cited Findings

**A. Protocol-defining / protocol-critiquing papers (most useful to copy)**

1. **Cold-Starts in Generative Recommendation: A Reproducibility Study** — Zhen Zhang, Jujia Zhao, Xinyu Ma, Xin Xin, Maarten de Rijke, Zhaochun Ren. arXiv 2603.29845 (cs.IR), v1 31 Mar 2026, v2 6 Apr 2026; search results state it is a SIGIR'26 reproducibility paper (the SIGIR'26 claim comes from a search snippet only). [FT/AbsV: abstract page + HTML] — https://arxiv.org/abs/2603.29845 , https://arxiv.org/html/2603.29845v1
   - What it does: unified cold-start protocols for generative recommenders (SASRec, GRU4Rec via RecBole; TIGER, LC-Rec, OneRec, P5, etc.); no new model.
   - Datasets: Amazon-Toys (footnote points to Amazon Reviews 2023, exact version not given), MicroLens, Steam. **MovieLens not used.** 5-core filtering on all.
   - Cold-item split: interactions sorted chronologically, first 90% train; the last 10% is validation+test (division between them not stated). A test interaction is "cold item" if the target item first appears after the 90% timestamp.
   - Cold-user split: 10% of users randomly held out of training; evaluated on interactions after the 90% timestamp; each cold user keeps 1-10 random historical interactions as context.
   - Metrics: Recall@10 and NDCG@10 (K=10 primary), averaged over test users; the text says items scored from the full catalog (full vs sampled ranking not stated explicitly).
   - Headline results: item cold-start much harder than user cold-start; scaling TIGER Flan-T5-small to XL gives only marginal cold-start gains; textual identifiers help item cold-start but hurt warm/user cold; atomic IDs collapse on item cold-start; RL does not reliably help.
   - Dataset stats (Table 1): e.g. Amazon-Toys 17,479 warm users / 1,942 cold users / 11,791 warm items / 133 cold items / 148,185 interactions; MicroLens 45,000/5,000/18,212/1,008/359,709; Steam 26,889/2,987/29,677/2,417/178,961.
   - Code: https://anonymous.4open.science/r/ColdGenrec-0DEC (anonymous link; may not persist).
   - Reproducibility caveats I noticed: val/test split of the 10% unspecified; some numbers differ between tables (e.g. OneRec item cold-test R@10 0.0045 vs 0.0043); significance tests claimed without seed/run details.
   - Reuse: best-documented 2026 template for a time-based cold-item split + random cold-user split with 1-10 context interactions. Would need re-running on MovieLens yourself.

2. **Recommendation Is a Dish Better Served Warm** — Danil Gusak, Nikita Sukhorukov, Evgeny Frolov. RecSys 2025; arXiv 2508.07856 (11 Aug 2025). [FT/AbsV: abstract + HTML] — https://arxiv.org/abs/2508.07856 ; code https://github.com/dalibra/recommendation-dish-better-served-warm
   - Argues cold-user/cold-item cutoffs are arbitrary and inconsistent across papers; shows data-derived thresholds.
   - Datasets: MovieLens-1M (6,040 users, 3,706 items), BeerAdvocate, Behance, Amazon Beauty. Baselines EASE^R, PureSVD, ItemKNN, SASRec-CE.
   - Split: global time point at the 0.9 quantile of interactions (they say this avoids leave-one-out leakage); validation users' last interaction is the target; item analysis uses 1,000 sampled items; user analysis keeps each test user's first post-timepoint interaction; user-based metrics NDCG@10/HR@10 unsampled; item-based HR*/NDCG* at K=1,5,10,50,100.
   - Findings: item-side cold/warm thresholds about 6-15 interactions per item (about 9-10 on ML-1M); user-side about 4-12 interactions per user; SASRec-CE needs more history than classical models. Criticises p-core filtering as heuristic. Recommends adopting the maximum threshold across models for fairness.
   - Reuse: if the project defines "cold" by N interactions (e.g. N=0/5/20/50), cite this paper to justify N values and report sensitivity to N.

3. **On Inherited Popularity Bias in Cold-Start Item Recommendation** — Gregor Meehan, Johan Pauwels (Queen Mary Univ. of London). RecSys '25 (Prague, 22-26 Sep 2025); arXiv 2510.11402 (13 Oct 2025). [FT/AbsV: HTML] — https://arxiv.org/abs/2510.11402 ; code https://github.com/gmeehan96/Cold-PopBias
   - Datasets (from MMRec): Amazon Clothing (39,387 users / 23,033 items / 278,677 interactions), Amazon Electronics (192,403 / 63,001 / 1,689,188), MicroLens (98,129 / 17,228 / 705,174).
   - Split: items 80:20 warm:cold; cold items split 50:50 val/test; warm interactions 80:10:10.
   - Metrics: NDCG@k and Recall@k, k=20 and 50; plus item-fairness (MDG) and Gini-Diversity.
   - Models: Heater, GAR, GoRec (each with/without proposed magnitude scaling), warm teacher FREEDOM, content-KNN baseline.
   - Finding: cold-start generative models inherit the warm teacher's popularity bias and over-expose cold items resembling popular warm ones; a simple KNN content baseline often beats the cold-start models on accuracy and fairness (esp. k=20).
   - Reuse: the cleanest "20% of items are cold, split 50/50 val/test, Recall/NDCG@20/50" recipe I verified. Also supports including a content-KNN baseline.

4. **Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap** — Wei Zhang, Yuanchen Bei, et al. (18 authors, incl. James Caverlee, Fakhri Karray, Irwin King, Philip S. Yu). arXiv 2501.01945, 3 Jan 2025. [AbsV via SciSpace abstract + search snippets; full text not opened] — https://arxiv.org/abs/2501.01945 ; list https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation
   - Search-snippet statements (not verified in full text): evaluation across cold-start papers is "diverse and inconsistent"; datasets rarely overlap; cold users/items are defined by time range, interaction count or random selection with no unified rule; RecBole/Elliot/BARS are general-purpose, not cold-start specific.

5. **ColdRec: open-source benchmark toolbox for cold-start recommendation** (YuanchenBei/ColdRec; undated page). [FT: GitHub README] — https://github.com/YuanchenBei/ColdRec
   - Unified pipeline and "unified division" of datasets; tasks: cold user/item, warm user/item, overall user/item; 26 models in table (DeepMusic, KNN, DUIF, VBPR, DropoutNet, AMR, MetaEmbedding, MTPR, LARA, Heater, AGNN, CLCRec, GAR, GoRec, CCFRec, ALDI, USIM, CGRC, FS-GNN, M2VAE...; backbones NGCF, LightGCN, NCL, SimGCL, XSimGCL, BPR-MF). Newest listed models: FS-GNN (AAAI 2025), M2VAE (AAAI 2026). The README page I read does not state split rules, metrics or dataset versions (it points to data/README.md; preprocessed data on Google Drive). Not itself a 2025-26 paper; it is the tooling most 2025-26 cold-item papers from that group appear to share (inference from the model list).

**B. MovieLens as used in 2025-2026 cold-start papers**

6. **M2VAE: Multi-Modal Multi-View Variational Autoencoder for Cold-start Item Recommendation** — Chuan He, Yongchao Liu, Qiang Li, Chuntao Hong, Wenliang Zhong, Xin-Wei Yao. arXiv 2508.00452 (Aug 2025); AAAI 2026, vol. 40(17) pp. 14811-14819. [FT/AbsV: arXiv HTML v2] — https://arxiv.org/abs/2508.00452 ; code https://github.com/hchchchchchchc/M2VAE
   - Datasets: **MovieLens-20M**, Amazon Video&Games, Amazon Sports&Outdoors (versions not stated on the page). Metrics HR@5, NDCG@5, HR@10, NDCG@10; 5 runs averaged. Baselines CLCRec, CCFCRec, PAD-CLRec, DropoutNet, MTPR, LARA, CVAR, GAR, MVDGAE, GoRec, CVAE, MVGAE, VMoSE. Relative gain over second-best on ML-20M only 1.5-2.6%.
   - Split and preprocessing not described on the page I read. Training uses sampled positive/negative pairs.
   - Reuse: for a ML-20M cold-item comparison use HR/NDCG@5/10; check the repo for the split before claiming comparability.

7. **Enhancing Collaborative Filtering With LLM Embeddings for User Cold-Start Recommendation (LLM-CF)** — Eranki Ramakrishna Nageswara Rao, Nellutla Saketh Koundinya, Sowmini Devi Veeramachaneni, Arun K. Pujari. PRML 2026 (IEEE), 1 Jul 2026, DOI 10.1109/prml71097.2026.11687003. [Snippet: SciSpace abstract only]
   - MovieLens-1M and Amazon Office Products; user cold-start NDCG@10 improved 3.4x and 2.0x over BPR-MF (as reported in abstract). Frozen sentence-encoder embeddings of user/item text fused via a warmth-aware gate. Admits specialised methods can win on item cold-start in extreme sparsity.

8. **LLM-FSGNN** — Di Jin, Zhizhi Yu, Songyuan Lei, Dongxiao He. ACM journal (DOI 10.1145/3811919), 4 Jun 2026. [Snippet: abstract] — rating-prediction (MAE) style: 4.8% lower MAE on MovieLens100K, 3.6% on MovieLens1M, 4.2% on Amazon. Shows that a rating-prediction/MAE protocol (MeLU/MAMO-style) still coexists with top-K protocols.

9. **Vectorized Context-Aware Embeddings for GAT-Based Collaborative Filtering** — Danial Ebrat, Luis Rueda. arXiv 2510.26461 (31 Oct 2025). [Snippet: abstract] — LLM-generated concise user profiles and item metadata (titles, genres, overviews) as GAT node features; MovieLens 100K and 1M; Precision, NDCG, MAP; BPR+cosine hybrid loss. Relevant to "item overviews as text features" and user-profile text.

10. **Large language model with retrieval semantics for cold-start recommendations (LaReS)** — Zelong Wu et al., Neural Networks, 20 Jul 2026, DOI 10.1016/j.neunet.2026.109401. [Snippet: abstract] — reports cold-start P@5 on MovieLens +7.98%; code https://github.com/wuzelong/LaReS (link taken from abstract).

11. **MI4Rec** (Zaiyi Zheng et al., CIKM 2025, DOI 10.1145/3746252.3761313). [Snippet: Snap Research page via search] — Yelp and Amazon, not MovieLens; average +20.4% Recall/NDCG over best baselines; code github.com/zhengzaiyi/MI4Rec (as reported by the search result).

12. **Are Large Language Models Really Effective for Training-Free Cold-Start Recommendation?** — Genki Kusano, Kenya Abe, Kunihiro Takeoka (NEC). arXiv 2512.13001, 15 Dec 2025. [FT/AbsV: abstract + HTML] — https://arxiv.org/abs/2512.13001
   - Closest to a MovieLens-1M user-cold-start LLM protocol: ML-1M with profile (age, gender, occupation) turned into text, a Kaggle job dataset (with/without experience), Amazon Review Music/Movie/Book (no profiles). 500 users/dataset with at least m+3 interactions; three most recent held out as positives; candidate set = 50 items (3 positives + 47 random unseen items, i.e. **sampled evaluation**); narrow cold-start m=0 (profile only), broad cold-start m>=1. Recall@10 / nDCG@10 (supplement @5 with 10 candidates). LLM rerankers gpt-4.1-mini, gpt-4.1, Qwen3-8B vs text-embedding models (multilingual-e5-large, bge-m3, gte-modernbert-base, gte-Qwen2, Qwen3-Embedding-0.6B/8B) and BM25.
   - Result: narrow cold-start (all datasets) BM25 R@10/nDCG@10 0.315/0.362, Qwen3-Embedding-8B 0.392/0.430, gpt-4.1 0.311/0.283, Qwen3-8B 0.114/0.101. TEMs beat LLM rerankers, also in warm start.
   - No code link on the page I read.
   - Reuse: directly applicable to an LLM-based new-user MovieLens project; but its 50-candidate sampled ranking (47 random negatives) is not comparable to full-ranking numbers and should be labelled as such.

**C. Other 2025-26 cold-start method papers using MovieLens/Amazon (listing only; snippet level)**
- LLM Powered Sequential Recommender (LLMSeqRec), Connor Lee, Bin Dong, Linsey Pang, ICDMW 2025 (12 Nov 2025): Amazon + MovieLens, HR@10/NDCG@10. [Snippet]
- BPE token-level embedding initialization with LLM, Yushang Zhao et al., arXiv 2509.13179 (17 Sep 2025): Recall@k/NDCG@k/HR, "benchmark datasets" unnamed in abstract. [Snippet]
- Improving Cold-Start User Recommendation via LLM-Judged Signals, Dae Seob Shin, Taehyun Ha, IMCOM 2026 (4 Jan 2026): Amazon All Beauty and Appliances; Recall/HR for cold users. [Snippet]
- CDLD-CSP (Young-Gi Hong et al., IEEE Access 2025): RMSE/MAE and new metrics CES@K / ACES to quantify adaptation efficiency with few interactions. [Snippet]
- Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation (Subbaraman et al., Google; arXiv 2511.21989, 27 Nov 2025): RL policy picks which users get LLM-simulated feedback; Amazon Product Review data (categories/versions not stated on the page I read), cold-item recall. [AbsV: abstract page only]. Extends the ColdLLM (2402.09176) augmentation line.
- LLM Reasoning for Cold-Start Item Recommendation (Shijun Li et al., Netflix; arXiv 2511.18261, 23 Nov 2025): proprietary Netflix data, SFT/RL fine-tuning; reported up to 8% above production ranker. [Snippet] Not reproducible, but shows an industry protocol.
- FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation — Ruochen Liu, Hao Chen, Yuanchen Bei, ..., Senzhang Wang; arXiv 2502.16924, 24 Feb 2025. [AbsV: abstract page] Alibaba deployment (>1 billion cold items in two months, >30x efficiency vs prior methods). The experimental section could not be read (PDF was binary); its dataset/split details are NOT verified here. No code link on the page.

### Inferences
- If the project is on MovieLens, the safest comparable choices are ML-1M (used in 2025-26 by Gusak et al., Kusano et al., LLM-CF, Ebrat & Rueda) or ML-20M (M2VAE). I found no 2025-26 cold-start paper in this pass using ML-32M, and no confirmation of ML-100K beyond rating-prediction/GNN papers.
- Metric set to report for max comparability: Recall@K and NDCG@K at K=10 and 20 (plus HR@5/10 for M2VAE comparability), full ranking over the catalogue. Add accuracy/AUC/F1 only as additional columns since top-K papers rarely report them.
- Split to copy: (i) item cold-start: 80/20 random item split with cold items split 50/50 val/test (Meehan & Pauwels) or time-based 90/10 with "target item first appears after the cut" (Zhang et al.); (ii) user cold-start: hold out 10% of users and give 1-10 (or N=0/5/20/50) context interactions (Zhang et al.; Gusak et al. for choice of N).

### Gaps
- No 2025-26 paper found that gives a MovieLens-specific, fully specified split AND releases code AND reports full-ranking numbers (M2VAE's split is not stated on the page; ColdRec README did not show metrics).
- FilterLLM, ColdRec data README, M2VAE repo split code, MI4Rec datasets were not opened at the level needed; check them before claiming "same as ColdLLM/FilterLLM protocol".
- Whether the SIGIR'26 acceptance of 2603.29845 is correct is unverified (snippet only).

---

## Q2. What do recent papers say about evaluation mistakes (leakage via popularity or text, unfair baselines, sampled-metric bias)?

### Takeaway
2025-26 papers document (a) arbitrary "cold" thresholds, (b) cold-start models inheriting popularity bias and being beaten by simple KNN/popularity/embedding baselines, (c) LLMs having memorised MovieLens-1M (a text/leakage risk for LLM-based recommenders), and (d) evaluation setups that differ in sampled vs full ranking. I did not find a 2025-26 paper specifically re-deriving sampled-metric bias; only an older foundation (see older section).

### Cited Findings
- **Threshold arbitrariness**: Gusak et al. (RecSys'25, 2508.07856) show inconsistent cold thresholds can discard valuable data or mislabel cold instances as warm; p-core filtering choice is heuristic; they use a 0.9-quantile global time cut instead of leave-one-out to avoid leakage. [FT/AbsV] — https://arxiv.org/abs/2508.07856
- **Inherited popularity bias**: Meehan & Pauwels (RecSys'25, 2510.11402): cold-start models inherit popularity bias from the warm teacher; content-KNN often beats Heater/GAR/GoRec. [FT/AbsV] — https://arxiv.org/abs/2510.11402
- **LLM memorisation of MovieLens-1M (leakage risk)**: **Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M** — Dario Di Palma, Felice Antonio Merra, Maurizio Sfilio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia. arXiv 2505.10212 (15 May 2025); related DOI 10.1145/3726302.3730178 suggests a SIGIR 2025 short paper (the venue was not on the abstract page; the SIGIR claim comes from search results). Abstract: GPT and Llama families all show some memorisation of ML-1M item attributes, user profiles and interactions; recommendation performance is related to degree of memorisation; memorisation may amplify popularity bias. The 80.76% item (titles from IDs) and 16.52% user-attribute figures for GPT-4o come from a third-party summary only [Snippet]. [FT/AbsV for the abstract] — https://arxiv.org/abs/2505.10212 ; code https://github.com/sisinflab/LLM-MemoryInspector
- **Detecting memorisation**: **Exploring Approaches for Detecting Memorization of Recommender System Data in LLMs** — Antonio Colacicco, Vito Guida, Dario Di Palma, Fedelucio Narducci, Tommaso Di Noia. Workshop SmaLLEXT@CIKM'25 (14 Nov 2025); arXiv 2601.02002 posted 5 Jan 2026 (so strictly an arXiv-2026 posting of a 2025 workshop paper). ML-1M, LLaMA models. Jailbreak prompting rarely recovered records; CCS/Cluster-Norm separated real from fake titles (balanced accuracy 0.92/0.94) but chance on users/ratings; automatic prompt engineering recovered up to ~18% (LLaMA-1B) and ~26% (LLaMA-3B) of items exactly. [FT/AbsV] — https://arxiv.org/abs/2601.02002
- **Strong simple baselines vs LLM rerankers**:
  - Lemdiasova & Zmanovskii, "Diagnosing LLM-based Rerankers in Cold-Start Recommender Systems: Coverage, Exposure and Practical Mitigations", arXiv 2604.16318 (page says submitted 9 Feb 2026, but the ID prefix implies April 2026; date inconsistency unresolved). Serendipity-2018 movie data, 500 users; popularity ranking HR@10 0.268 vs LLM reranking 0.008 (p<0.001); rerankers' Recall@200 0.109 vs 0.609 for baselines; rerankers concentrated on 3 unique items vs 497 for random. [FT/AbsV: abstract page] — https://arxiv.org/abs/2604.16318 ; code https://github.com/nikita-zmanovskiy/cold-start-algorithm
  - Dong, Qin, Shah, Wang, "Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation", arXiv 2606.29947 (29 Jun 2026; the page names 4 of "6 authors"). Five-domain benchmark; two regimes (positive-controlled: gold item guaranteed in candidates; retrieval-realistic: not injected); 200-item pool; calibrated Qwen3-8B/32B rerankers do not consistently beat collaborative/content baselines; in the realistic regime single retrievers place the target in the pool only 4.6-22.9% of the time because 32-91% of targets are cold items; proposed LHF hybrid fusion. Data https://doi.org/10.5281/zenodo.20991039 ; code https://doi.org/10.5281/zenodo.20993306. A search snippet says domains include Amazon Reviews 2023, MIND, MovieLens-20M and Yelp (unverified; the abstract page I read does not name them). [FT/AbsV: abstract page] — https://arxiv.org/abs/2606.29947
  - Kusano et al. (2512.13001): text-embedding models beat LLM rerankers, BM25 is a non-trivial baseline (R@10 0.315 narrow cold-start). [FT/AbsV] — https://arxiv.org/abs/2512.13001
- **Fair/ceteris paribus comparison**: Zhang et al. (2603.29845) note that gains in prior generative-rec papers are confounded because model scale, identifier design and training strategy change together; they also say cold-start is rarely a main evaluation setting. [FT/AbsV]
- **Survey-level recommendation**: "A Focused Review of Multimodal CTR Prediction for Cold-Start Recommendation" (Sharm Fernando, Nethmi Wijesinghe; ICCRAIDS 2026, 10 Apr 2026) says evaluation is a key source of inconsistency; clarifies "cold" (zero-shot vs warm-up); recommends time-based and stratified protocols and minimum reproducibility requirements. [Snippet: abstract]
- **Sampled-metric evidence in 2025**: Pereira, Said, Santos (Aug 2025) study when sampling distorts offline evaluation using a fully observed dataset as ground truth. [Snippet only; I did not retrieve the title/ID; treat as unverified lead.]
- **Sampled vs full in the cold-start papers I opened**: Kusano et al. use 50 candidates (3 positives + 47 random negatives); Gusak et al. use unsampled user metrics but 1,000 sampled items for item analysis; Zhang et al. and Meehan & Pauwels use the full catalogue/warm+cold item pool as far as stated. This heterogeneity itself is a comparability hazard.
- **Fairness audit benchmark in zero-context setting**: **Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting** — Alexandre Andre, Gauthier Roy, Eva Dyer, Kai Wang. EARL 2025 workshop at RecSys 2025; arXiv 2508.20401 (28 Aug 2025). Modular zero-context pipeline (inputs only age/gender/language) over music, movies, colleges; Gemma 3 and Llama 3.2 show gendered/cultural stereotypes; non-linear size/fairness relation. No code link on page. [FT/AbsV] — https://arxiv.org/abs/2508.20401

### Inferences
- For a project using an LLM on MovieLens: (i) disclose possible ML-1M memorisation (GPT-4o evidence is third-party summary); consider evaluating also on a post-training-cutoff or non-public item subset, or hide titles/IDs; (ii) always include popularity, BM25/text-embedding and content-KNN baselines because several 2025-26 papers show they are hard to beat in cold-start; (iii) report both full-ranking and, if sampled, say so explicitly with the number of negatives.
- Because ML titles/plots are public text, text-based cold-item leakage is plausible; none of the opened papers tested this directly for item descriptions except via memorisation probing.

### Gaps
- No 2025-26 paper located that quantitatively revisits sampled-metric bias for cold-start specifically.
- Full text of Pereira et al. (2025) and of 2604.16318's protocol not read.

---

## Q3. Are there new 2025-2026 benchmarks or datasets for LLM-era cold-start recommendation?

### Takeaway
Yes, but fragmented and mostly narrow: a cold-start-specific reproducibility suite for generative recsys (2603.29845), a two-regime retrieval/rerank benchmark (2606.29947), a zero-context fairness benchmark (2508.20401), a job-recommendation dataset with a cold-start evaluation framework (JTH), and a social-media entity-preference dataset (2603.10148). I did not find a single widely adopted LLM-era cold-start benchmark; the survey itself calls for one.

### Cited Findings
- **Cold-Starts in Generative Recommendation: A Reproducibility Study** (2603.29845): unified warm / user-cold / item-cold splits over Amazon-Toys, MicroLens, Steam. See Q1 item 1. [FT/AbsV]
- **Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation** (2606.29947): five-domain benchmark separating reranking quality from retrieval coverage; releases protocol, splits, prompts, evaluation tooling and archived artifacts on Zenodo. [FT/AbsV]
- **JTH: A Dataset for Evaluating Cold-Start and Temporal Dynamics in Job Recommendation** — Julien Romero (Telecom ParisTech), Yann Millet, Eric Behar. ACM 2026-09-27, DOI 10.1145/3773078.3831846. Seven-year corpus by recruiters: LLM-enriched profiles for 38k candidates and 6k vacancies, day-level traces for 42k application trajectories; k-anonymity de-identification; releases a time-respecting cold-start evaluation framework and baselines. [Snippet: SciSpace abstract] (not movies, but a new cold-start dataset with a time-respecting protocol).
- **tau-Rec: A Verifiable Benchmark for Agentic Recommender Systems** — Bharath Sivaram Narasimhan, Karthik R. Narasimhan. arXiv 2606.10156 (v3 28 Jul 2026), accepted at ACM RecSys 2026 Reproducibility and Resource Track; code https://github.com/nbharaths/tau-rec. Multi-turn conversational agent benchmark with verifiable rewards and a pass^k reliability metric; best model about 57% pass^1 and 35% pass^4. A search snippet says its catalogue is built from TMDB /discover/movie with 2025-2026 releases (not confirmed on the abstract page; PDF text not readable by my tool). Not cold-start in the classic sense but relevant to TMDB-derived, post-cutoff catalogues. [FT/AbsV for abstract; TMDB detail Snippet]
- **Zero-context fairness benchmark** (2508.20401): see Q2. [FT/AbsV]
- **User profile datasets**: **User Profile with Large Language Models: Construction, Updating, and Benchmarking** — Nusrat Jahan Prottasha et al., arXiv 2502.10660 (15 Feb 2025); two open-source profile datasets (construction/updating); Mistral-7b and Llama2-7b strong; datasets not named on the page. [FT/AbsV: abstract]
- **Social knowledge dataset** (2603.10148): 14 categories, 280 entities, ~12-14K Twitter users (see Q4). [FT/AbsV]
- **Tooling**: ColdRec toolbox (undated) and the survey roadmap (2501.01945) which, per a search snippet, sketches a small benchmark (item-text dataset + held-out cold entities, Recall/NDCG@K, cold coverage, inference cost) but that is a proposal, not a release. [Snippet]
- **Awesome-Cold-Start-Recommendation** list (checked, not a benchmark list): 2025 items include DisCo (AAAI'25, 2412.15005), LLM Simulator for Cold-Start (WSDM'25), MI4Rec (CIKM'25), Meta-Adaptive Network (CIKM'25), Harnessing Light/epistemic uncertainty (CIKM'25), Item-centric Exploration (RecSys'25, 2507.09423), Cold-Start Rec with Knowledge-Guided RAG (2505.20773), SaviorRec (2508.01375), MoToRec and M2VAE (AAAI'26); none is flagged as a benchmark/dataset on the page. [FT: GitHub README] — https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation (most of those entries I did not individually open)
- **Related surveys 2026**: hybrid-AI news-rec cold-start survey (Mmaduakonam et al., 29 Jul 2026) says MIND is the most frequent dataset with Precision/Recall/AUC/nDCG/MRR [Snippet]; useful only if the project broadens beyond movies.

### Inferences
- For "LLM-generated item descriptions / TMDB metadata": 2025-26 papers I verified use item text already in the dataset (Amazon titles, ML genres/overviews per Ebrat & Rueda [Snippet]); I did not find a benchmark that standardises TMDB/IMDb enrichment. A TMDB-enriched ML split would therefore be the project's own contribution and must document the join (tmdbId in `links.csv`), fields and the time of fetching.
- LLM memorisation risk (Q2) means TMDB plots for classic films are likely pre-training text; prefer post-cutoff movies or masked titles if the project evaluates zero-shot LLM item knowledge.

### Gaps
- No verified 2025-26 benchmark with MovieLens-32M or Amazon-2023 cold-start splits released as a standard suite.
- Detailed dataset content of 2606.29947 (which five domains) not confirmed from primary text.

---

## Q4. What works for social-media-profile-based new-user cold start in 2025-2026?

### Takeaway
Evidence is thin. The strongest verified item is a March 2026 paper that embeds users from the accounts they follow on Twitter/X (no posts needed), which beats a follower-count popularity baseline by about 12-22% MAP in cross-domain zero-shot ranking; the rest are small applied studies or tangential profile-generation work. I found no 2025-26 paper that does classic new-user cold start on MovieLens/Amazon using Reddit or Twitter text with a standard top-K protocol.

### Cited Findings
1. **Social Knowledge for Cross-Domain User Preference Modeling** — Nir Lotan, Adir Solomon, Ido Guy, Einat Minkov. arXiv 2603.10148 (cs.SI, cross-list cs.AI), submitted 10 Mar 2026. [FT/AbsV: abstract + HTML] — https://arxiv.org/abs/2603.10148
   - Method: users and about 200K popular accounts in a joint space (SocialVec, a skip-gram/Word2Vec-style embedding learned from a sample of Twitter follow networks); a user = average of embeddings of the entities they follow, so new users are projected inductively without retraining; candidate entities in a target domain are ranked by cosine similarity (zero-shot, no target-domain feedback).
   - Data: 14 target categories (musical artists, news outlets, politicians, TV shows, car makers, ...) x 20 candidates = 280 entities; random U.S. Twitter users who follow them, 50 per entity; the paper text gives about 1K users per category and "14K overall" (abstract says roughly 12K: internal inconsistency). Data "may be shared for research"; no URL on the page I read.
   - Protocol: link prediction; candidate entities removed from the user's profile; rank candidates by cosine; MAP; baseline = popularity (follower count).
   - Results: MAP 0.468 (popularity) vs 0.571 (social embeddings, +22%) with full profile; 0.522 (+12%) in the closed-world "Sampled" setting; gains in 13 of 14 categories (largest films +81%, TV shows +74%; food chains +1%, car makers -7%). Performance converges quickly: 10 followed entities give 93.1% of full-profile MAP, 50 give 98.4%. GPT-4o prompting with the user's followed entities: about +13% over its non-personalised baseline at k=12 and about +23% at k=50.
   - Demographics: socio-demographic classifiers (gender, age, ethnicity, education, political affiliation) applied to follower embeddings show correlation with preferences; authors note risk of reinforcing bias.
   - Limitations: U.S.-centric Twitter sample, follow links as proxy for preference, single LLM.
   - Reuse: a model for "N entities from a social profile -> cold-start ranking" with N=10/20/50 analogous to the project's N=0/5/20/50 protocol; popularity baseline included. Needs a Twitter-like follow graph that MovieLens does not have, so reuse the design rather than the data.

2. **A Novel Framework Leveraging Social Media Insights to Address the Cold-Start Problem in Recommendation Systems** — Enes Celik, Sevinc Ilhan Omurca. J. Theor. Appl. Electron. Commer. Res. 20(3):234, 2 Sep 2025, DOI 10.3390/jtaer20030234. [Snippet: SciSpace abstract; the MDPI page returned HTTP 403 so full text unverified] — Builds behavioural profiles from users' X activity, then six boosting classifiers predict preferences for the top-20 most-rated films on Letterboxd; CatBoost best with F1 0.87 and MAE 0.21; claims to outperform existing methods. Dataset size/availability and baselines not verified. Note: classification of 20 films is far from a full-catalogue top-K protocol.

3. **Connecting Social Media to E-Commerce for Cold Start Product Recommendation using Microblogging Information** — Mekala Vamshi Krishna et al., IJSREM, 27 Dec 2025, DOI 10.55041/ijsrem55231. [Snippet: abstract] — Maps hashtags/sentiment/topics from microblog posts to product attributes for new items; abstract gives no dataset or numbers; low-rigor venue.

4. **From Millions of Tweets to Actionable Insights: Leveraging LLMs for User Profiling** — Vahid Rahimzadeh, Ali Hamzehpour, Azadeh Shakery, Masoud Asadpour. arXiv 2505.06184 (9 May 2025), accepted at MisD @ AAAI ICWSM 2025. [FT/AbsV: abstract] — LLM profiles from tweets (abstractive and extractive), Persian political Twitter dataset contributed; reports +9.8% over SOTA on its own profiling evaluation. Not a recommender; useful for how to summarise a social-media history into a text profile.

5. **Probing the LLMs for Generating User Profile of Session-based Social Recommendation (UPRec)** — Meng-Ying Lu, Hai-Tao Zheng, et al.; ACM 2026-09-27, DOI 10.1145/3773078.3831732. [Snippet: abstract] — LLM user profiles + similarity-based social embedding for session-based social recommendation, motivated by cold-start; benchmark datasets not named.

6. **Demographic/profile text as cold-start input**: Kusano et al. (2512.13001, ML-1M age/gender/occupation converted to text; narrow cold-start m=0) shows embedding models on profile text reach R@10 0.392 vs BM25 0.315, and LLM rerankers lower (see Q1). [FT/AbsV] Zero-context fairness benchmark 2508.20401 uses age/gender/language as the only signal and finds stereotyped outputs. [FT/AbsV]

7. **Tangential LLM-profile recommenders (not cold-start verified)**: ProMax (SIGIR 2026; Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin; arXiv 2604.26231, 29 Apr 2026) and ProEx (KDD 2026; arXiv 2512.00679, 30 Nov 2025) use LLM-derived user/item profiles; their abstract pages do not mention cold start or name datasets. [FT/AbsV: abstracts] Keyword-driven RAG for cold-start users (KALM4REC, arXiv 2405.19612, WWW 2025 per the Awesome list) is the 2025-venue version of a 2024 preprint, so it is listed under older protocols below.

### Inferences
- The only verified positive result with a popularity-baseline comparison is the follow-graph embedding (2603.10148): social-graph signal helps most in entertainment domains (films, TV, music) and less in categories with weak social identity.
- For a thesis project without Twitter data, the transferable ideas are: (a) project the cold user into a pre-trained space using a handful of entities; (b) test sensitivity to the number of entities (10 vs 50); (c) include a popularity baseline; (d) audit demographic bias.

### Gaps
- No verified 2025-26 paper using Reddit text or user bios for new-user cold start with a standard protocol (search tools returned none).
- Celik & Omurca full text blocked (403); dataset sizes, number of users, train/test protocol unverified.
- 2603.10148 code/data link not found on the page.

---

## Older protocols mentioned (not 2025-26; one line each)
- ColdLLM / LLM-InS, Huang et al., arXiv 2402.09176 (Feb 2024; WSDM 2025 version "Large Language Model Simulator for Cold-Start Recommendation" appears in the Awesome list): LLM-simulated interactions for cold items; likely the source protocol for FilterLLM's cold-item setup, but I did not open FilterLLM's experimental section to confirm.
- KALM4REC, Keyword-driven RAG for cold-start users, arXiv 2405.19612 (listed as WWW 2025 in the Awesome list): compact keyword profiles from new-user reviews.
- Krichene & Rendle (KDD 2020) and Aixin Sun (arXiv 2210.04149, 2022): foundational critiques of sampled metrics and of popularity-baseline/timeline leakage, repeated in search snippets as background.
- Wang et al., "Large Language Models as Data Augmenters for Cold-Start Item Recommendation", arXiv 2402.11724 (2024), and "Using LLMs as user interests interpretation..." (ISCIT 2024, Amazon 2018 Sports and Outdoors): earlier LLM augmentation protocols that 2025 work (2511.21989) extends.

---

## Tool notes
- **SciSpace search-papers** (natural-language questions): best for discovering 2025-26 candidates and getting abstracts, dates, DOIs, venues and some 2026 journal/conference items that the web search missed (JTH, LLM-CF, LaReS, LLM-FSGNN, Celik & Omurca, UPRec). Limits: abstracts only, results mix pre-2025 papers, author lists sometimes truncated, venue/date metadata sometimes ambiguous (e.g. arXiv-listed items flagged as 2026 with ID prefixes that do not match).
- **WebSearch**: useful for arXiv IDs and for the Awesome list/ColdRec pointers; search "answers" are model-written summaries, so every number from them is marked Snippet unless re-fetched.
- **WebFetch on arxiv.org/abs and arxiv.org/html**: reliable for abstract pages and for HTML full text (gave the protocol detail for 2603.29845, 2510.11402, 2508.00452, 2508.07856, 2512.13001, 2603.10148). Limits: arxiv.org PDFs came back as binary (FilterLLM 2502.16924 and tau-Rec 2606.10156 could not be read); MDPI returned HTTP 403; the fetcher summarises with a small model and often declines to restate full abstracts, so some "abstract" text above is paraphrase; "code link" statements reflect only what the page contained.
- Call budget exceeded the suggested ~15 because the topic is wide; still not covered: ACM DL proceedings pages directly (RecSys 2025/2026, SIGIR 2026, WWW 2026, KDD 2026), OpenReview, and the ColdRec data README.
