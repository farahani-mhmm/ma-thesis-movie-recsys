# LLM-based cold-start recommendation methods, 2025-2026 (catalogue for a MovieLens + TMDB practice project)

Scope: papers first published 2025-01-01 .. 2026-10-30. Verification labels used below:
- **[FULL]** = I fetched the arXiv abstract page and/or the full HTML text and read the numbers myself.
- **[ABS]** = abstract page only (arXiv abs page fetched; no experimental details available to me).
- **[SNIPPET]** = only the SciSpace search-papers abstract/metadata (or a WebSearch snippet); I did not open the paper. Numbers are copied from the abstract, not checked.
Nothing below was invented; where a field was not visible to me it says "not verified / not stated".

---

## Key question 1: Which 2025-2026 LLM cold-start methods are strongest and most reproducible on MovieLens, and what gains over popularity / text-embedding baselines are reported?

### Takeaway
The most useful and honest MovieLens-relevant 2025-2026 evidence is (a) a controlled study showing that open text-embedding models beat LLM rerankers for training-free cold-start on ML-1M (arXiv 2512.13001), (b) a June 2026 diagnostic benchmark on ML-20M and four other domains with released code showing LLM rerankers do not reliably beat CF/content baselines and that retrieval coverage is the bottleneck (arXiv 2606.29947), and (c) LaReS (Neural Networks 2026), a frozen-LLM item-cold-start method with public code and MovieLens data. Most "SOTA" claims on MovieLens come from papers I could only verify at abstract level; no paper I verified reports a clean "gain over popularity" number on MovieLens.

### Ranked catalogue (usefulness for a one-GPU MovieLens + TMDB-text practice project)

Ranking criteria: MovieLens used, open-weight / small models, code or released protocol, cold-start protocol clear, cheap to reproduce.

---

#### Rank 1. Are Large Language Models Really Effective for Training-Free Cold-Start Recommendation? [FULL]
- Authors: Genki Kusano, Kenya Abe, Kunihiro Takeoka (NEC). arXiv 2512.13001 (cs.IR), v1 **15 Dec 2025**. https://arxiv.org/abs/2512.13001 (full text: https://arxiv.org/html/2512.13001v1)
- Cold-start type: **new user** (training-free; m = 0 interactions = "narrow CS" with only a profile, m >= 1 = "broad CS").
- LLM role: LLM **reranker** (prompting) compared head-to-head with **text-embedding models (TEMs)**; also discusses TEMs trained with LLM supervision/synthetic data.
- Datasets: MovieLens-1M (age/gender/occupation turned into text profiles), a Job recommendation dataset (with/without work experience), Amazon Reviews (Music, Movie, Book).
- Protocol: 500 users per dataset with >= m+3 interactions; 3 most recent items held out as positives; candidate set of 50 = 3 positives + 47 random unseen items from the same domain; main tables m = 1, extra analysis m in {0,1,3,5,10}.
- Models: BM25; TEMs multilingual-e5-large, bge-m3, gte-modernbert-base, gte-Qwen2-1.5B-instruct, gte-Qwen2-7B-instruct, Qwen3-Embedding-0.6B, Qwen3-Embedding-8B; LLM rerankers gpt-4.1-mini, gpt-4.1, Qwen3-8B.
- Metrics: Recall@10, nDCG@10; one-sided Wilcoxon signed-rank test.
- Headline numbers (all datasets pooled, as read from the HTML): narrow CS: best TEM (Qwen3-Embedding-8B) Recall 0.392 / nDCG 0.430 vs best LLM (gpt-4.1) 0.311 / 0.283 vs BM25 0.315 / 0.362. Broad CS m=1: best TEM (gte-Qwen2-7B-instruct) 0.536 / 0.581 vs gpt-4.1 0.433 / 0.377. (The extraction noted one internal inconsistency in the BM25 broad-CS row; check the paper table before quoting BM25.)
- Baselines vs "popularity": not included (BM25 is the sparse baseline). The paper's own limitation text says narrow-CS on MovieLens is especially hard with low absolute scores.
- Code / GPU: no code link found on the page; hardware not stated. All TEMs are open weights; 0.6B-8B embedding models fit on one modest GPU (inference only).
- Limitations (from paper): three datasets only; synthetic data for TEM training may be biased/hallucinated; only natural-language supervision; long item text hurts LLM reranking.
- Why useful: gives a cheap, strong, fair baseline recipe (TEM retrieval) and a clean 1-user-cold protocol that can be re-implemented on ML-1M + TMDB overviews in an afternoon.

#### Rank 2. Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation [FULL]
- Authors listed on the arXiv page: Zhe Dong, Fang Qin, Manish Shah, Yicheng Wang (the page says "5 other authors" but names four). arXiv 2606.29947, v1 **29 Jun 2026**. https://arxiv.org/abs/2606.29947 (HTML: https://arxiv.org/html/2606.29947v1)
- Cold-start type: **mostly new item** (item-new = zero training interactions makes up 32-91% of test targets) plus user-cold and long-tail buckets, i.e. a joint view.
- LLM role: LLM **reranker** (Qwen3-8B primary; Qwen3-32B and a 4-bit Llama-3.3-70B-Instruct as scale checks), candidates scored by log-prob of a calibrated "Yes"; plus a proposed non-LLM **learned hybrid fusion (LHF)** retriever layer.
- Datasets: Amazon Reviews 2023 Arts and Video Games, MIND-small, **MovieLens-20M**, Yelp Philadelphia. ML-20M test set 2,951 interactions: 1,894 item-new, 3 item-cold, 22 long-tail, 61 user-cold, 971 warm (so ML-20M is dominated by item-new targets).
- Protocol: chronological 70/10/20 split; bucket labels computed from the training split only; one bucket per test interaction by priority.
- Baselines: Popularity, item-kNN, TF-IDF, Markov, BPR, LightGCN, SASRec, graph co-occurrence, **SBERT/BGE dense retrieval (text-embedding baselines)**, RRF/round-robin/CARA fusions, cross-encoder.
- Metrics: Recall@10 (rerank), Coverage@200 (retrieval); bootstrap CIs, McNemar tests.
- Headline findings: Qwen3-8B wins no domain; Qwen3-32B beats 8B on four domains but still trails the best CF/fusion baseline on four of five; single retrievers put the gold item in the top-200 pool only 4.6-22.9% of the time; LHF reaches 6.1-24.3%; end-to-end LLM Recall@10 <= 1.4% in all domains; LightGBM over the LHF pool adds +1.3-2.7 Recall@10 points. Throughput: LightGCN ~55,455 pairs/s (4.2 GB) vs Qwen3-8B 58.7 pairs/s (36 GB). Per-dataset popularity numbers were not extracted by my fetch.
- Code: https://github.com/dongzhe1/rec-diag-bench/tree/0.0.0 (Zenodo DOI https://doi.org/10.5281/zenodo.20993306); data https://doi.org/10.5281/zenodo.20991039. Hardware: single H100-class GPU, Qwen3-8B in bf16.
- Limitations (paper): LLM reranking evaluated on sampled subsets; LHF validated offline only; prompts not exhaustively searched; LHF is a lightweight mitigation.
- Why useful: the only 2026 paper I verified with MovieLens + open-weight Qwen3 + released benchmark code; a very realistic reference for "negative result / careful protocol" chapters. Note: it is not a method that makes LLMs win; it argues they currently do not.

#### Rank 3. Large language model with retrieval semantics for cold-start recommendations (LaReS) [SNIPPET for paper; README FULL]
- Authors: Zelong Wu, Dongsheng Wang, Yue Wang, Rui Zhang, Lisi Chen, Peng Han. **Neural Networks, 2026** (SciSpace date 2026-07-20), DOI 10.1016/j.neunet.2026.109401. Could not open the publisher page; the title was not found by WebSearch.
- Cold-start type: **new item**.
- LLM role: **knowledge/feature enhancer** - self-retrieval of semantically similar warm items as context; a *frozen* LLM + trainable semantic layers produce dense item embeddings; joint preference learning (multi-objective, distribution alignment) distilling warm-item signals into cold items.
- Datasets: "MovieLens" (version not stated in abstract). The GitHub README log shows 6,040 users / 3,706 items, consistent with ML-1M (my inference, not stated). Cold-start splits come from the ColdRec benchmark data (download into ./data).
- Headline: "improves cold-start P@5 on MovieLens by 7.98%" (abstract; baseline not stated in snippet).
- Code: https://github.com/wuzelong/LaReS [README read]. Python 3.9, CUDA 12.8; training `python main.py`; offline LLM step (`prompt/movielens_prompt.py`, `prompt/movielens.py`) generates item documents/semantic context, precomputed MovieLens context and LightGCN/LaReS embeddings are in the repo; README recommends Ollama for local LLM; ~0.17 s/epoch on one GPU in the shown log; no license stated; LLM model name not stated in README.
- Baselines / metrics other than P@5: not verified.
- Why useful: the only verified 2025-26 repo that already runs on MovieLens with cached LLM outputs; the LLM is used offline and frozen, so it is cheap.

#### Rank 4. Enhancing Collaborative Filtering With Large Language Model Embeddings for User Cold-Start Recommendation (LLM-CF) [SNIPPET]
- Authors: Eranki Ramakrishna Nageswara Rao, Nellutla Saketh Koundinya, Sowmini Devi Veeramachaneni, Arun K. Pujari. IEEE PRML 2026 (SciSpace date 2026-07-01), DOI 10.1109/prml71097.2026.11687003.
- Cold-start type: **new user** (and discusses item cold-start as a limitation).
- LLM role: frozen **sentence-encoder embeddings** of user/item text projected by a small MLP and fused with CF embeddings via a warmth-aware gate.
- Datasets: MovieLens-1M and Amazon Office Products. Baseline: BPR-MF. Headline (abstract): user cold-start NDCG@10 improved 3.4x (ML-1M) and 2.0x (Amazon) over BPR-MF while matching warm performance; specialised cold-start methods can beat it for item cold-start in extreme sparsity.
- Code/GPU/protocol: not verified. Very cheap to replicate (frozen small encoder + MLP + gate). Caveat: baseline is plain BPR-MF (weak for cold users), so "3.4x" is not evidence of beating strong content baselines.

#### Rank 5. FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation [FULL, arXiv HTML]
- Authors: Ruochen Liu and 8 co-authors (Yuanchen Bei submitter; full list not captured). arXiv 2502.16924 (cs.IR), v1 **24 Feb 2025**. https://arxiv.org/abs/2502.16924 (HTML: https://arxiv.org/html/2502.16924v1)
- Cold-start type: **new item**.
- LLM role: **LLM fine-tuning (LoRA)** as an **interaction simulator**: instead of judging one (user,item) pair per call ("text-to-judgment"), the LLM outputs a distribution over the whole user vocabulary in one pass ("text-to-distribution"); the simulated interactions train the CF model.
- Datasets: CiteULike (5,551 users / 16,980 items), **ML-10M** (69,797 users, 10,258 items, 3.66M interactions after rating >= 4; titles + genres as item content); plus an Alibaba online A/B test.
- Protocol: 20% of items cold (their interactions split 1:1 into cold val/test); remaining 80% warm split 8:1:1; strict item cold-start.
- LLM: LLaMA2-7B (main); LLaMA3-1B/3B, LLaMA2-7B/13B studied; only LoRA layers + user-vocabulary embeddings tuned.
- Baselines: DropoutNet, MTPR, Heater, CLCRec, DeepMusic, MetaEmb, GAR, ALDI, USIM, MI-GNN, Wang et al. 2024b, ColdLLM; backbones MF/NGCF/LightGCN.
- Metrics: Recall@20 / NDCG@20 (overall, cold, warm).
- Headline: vs best baseline overall, +13.28% Recall / +20.08% NDCG on CiteULike but only **+2.13% Recall / +2.82% NDCG on ML-10M**; >30x (38.84x CiteULike, 36.53x ML-10M) faster inference than ColdLLM; Alibaba online: vs ColdLLM +5.13% cold PV, +3.93% PCTR, +10.86% GMV.
- Compute: training on 4x A100 80GB; inference timing on one A100. No code link found in text.
- Limitations: modest ML-10M gain; two datasets; no code.
- Why useful: a concrete LoRA-LLM-as-simulator recipe; shrink to LLaMA3-1B/3B (the paper's own generalisation study) for one GPU. Popularity baseline not reported.

#### Rank 6. Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation [FULL]
- Authors: Nachiket Subbaraman, Jaskinder Sarai, Zhe Zhao (UC Davis); Aniruddh Nath, Lukasz Heldt, Li Wei (Google); Lichan Hong (Google DeepMind). arXiv 2511.21989, v1 **27 Nov 2025**. https://arxiv.org/abs/2511.21989
- Cold-start type: **new item**. LLM role: **LLM as simulated user** (pairwise cold-item preferences) for data augmentation, with an **RL policy** (REINFORCE-style contextual bandit, 5-parameter linear policy) choosing which warm users to simulate.
- Datasets: Amazon Beauty (22,363 users / 12,094 items) and Sports (35,597 / 18,267), 5-core. **No MovieLens.** Single-time-point 7:3 split; cold Recall@K counts only test examples whose positive is cold; augmentation for 20% of warm users.
- LLM: LLaMA-3.2-3B-Instruct. Backbone: two-tower, in-batch softmax + LogQ.
- Headline (cold Recall@50 %, None / Random / Learned): Beauty 1.05 / 2.18 / 2.65; Sports 1.15 / 2.60 / 3.17. Learned vs random +21.6% / +21.9%.
- Compute: 4x RTX 6000 Ada; ~24 GPU hours for the true-reward policy. No code link. Trade-off: warm recall decreases.
- Why useful: shows a small (3B) open LLM suffices as a simulator; "which users to simulate" is a cheap, publishable twist.

#### Rank 7. LLM Reasoning for Cold-Start Item Recommendation (Netflix) [FULL]
- Authors: Shijun Li, Yu Wang, Jin Wang, Ying Li, Joydeep Ghosh, Anne Cocos. arXiv 2511.18261, v1 **23 Nov 2025**; SciSpace also lists a DOI 10.1145/3774904.3792872 (WWW '26 per a WebSearch snippet; the title was found with ACM DOI only via SciSpace, I did not open ACM). https://arxiv.org/abs/2511.18261 (HTML: https://arxiv.org/html/2511.18261v1)
- Cold-start type: **new item**. LLM role: **LLM fine-tuning (QLoRA SFT, GRPO RL, SFT+GRPO)** for reasoning-based **re-ranking**.
- Base model: Qwen-2.5-32B-Instruct (SFT with 4-bit QLoRA; GRPO per DeepSeekMath). Reasoning styles "Base-Reason", "Soft Self-Consistency", baselines Direct-Rec / Fast-Reason.
- Dataset: **proprietary Netflix logs; MovieLens NOT used.** Cold items = launched after the users' latest history and after the base LLM's training cutoff. Task: re-rank 40 production-ranker titles + 10 cold titles (50 candidates). SFT set 16,218 interactions (7,252 reasoning paths); GRPO prompts 2,834 / 3,484.
- Metric: Recall@1 (AnyPlay, Discovery); only **relative %** reported. Cold-start, relative to non-fine-tuned variant: Base-Reason+GRPO Discovery +18.42%, AnyPlay +5.10%; Soft-SC+SFT AnyPlay +22.45%, Discovery +0.21%; Base-Reason+SFT+GRPO +7.14% / +7.89%.
- The abstract's "outperform Netflix's production model by up to 8%" refers to the **warm-start** comparison (Table 3), per my reading of the HTML; the paper itself says SFT+GRPO "struggles with cold-start" (flagged because the abstract wording can mislead).
- Hardware, code: not reported / none. Limitations: proprietary data, single base model, small training sets, only relative numbers, internal inconsistencies between text and tables.
- Why useful: the only 2025 paper with a SFT vs RL (GRPO) vs hybrid comparison for cold-start; the recipe is transferable to MovieLens + TMDB text but a 32B model is beyond one modest GPU (use 3B-8B + QLoRA instead, my inference).

#### Rank 8. Adaptive Candidate Retrieval with Dynamic Knowledge Graph Construction for Cold-Start Recommendation (ColdRAG) [FULL]
- Authors: Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee, Philip S. Yu. arXiv 2505.20773, v1 **27 May 2025** (v3 24 Nov 2025). Preprint, no venue found. https://arxiv.org/abs/2505.20773
- Cold-start type: **new item**; zero-shot, training-free. LLM role: **LLM as recommender via prompting + RAG**: builds a KG from item metadata, LLM-guided multi-hop retrieval, then ranking.
- Datasets: Amazon Games (2,096 users / 2,027 items), Toys, Office (core-filtered). **No MovieLens.** Cold items = least frequent 10% of items; leave-one-out, 500 user sequences per domain ending in a cold item.
- LLMs: gpt-4o-mini (Azure) and Qwen2.5-32B-Instruct via vLLM; embeddings text-embedding-3-small and bge-m3 + FAISS.
- Baselines: UniSRec, CLCRec, TDRO; training-free: plain LLM, LLMRank, TaxRec, KALM4Rec. Metrics Recall@10, NDCG@10 (5 runs).
- Headline: ColdRAG (Qwen) Recall@10 19.57 (Games), 4.10 (Toys), 9.40 (Office); reported gains over strongest baseline 78.22% / 26.76% / 123.81% (the text quotes +78.6% for Games, mismatching the table).
- Code: https://github.com/WooseongYang/ColdRAG. GPU not stated. Limitations: repeated LLM calls = cost/latency; reliance on closed GPT; static hyperparameters.
- Note: the arXiv title differs from the name ColdRAG (a WebSearch snippet and the Awesome-Cold-Start list refer to "Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation", probably an earlier title; not verified).

#### Rank 9. LLM-FSGNN: LLM-Guided Feature-Structure Augmentation Graph Neural Network for Cold-Start Recommendation [SNIPPET]
- Authors: Di Jin, Zhizhi Yu, Songyuan Lei, Dongxiao He (Tianjin University). ACM journal, SciSpace date **2026-06-04**, DOI 10.1145/3811919 (journal name not shown in the record).
- Cold-start type: **both (new user and new item)**. LLM role: **knowledge/feature enhancer** (LLM-generated textual descriptions for users/items, LLM-predicted user-item intentions to add edges) plus a graph-structure enhancement module to curb LLM hallucination.
- Datasets: MovieLens-100K, MovieLens-1M, Amazon. Metric **MAE** (rating prediction, not ranking). Headline (abstract): 4.8% lower MAE on ML-100K, 3.6% on ML-1M, 4.2% on Amazon vs current SOTA. Protocol, LLM, code, GPU: not verified.
- Fits the meta-learning / graph cold-start line (rating-prediction cold-start tasks); compare to baselines of that line, not ranking ones.

#### Rank 10. Further MovieLens-touching items, abstract/snippet level only
- **Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM** [ABS]. Yushang Zhao, Xinyue Han, Qian Leng, Qianyi Sun, Haotian Lyu, Chengrui Zhou. arXiv 2509.13179, v1 **16 Sep 2025**; also IEEE (10.1109/aiac68175.2025.11332482, SciSpace date 2025-10-15). New users and items; LLM as **embedding model** (BPE subword-level vectors aligned to the BPE vocabulary, aggregated into a cold-start vector). Abstract claims higher Recall@k, NDCG@k, Hit Rate than "the standard baseline" at comparable cost. **The arXiv abstract page names no dataset, no numbers, no code link**; third-party summaries (not verified against the PDF) mention DistilBERT/RoBERTa encoders and GraphSAGE/LightGCN baselines. Treat as low-verifiability; do not rely on for MovieLens.
- **Selecting / simulation family from the Alibaba/Tsinghua group**: *SetLLM: Set Large Language Model for Cold-Start Item Recommendation* (Ruochen Liu, Hao Chen, Yuan-Qi Bei, ... Senzhang Wang; ACM DOI 10.1145/3770855.3817707, SciSpace date 2026-08-08) [SNIPPET]; *Turning Cold Entities Warm: LLM-Based User and Item Interaction Simulation for Recommendations* (ProbLLM; Hao Chen, ..., Feiran Huang; ACM DOI 10.1145/3848638, 2026-09-30; new user and new item, "warm-up" and strict cold entities) [SNIPPET]; *Synergizing Large Language Model and Knowledge Graph for Cold-Start Recommendation* (SCSRec; Haobo Zhang, Qiannan Zhu, Zhicheng Dou; ACM TOIS, DOI 10.1145/3830081, 2026-07-15) [SNIPPET]. All three simulate cold-item interactions with LLM + (KG or set-generation); datasets are not named in the abstracts ("two" / "three public datasets"). SetLLM's abstract claims "over 30 times" efficiency vs PairLLM, closely mirroring the FilterLLM claim (>30x vs ColdLLM) from the same group - I could not tell whether SetLLM is an extension of FilterLLM; flag for the writer.
- **Zero-shot / embedding-style small studies** [SNIPPET]: *Semantic Embedding-Based Zero-Shot and Few-Shot Learning for Cold-Start Movie Recommendation* (Bandara et al., IEEE ICATC, 2025-12-16; Sentence-BERT on MovieLens titles/genres/plots; "significantly outperforming baseline models", no numbers); *Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)* (N. Zmanovskii, arXiv 2603.03309, v1 8 Feb 2026) [ABS: single author; hybrid LLM+KG+VARK framework claims validation on MovieLens-1M; no numbers on the abstract page; not a survey]; *Research on Lightweight LLM Recommendation Algorithm in Few-Shot Cold-Start Scenarios* (LLM-RecLite; Zhenyu Ni; SciSpace date 2026-04-08; QLoRA + hierarchical prompt + distillation; ML-1M and Amazon Beauty; abstract claims NDCG@10 +18.3% / +9.7% and 4.2x faster inference; single author, venue not identified, abstract-level only, treat with caution); *Hybrid User-LLM* (Saran, Choudhary, Meena; ITM Web of Conferences, 2026-06-05; ML-20M 50k users; claims Hit@10 10.1% overall, cold users 11.8% vs warm 11.2%; abstract-level only).
- **GAT + LLM profile papers on MovieLens** [ABS for 2508.01514]: *End-to-End Personalization: Unifying Recommender Systems with Large Language Models* (Danial Ebrat, Tina Aminian, Sepideh Ahmadian, Luis Rueda; arXiv 2508.01514, v1 **2 Aug 2025**; ML-100K/1M, LLM-generated user/item profiles as GAT inputs, LLM re-ranker + explanations). The abstract page does not report cold-start experiments, so it is "sparse feedback" work, not strict cold-start. A related earlier version, *Vectorized Context-Aware Embeddings for GAT-Based Collaborative Filtering* (arXiv 2510.26461, SciSpace 2025-10-31; also a 2025-05-19 record) [SNIPPET], claims robustness for users with limited interaction history.
- **LLM-judged signals for cold users**: *Improving Cold-User Recommendation via LLM-Judged Signals* (Dae Seob Shin, Taehyun Ha; IEEE IMCOM 2026-01-04) [SNIPPET]; Amazon All Beauty/Appliances only; LLM pairwise user preference signals; Recall/Hit Rate gains.
- **M-LLM^3REC** (Lining Chen, Qingwen Zeng, Huaming Chen; arXiv 2508.15262, v1 **21 Aug 2025**; ACM CIKM 2025 DOI 10.1145/3746252.3761214 via SciSpace) [ABS]: motivation-aware profile extraction from limited interactions; abstract page lists no datasets/LLM/numbers.
- **Prompt-as-Policy over Knowledge Graphs for Cold-start Next POI Recommendation** (arXiv 2510.08012) [SNIPPET via WebSearch only]: inactive-user POI, contextual bandit over prompts of a frozen LLM, avg +7.7% relative Acc@1 for inactive users; not movies.
- **Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users** (Haowei Yang et al.; IEEE ISPDS 2025-10-31) [SNIPPET]: few-shot prompts on BioGPT, LLaMA-2, GPT-4; dataset not named in abstract.
- **Generative / semantic-ID recommenders that report cold-start results** [all SNIPPET]: *SETRec++* (Xinyu Lin, ..., Tat-Seng Chua; ACM Trans. Recommender Systems, 2026-08-10; instantiated on T5 and Qwen 1.5B-7B; cold-start ranking results on four datasets); *VaLiDRec: Variable-Length LLM-Aligned Semantic IDs* (Shutong Qiao et al., arXiv 2607.25209, 2026-07-28; claims superior zero-shot item cold-start, 87.49x faster than LC-Rec); *GenRecEdit* (Chenglei Shen et al., arXiv 2603.14259, 2026-03-15; model editing to fix "cold-start collapse" of generative recommenders with ~9.5% of retraining time); *Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval* (Ruohan Zhang, Jiacheng Li, Julian McAuley, Yupeng Hou; arXiv 2509.16446, 2025-09-23; reports improved cold-start); *AlignGenRec* (Jinshuo Xing et al., IEEE SMC 2025-10-05; "7-11%" gains in cold-start across three datasets); *GRLM / Term IDs* (Zhiyang Zhang et al., arXiv 2601.06798, 2026-01-11). A WebSearch hit also lists *IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu* (arXiv 2603.01590) - title only, not opened.
- Out of scope for movies: *A Novel Framework Leveraging LLMs to Enhance Cold-Start Advertising Systems* (Uruqi et al., Future Internet, 2025-08-08; DigiX/Avazu CTR) [SNIPPET].

### Cited Findings (key claims, each with a source)
- TEMs outperform LLM rerankers (gpt-4.1, Qwen3-8B) for training-free cold-start on ML-1M/Job/Amazon, in cold and warm settings — [arXiv 2512.13001](https://arxiv.org/abs/2512.13001), numbers from [HTML](https://arxiv.org/html/2512.13001v1)
- LLM rerankers (Qwen3-8B/32B) do not consistently beat strong CF/content baselines; in a retrieval-realistic setting, single retrievers cover the gold item in top-200 only 4.6-22.9% of the time; ML-20M is dominated by item-new targets (1,894 of 2,951) — [arXiv 2606.29947](https://arxiv.org/html/2606.29947v1)
- LaReS improves cold-start P@5 on MovieLens by 7.98%, uses a frozen LLM, code released — SciSpace record of [Neural Networks 2026, DOI 10.1016/j.neunet.2026.109401](https://doi.org/10.1016/j.neunet.2026.109401); repo [github.com/wuzelong/LaReS](https://github.com/wuzelong/LaReS)
- FilterLLM gains on ML-10M are modest (+2.13% Recall@20 overall vs best baseline) compared with CiteULike (+13.28%) — [arXiv 2502.16924 HTML](https://arxiv.org/html/2502.16924v1)
- Netflix reasoning paper uses proprietary data, Qwen-2.5-32B-Instruct, QLoRA-SFT + GRPO; hybrid SFT+GRPO "struggles" in cold-start — [arXiv 2511.18261 HTML](https://arxiv.org/html/2511.18261v1)
- LLM-FSGNN: 4.8% / 3.6% lower MAE on ML-100K / ML-1M vs SOTA (abstract) — SciSpace record of [ACM DOI 10.1145/3811919](https://doi.org/10.1145/3811919)
- LLM-CF: user cold-start NDCG@10 3.4x (ML-1M) over BPR-MF — SciSpace record of [IEEE DOI 10.1109/prml71097.2026.11687003](https://doi.org/10.1109/prml71097.2026.11687003)

### Inferences
- For a one-GPU MovieLens project, a defensible, reproducible experiment is: popularity vs BM25 vs open text-embedding retrieval (bge-m3 / Qwen3-Embedding-0.6B / gte-Qwen2) over TMDB overviews vs a small LLM reranker (Qwen3-8B, 4-bit) under the 2512.13001 / 2606.29947 protocols. Both recent studies suggest embeddings (and retrieval coverage) matter more than LLM reranking.
- LaReS (frozen LLM, cached offline outputs) is the cheapest "LLM as knowledge enhancer" method with public MovieLens code; FilterLLM/ColdLLM-style simulation and the GRPO recipe are the "heavier" families, to be run at 1B-8B scale.
- Gains are reported against many different baselines, so numbers across papers are not comparable. No verified paper gives a headline "vs popularity" gain on MovieLens.

### Gaps
- Exact datasets/baselines/code for LaReS, LLM-FSGNN, SetLLM, ProbLLM, SCSRec, LLM-CF, BPE-LLM, M-LLM3REC were not verifiable (publisher pages not opened; arXiv entries absent or abstract-only).
- No verified numbers versus popularity baselines on MovieLens in any 2025-2026 LLM cold-start paper (2606.29947 includes Popularity but my fetch did not return per-dataset values).
- Which exact MovieLens version LaReS uses is inferred only from the README log (6,040 users / 3,706 items).
- Whether SetLLM is a separate paper from FilterLLM (very similar abstract claims) was not resolved.

---

## Key question 2: What are the main method families, and which are feasible on one modest GPU?

### Takeaway
The 2025-2026 literature falls into (1) LLM/embedding as knowledge or feature enhancer (cheapest), (2) LLM as interaction simulator / data augmenter (offline LLM cost, tuned recommender), (3) LLM as direct recommender/reranker via prompting (training-free but weak in controlled tests), (4) LLM fine-tuning with LoRA / RL / reasoning (highest quality ceiling in industrial data, heaviest), and (5) LLM-generative recommenders with semantic IDs where cold-start is a reported problem.

### Cited Findings
- **Family A, LLM/embeddings as knowledge/feature enhancer**: LaReS (frozen LLM + trainable semantic layers, self-retrieval of warm neighbours) [SciSpace/Neural Networks](https://doi.org/10.1016/j.neunet.2026.109401); LLM-CF (frozen sentence encoder + MLP + warmth gate, ML-1M) [IEEE PRML 2026](https://doi.org/10.1109/prml71097.2026.11687003); LLM-FSGNN (LLM textual descriptions + GNN, ML-100K/1M) [ACM 3811919](https://doi.org/10.1145/3811919); BPE-LLM subword embeddings [arXiv 2509.13179](https://arxiv.org/abs/2509.13179). GPU need: low (inference of an encoder, or one offline pass of an LLM).
- **Family B, LLM interaction simulator / data augmenter**: FilterLLM (LoRA LLaMA2-7B text-to-distribution; 4x A100 training, 1 A100 inference) [arXiv 2502.16924](https://arxiv.org/html/2502.16924v1); user-selection RL for LLM users with LLaMA-3.2-3B (4x RTX 6000 Ada, ~24 GPU hours) [arXiv 2511.21989](https://arxiv.org/abs/2511.21989); SetLLM / ProbLLM / SCSRec (abstract level) [SetLLM](https://doi.org/10.1145/3770855.3817707), [ProbLLM](https://doi.org/10.1145/3848638), [SCSRec](https://doi.org/10.1145/3830081). The LLM runs offline; cost grows with number of (user,item) pairs unless the text-to-distribution/set design is used.
- **Family C, LLM as recommender/reranker via prompting**: ColdRAG (training-free, KG-RAG, Qwen2.5-32B or gpt-4o-mini; code released) [arXiv 2505.20773](https://arxiv.org/html/2505.20773v3); the controlled comparison finds TEMs beat LLM rerankers [arXiv 2512.13001](https://arxiv.org/abs/2512.13001); the diagnostic benchmark finds retrieval coverage, not LLM reranking, is the bottleneck [arXiv 2606.29947](https://arxiv.org/html/2606.29947v1). Qwen3-8B reranking ran at 58.7 pairs/s and 36 GB vs 55k pairs/s and 4.2 GB for LightGCN (same source).
- **Family D, fine-tuning / RL / reasoning**: SFT (QLoRA) vs GRPO vs SFT+GRPO on Qwen-2.5-32B for cold-item re-ranking (Netflix, proprietary) [arXiv 2511.18261](https://arxiv.org/html/2511.18261v1); RL used only for user selection (REINFORCE) in [2511.21989](https://arxiv.org/abs/2511.21989); LLM-RecLite claims QLoRA + distillation on lightweight LLMs for ML-1M (single-author, snippet) [DOI 10.71222/4rktf787](https://doi.org/10.71222/4rktf787). I found no 2025-26 paper with open GRPO code on MovieLens cold-start.
- **Family E, generative recommenders (semantic IDs / LLM backbones)**: reported cold-start collapse and fixes (GenRecEdit [arXiv 2603.14259](https://arxiv.org/abs/2603.14259) via SciSpace; VaLiDRec [arXiv 2607.25209](https://arxiv.org/pdf/2607.25209); SETRec++ on T5/Qwen 1.5B-7B [DOI 10.1145/3838805](https://doi.org/10.1145/3838805)); a WebSearch snippet reports that semantic-ID generative models perform poorly on cold-start items, source [Purely Semantic Indexing, arXiv 2509.16446](https://arxiv.org/pdf/2509.16446) (snippet-level).

### Inferences
- Feasible on one modest GPU (<= 24 GB): family A fully; family B at 1B-3B (FilterLLM's own study includes LLaMA3-1B/3B; 2511.21989 used a 3B model) with 4-bit/LoRA and a capped candidate set; family C only for small candidate lists (50 candidates as in 2512.13001); family D at <= 8B with QLoRA/GRPO via small batch (inference from the Netflix recipe, not demonstrated on a modest GPU in any verified paper); family E feasible with 1.5B models.
- Recommended practical plan: popularity + BM25 + TEM retrieval as strong baselines; LLM-enhanced features (A) as the main method; optional ablation with a small LLM reranker (C) or LoRA simulator (B).

### Gaps
- GPU requirements are rarely stated (LaReS, ColdRAG, 2512.13001: not stated). Only 2511.21989, 2502.16924, 2606.29947 give hardware.
- No 2025-26 paper verified that reports a joint new-user + new-item protocol at scale; LLM-FSGNN and ProbLLM claim both but are snippet-only.

---

## Key question 3: Which 2025-2026 surveys exist and what taxonomy / open problems do they list?

### Takeaway
One dedicated LLM-era cold-start survey (arXiv 2501.01945, Jan 2025) plus several general LLM-for-recsys surveys from 2026 that contain a cold-start component. I could read only abstracts, so the detailed taxonomy / open problems of 2501.01945 are not verified.

### Cited Findings
- **Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap** [ABS] — Weizhi Zhang, Yuanchen Bei, Liangwei Yang, Henry Peng Zou, Peilin Zhou, Aiwei Liu, Yinghui Li, Hao Chen, Jianling Wang, Yu Wang, Feiran Huang, Sheng Zhou, Jiajun Bu, Allen Lin, James Caverlee, Fakhri Karray, Irwin King, Philip S. Yu; arXiv 2501.01945, v1 3 Jan 2025, v2 16 Jan 2025; no venue listed. Organises the field by information source: content features, graph relations, domain information, and LLM world knowledge; gives a roadmap and a continuously updated list https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation — [arXiv](https://arxiv.org/abs/2501.01945). Detailed taxonomy and future directions: not visible to me.
- The Awesome-Cold-Start-Recommendation README (fetched) lists LLM cold-start entries under sections "LLMs for Relation Augmentation", "Prompting LLMs for Recommendation", "Efficient Tuning LLMs for Recommendation", including WSDM 2025 *Large Language Model Simulator for Cold-Start Recommendation*, FilterLLM, SIGIR 2025 *Comprehending Knowledge Graphs with LLMs for Recommender Systems* (arXiv 2410.12229), ECIR 2025 *LLM is Knowledge Graph Reasoner* (arXiv 2412.12464), and CIKM 2025 *MI4Rec: Pretrained Language Model based Cold-Start Recommendation with Meta-Item Embeddings* — [GitHub list](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation). (Several of these were first posted on arXiv in 2024; treat them as venue 2025 / first publication 2024.)
- **LLMs Integration in Recommender Systems: A Comprehensive Survey of Frameworks, Taxonomies and Applications** [SNIPPET] — Ardeshir Shojaei, Shahla Asadi, Mostafa K. Ardakani, Mahmood Safaei; IEEE Access, 2026-01-01 (SciSpace), DOI 10.1109/access.2026.3681933; SLR of 55 studies (2018-2025); taxonomy includes cold-start mitigation approaches and integration strategies: embedding-based, prompt-based, fine-tuned, agent-oriented — [DOI](https://doi.org/10.1109/access.2026.3681933).
- **A Survey on Large Language Models for Recommendation Systems** [SNIPPET] — Junlei Yang; 2026-10-09 (SciSpace), DOI 10.54097/rparmz14; paradigms: representation-enhanced, user-interest/context enhancement, prompt learning/task reformulation, generative recommendation; challenges: hallucination, inference efficiency, privacy, fairness, recommendation agents — [DOI](https://doi.org/10.54097/rparmz14). Single author, cold-start not a focus.
- A 2026 CMC article (sciopen 10.32604/cmc.2026.079504) was surfaced by WebSearch as a survey noting that LLM reliance on static parametric knowledge limits robustness under long-tail and cold-start; snippet only — [SciOpen](https://www.sciopen.com/article/10.32604/cmc.2026.079504).
- Not a survey despite its title ("survey-driven conceptual framework"): *LLM-Boosted Graph Contrastive Learning for Cold-Start Cross-Domain Recommendation* (Chandan Gautam, Bam Bahadur Sinha; IEEE IATMSI 2026-03-12) - explicitly no experimental evaluation [SNIPPET].

### Inferences
- Open problems recurring across the verified primary papers (not taken from the survey text): hallucination of LLM-generated item knowledge (LLM-FSGNN, ColdRAG, SCSRec), inference cost of per-pair LLM calls (FilterLLM, SetLLM, 2606.29947), warm-performance degradation after augmentation (2511.21989), proprietary/non-reproducible data (Netflix), and low retrieval coverage of new items (2606.29947).

### Gaps
- The full taxonomy/future-directions text of 2501.01945 was not retrieved (only abstract page). I did not search for other 2025-26 cold-start surveys beyond those above; a systematic search for e.g. "agentic recommender cold-start survey" was not run.

---

## Key question 4 (extra, requested): Verification of the user's "known starting points"

- **LaReS (Neural Networks 2026)**: exists in SciSpace with DOI 10.1016/j.neunet.2026.109401, date 2026-07-20, authors as above, claims P@5 +7.98% on MovieLens, code at github.com/wuzelong/LaReS (README read). Publisher page and arXiv not opened: [SNIPPET + README]. Venue/date from SciSpace metadata only.
- **LLM-FSGNN (ACM 2026)**: exists in SciSpace (DOI 10.1145/3811919, 2026-06-04); MAE gains on ML-100K/1M and Amazon. [SNIPPET].
- **LLM Reasoning for Cold-Start Item Recommendation (2511.18261)**: verified [FULL]; it is Netflix-proprietary, not MovieLens; see rank 7 for the warm-vs-cold caveat on the "8%" claim.
- **BPE token-level embedding initialization (2025)**: verified to exist as arXiv 2509.13179 (v1 16 Sep 2025) [ABS]; no dataset/number/code visible; low verifiability.
- **Survey 2501.01945**: verified [ABS] (see Key question 3).

---

## Older baselines mentioned (2024 or earlier; NOT part of the catalogue)
- ColdLLM / LLM-InS ("Large Language Model Interaction Simulator for Cold-Start Item Recommendation", Huang et al., arXiv 2402.09176, Feb 2024; WSDM 2025 version titled "Large Language Model Simulator for Cold-Start Recommendation") - main baseline of FilterLLM/SetLLM.
- Wang et al. 2024, "Large Language Models as Data Augmenters for Cold-Start Item Recommendation" (arXiv 2402.11724) - baseline in 2511.21989 and FilterLLM.
- KALM4Rec (arXiv 2405.19612, May 2024) - baseline in ColdRAG.
- LLMTreeRec (arXiv 2404.00702, COLING 2025 proceedings) - first posted Apr 2024.
- LLMRank / TaxRec (arXiv 2406.14043 first posted 2024) - baselines in ColdRAG.
- Language-based User Profiles for Recommendation (arXiv 2402.15623, Feb 2024; MovieLens, cold-start).
- Sanner et al. 2023, "LLMs are Competitive Near Cold-start Recommenders" (arXiv 2307.14225).

---

## Tool notes
- **SciSpace search-papers**: best for discovery. Returns abstract, date, DOI, authors; semantic search with full questions surfaced nearly every 2025-2026 candidate above (LaReS, LLM-FSGNN, SetLLM, ProbLLM, SCSRec, LLM-CF, surveys). Limits: no date filter (results mix 2023-2026), duplicate records for the same paper (arXiv + publisher DOI), dates are sometimes the SciSpace-indexed date rather than first posting date (e.g. M-LLM3REC shows 2025-11-07 vs arXiv v1 2025-08-21), no code/datasets/numbers beyond abstracts, no full text for most items, ~10 results per query.
- **WebFetch of arxiv.org/abs/<id>**: reliable for title, authors, v1 date, version history; useless for datasets/numbers (abstract only). **WebFetch of arxiv.org/html/<id>v1**: worked well for full text (extracted datasets, protocols, numbers) for 2512.13001, 2511.18261, 2511.21989, 2502.16924, 2505.20773, 2606.29947; one timeout ("Socket is closed") on the first attempt, succeeded on retry. The fetch model summarises the page, so table numbers can contain extraction errors (flagged where I noticed inconsistencies).
- **WebSearch**: good for finding arXiv IDs and for catching papers SciSpace missed (2512.13001, 2511.21989, ColdRAG, 2606.29947, 2502.16924, BPE paper); returns summaries of unknown fidelity - I used it only to find IDs and re-verified via fetch where possible. Queries mixing many keywords returned reference-list noise; short topical queries worked better.
- **WebFetch of GitHub README / awesome list**: worked (LaReS README; Awesome-Cold-Start-Recommendation); the awesome-list extraction only covers titles with "LLM" in them.
- **Wiley full-text search**: loaded but NOT used; the target literature is arXiv/ACM/IEEE/Elsevier and Wiley is unlikely to cover it.
- **Not done / coverage gaps**: no ACM DL, ScienceDirect, OpenReview or Google Scholar pages were opened (paywalls/JS), so ACM/Elsevier/IEEE 2026 items are snippet-level. No systematic sweep of arXiv listing pages by month, so 2026 arXiv coverage is probably incomplete. HuggingFace dataset `yufan/recsys-papers-2025-2026` surfaced in a WebSearch result and may be a useful bulk index; I did not open it.
