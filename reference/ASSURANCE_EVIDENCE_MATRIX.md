# Claim–evidence matrix for reviewers and authors

| Research claim | Evidence in updated manuscript/source | Valid scope | Known limitation |
|---|---|---|---|
| Missing views receive exactly zero fusion weight | Masked-softmax definition, numerical tests | All nonempty masks with correct implementation | All-missing inputs must be rejected |
| Bounded participation for fixed availability | Fixed-mask propositions and design floor | Fixed mask and `kappa` satisfying the floor rule | High gate strengths can violate a specified 10% floor |
| Mask change with view addition/removal gives Hellinger geometry bound | New mask-switch proposition and unit tests | Bounded view vectors and bounded drift on common observed views | Does not automatically estimate semantic value of the new/removed view |
| Exact norm jump on hard view dropout | New exact identity, randomized public Digits and 160 global synthetic cases | Row-normalized separated blocks, unit-norm views, retained logits/embeddings frozen, nonempty new mask | Cannot be transferred unchanged to unnormalized sum fusion or to changing encoders |
| Positive view-participation floor imposes unavoidable norm jump | New sharp tradeoff proposition | Same exact dropout assumptions | A structural tradeoff, not a neural performance improvement |
| A point remains nearest after one membership-preserving centroid refit | New Lloyd-certificate proposition; 64-test unit suite; Digits and two observational group audits | Explicit per-record perturbation budgets and fixed original partition | Does not guarantee global optimum, subsequent Lloyd convergence or any semantic accuracy |
| Globally optimal partition unchanged | Previous exhaustive objective-gap proposition and new exact small-n masks | Full enumeration of all finite `K`-partitions with valid budgets | Computationally exponential and not usable as a deployment-scale guarantee |
| A 10% random view-loss event can be certified safely | Digits 5 disjoint test folds | 77.1% of all records, 0% of actually dropped records certified; remaining 99.6% observed stable | Cannot advertise this as 77.1% of damaged records protected |
| Lower retained weight makes losing that channel easier to certify | Idealized score-vector `[0,1,1]` stress test | Designed scores only; 167 dropped rows and five folds | Stronger `kappa` sacrifices participant floor, artificial score assumptions |
| External validation supports general semantic gains | Original ANES and transport-choice first-run scores | Mixed and statistically uncertain observational descriptive results | No independent natural sensor modalities, sources or confirmatory independent populations |
| Official specialist clustering models are benchmarked | Missing | None | `OFFICIAL_CODE_COMPARATOR_STATUS.json` remains `NOT_RUN` |
| Final IJSS submission policies and authorship verified | Funding/disclosure incorporated; checklist and AI-use disclosure | Not fully verified | Both authors must approve statements, CRediT and publisher AI-policy compatibility |
