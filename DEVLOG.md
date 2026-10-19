# Dev log

## 2026-10-19 — Kafka serving and Hadoop archive

Added an optional serving bundle exporter, Kafka observation replay, transactional
scoring/dead-letter handling, an idempotent SQLite decision projection, and a
Streamlit live mode that does not load the research pipeline. Added Spark jobs
for HDFS archival and preparing history from explicitly observed labels, plus a
local Docker Compose deployment and docs/STREAMING.md.

Validation: the existing 57 tests passed, and 20 new streaming/dashboard tests
passed, including actual LightGBM save/load score parity. Fixed JSON nulls causing
the existing UID resolver's one-row numeric conversion to fail. These checks ran
on local Python 3.11 with compatible available dependencies; the Docker image
uses Python 3.13 and the repository's pinned requirements. Python compilation,
Compose YAML parsing, Hadoop XML parsing, and git whitespace checks passed.
Docker is absent and the real CSVs are absent, so no trained real-data bundle or
running broker/HDFS deployment was produced here. The optional Spark test was
skipped; a real-data Kafka round-trip smoke command is provided for deployment.
The original training-label and threshold-selection limitations are preserved
and disclosed in manifests and the run guide; no model-quality claim was changed.

Honest, specific notes on what was built, what was surprising, and what broke
or needed correcting. Written as work happens, not cleaned up afterward.

## 2026-08-31 — Session setup (branch, CLAUDE.md rules)

Branched to `uid-validation` off `main`. Added two rules to CLAUDE.md:
test_transaction.csv has no labels (it's the Kaggle competition's actual
holdout, never released to competitors), so every train/test split in this
project has to happen inside train_transaction.csv; and synthetic label/score
arrays used as metric-function test fixtures (e.g. `np.array([0,1,1,0])`) are
explicitly not "fraud generation" under the defense-only rule. `.venv/` was
already covered by the existing `.gitignore`, so nothing to add there.

Worth recording from the *previous* scaffolding session even though it
predates this devlog: the original `.gitignore` had a bare `data/` rule plus
`!data/.gitkeep`, intending to keep the placeholder trackable while ignoring
real data. That doesn't work — git's negation rule cannot re-include a file
inside a directory that's already excluded at the directory level, so
`data/.gitkeep` was silently still ignored. Fixed by changing the pattern to
`data/**` (ignores contents, not the directory entry itself), which lets the
negation actually take effect. Verified with `git check-ignore -v` and
`git add --dry-run`.

## 2026-08-31 — Task 1: src/data.py

Implemented `load_transactions`: left-join train_transaction.csv with
train_identity.csv on TransactionID (left, not inner, since identity covers
only ~24% of transactions), downcast float64->float32 and low-cardinality
(<=50 unique values) object columns to category, and raise a
`FileNotFoundError` naming `scripts/download_data.sh` when either CSV is
missing.

Added a root-level `conftest.py` (empty file) so `tests/` can `import src.*`
regardless of how pytest is invoked. Without it, pytest's default "prepend"
import mode only adds the `tests/` directory itself to `sys.path` (since it
has no `__init__.py`), not the repo root — `import src.data` would fail
depending on cwd. This wasn't something the task called out explicitly; found
it by reasoning through pytest's import mechanics rather than by hitting the
failure, so it's untested against every possible invocation style, just the
common ones (`pytest`, `python -m pytest`, both from repo root).

The category-cutoff constant (50 unique values) is a judgment call, not
something tuned against real data — no CSVs are present in this environment
yet (see Task 3 below), so the actual cardinality of columns like ProductCD,
card4/card6, M1-M9, or the id_* columns hasn't been checked. It may need
revisiting once real data is loaded; DeviceInfo in particular is known from
the competition's public documentation to have far more than 50 distinct
values and will correctly stay `object` under this rule.

Tests cover only the missing-file error path (3 cases: both missing, identity
missing, transaction missing), per the task's explicit scope — the
load/join/downcast happy path needs the real dataset.

## 2026-08-31 — Task 2: evaluate.py implemented and frozen

This is the metrics module that can never be edited later to look better, so
getting the design right now mattered more than usual. Notes on the real
decisions made, since the task description left some of them open:

- **temporal_train_test_split**: a naive "take the row at the (1-test_size)
  quantile" split can land in the middle of a group of rows sharing one
  TransactionDT value, putting identical timestamps on both sides of the
  boundary. Implemented as a backward walk over *unique* TransactionDT values
  from most recent to oldest, accumulating each group's full row count atomically,
  stopping once the accumulated count reaches the target test size. This
  guarantees no timestamp group is ever split, at the cost of the realized
  test fraction sometimes drifting from the requested `test_size` when a
  large group straddles the boundary (tested explicitly with a synthetic
  20-row tied group). Added an internal `assert train[dt_col].max() <
  test[dt_col].min()` inside the function itself as a cheap self-check, given
  how much rides on this invariant actually holding.
- **evaluate_model's signature** needed `cost_fn`/`cost_fp` parameters that
  weren't in the task's literal function-reference list, since it has to call
  `cost_per_10k` internally and the instructions were explicit that costs must
  never be hardcoded. Added them as optional kwargs defaulting to a neutral
  1:1 ratio (not a guessed rupee amount) so callers can override with real
  business costs later.
- **evaluate_model calls `model.predict(X_test)` directly** (LightGBM's native
  Booster API), not `src.model.predict()`. src/model.py is explicitly out of
  scope for this session and still raises NotImplementedError, so routing
  through it would make evaluate.py's tests fail for reasons that have
  nothing to do with evaluate.py. This does mean evaluate.py currently assumes
  callers pass something with a LightGBM-compatible `.predict()`, which is
  worth knowing about when model.py is implemented later.
- **cost_per_10k's threshold semantics**: chose `score >= threshold` (not
  `>`) as "flagged". Arbitrary but has to be picked consistently somewhere,
  and this is the natural reading of "at or above."
- The pr_auc "near base rate" test uses a fixed-seed (`default_rng(0)`)
  20,000-row uncorrelated random score vs. a 3.5% synthetic base rate, with an
  0.02 absolute tolerance picked before running the test rather than measured
  first. It passed on the first run (no correction needed), but that
  tolerance was a guess about how much a fixed-seed run of that size would
  wobble around the true base rate, not a derived bound — if this test ever
  gets flaky after a numpy/sklearn version bump, that tolerance is the first
  thing to revisit.

Full suite result (14 tests, all passing):

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\sohan\Desktop\not-strangers\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\sohan\Desktop\not-strangers
plugins: anyio-4.14.2
collecting ... collected 14 items

tests/test_data.py::test_raises_when_both_files_missing PASSED           [  7%]
tests/test_data.py::test_raises_when_identity_missing PASSED             [ 14%]
tests/test_data.py::test_raises_when_transaction_missing PASSED          [ 21%]
tests/test_evaluate.py::test_temporal_split_train_strictly_before_test PASSED [ 28%]
tests/test_evaluate.py::test_temporal_split_never_splits_a_tied_timestamp_group PASSED [ 35%]
tests/test_evaluate.py::test_temporal_split_rejects_invalid_test_size PASSED [ 42%]
tests/test_evaluate.py::test_pr_auc_perfect_ranking_is_one PASSED        [ 50%]
tests/test_evaluate.py::test_pr_auc_uncorrelated_scores_are_near_base_rate PASSED [ 57%]
tests/test_evaluate.py::test_recall_at_fpr_perfect_classifier_gets_full_recall PASSED [ 64%]
tests/test_evaluate.py::test_recall_at_fpr_inverted_classifier_gets_near_zero_recall PASSED [ 71%]
tests/test_evaluate.py::test_cost_per_10k_hand_computed PASSED           [ 78%]
tests/test_evaluate.py::test_cost_per_10k_is_parameterised_not_hardcoded PASSED [ 85%]
tests/test_evaluate.py::test_evaluate_model_returns_expected_keys PASSED [ 92%]
tests/test_evaluate.py::test_evaluate_model_accepts_threshold_and_cost_overrides PASSED [100%]

============================= 14 passed in 1.66s ==============================
```

## 2026-08-31 — Task 3: data check — stopping here

`data/train_transaction.csv` (and `data/train_identity.csv`) are not present
in this environment -- `data/` contains only `.gitkeep`. Per instructions,
skipping Task 4 (entities.py UID derivation) and Task 5
(`results/uid_validation.md` plus the size-distribution histogram) rather than
attempting a Kaggle download, since the Kaggle API credentials and
competition-rules acceptance aren't set up in this session.

Everything in Tasks 4 and 5 needs the real dataset to produce honest numbers
(row counts, uid label-purity, the actual distribution of transactions per
uid) -- there's no meaningful way to do them against synthetic data without
either fabricating the exact numbers the deliverable is supposed to report,
or building a synthetic-fraud-pattern generator to make the numbers
interesting, and both of those cut against what this session is for.

To pick this back up: run `bash scripts/download_data.sh` (needs
`pip install kaggle`, a Kaggle API token at `~/.kaggle/kaggle.json`, and
having accepted the competition rules -- see that script's header comment),
then re-run Task 4 and Task 5 from the original instructions.

## 2026-08-31 — entities.py implemented, uid_validation.md computed on real data

Kaggle credentials were set up between sessions and the real CSVs were
already present in `data/` (683MB train_transaction.csv, 26.5MB
train_identity.csv) -- no download needed this time.

Implemented `card1_addr1_origin_day` in `src/entities.py`. Changed
`resolve_entities`'s signature from the original scaffold stub
(`resolve_entities(transactions, key_columns)`, a generic union-find-style
design) to `resolve_entities(transactions)`, since the uid formula is now
fully specified and there's no longer a set of candidate key columns to
choose between -- keeping the generic parameter would have been dead
flexibility. 5 unit tests on a small hand-built frame (persistence across
days, discrimination on origin_day, all three null-cause cases, no dropped
rows), all passing.

Wrote `scripts/validate_uids.py` as a standalone analysis script (not wired
into run_pipeline.py or the Makefile) that reads only the 6 columns needed
from train_transaction.csv, computes every number in the spec, and writes
`results/uid_validation.md` and `results/uid_size_distribution.png` directly
from the computed values -- deliberately not hand-transcribed, to remove
transcription-error risk between the computed numbers and the report.

**Got wrong and corrected:** the script's first run crashed with
`pandas.errors.IndexingError: Unalignable boolean Series` at
`df.loc[uid.notna(), ["isFraud"]]`. `resolve_entities` returns a Series
indexed by TransactionID (as documented and tested), but `df` still had a
plain 0..n-1 RangeIndex from `read_csv`, so the boolean mask and the frame
didn't line up positionally by index label. Fixed by `df =
df.set_index("TransactionID")` immediately after computing `uid`, before any
`.loc` masking against it. Worth flagging because this is exactly the kind
of mismatch that silently produces wrong numbers instead of crashing when
the two objects happen to share a compatible-looking index -- here it
crashed loudly, which is the good outcome, but it's a real gotcha for any
other code that consumes `resolve_entities`'s output against the original
transactions frame.

**What surprised me:** two real findings from the actual data (reported
honestly, derivation NOT tuned to change either of them):

1. Fraud rate is very different between rows that got a uid and rows that
   didn't: **2.46%** for uid'd rows vs. **11.63%** for NaN-uid rows (see
   uid_validation.md section 5). Almost all of the NaN cases are addr1 being
   null (65,525 of 66,794), so this says addr1-missing transactions are
   substantially higher-risk than average, not just an artifact of the join. Any
   downstream cluster feature that silently excludes NaN-uid rows is
   therefore excluding a disproportionately fraud-heavy slice, not a random one.
2. `origin_day` (`day - D1`) is negative for **30.36%** of all transactions
   (mean -206.6, min -633), not just a rare edge case -- e.g. the largest
   impure uid in the report, `12695_325_-342`, has origin_day = -342. D1 is
   documented as "days since the card was first seen," so a negative
   origin_day means D1 exceeds the transaction's own elapsed-day count,
   which shouldn't happen if D1 and TransactionDT shared one consistent
   day-zero reference. It's a known-ish quirk of this dataset's D-columns
   (they don't reliably share TransactionDT's epoch), and it doesn't break
   the uid as an identifier -- the 98.53% unweighted purity result shows the
   card1+addr1+origin_day combination is still highly stable per persistent
   client -- but it does mean origin_day's literal value should not be read
   as a real calendar day. Not something to fix here per instructions (the
   derivation is exactly as specified); just flagging that the number's sign
   isn't meaningful on its own.

Label purity result: 98.53% of the 83,557 multi-transaction uids are fully
label-pure (unweighted), 97.61% weighted by transaction count -- the uid
derivation collapses card+address+cohort into a genuinely consistent-label
identity almost all of the time. Full numbers, the ten largest impure uids,
and the size-distribution histogram are in results/uid_validation.md.

## 2026-08-31 — D1/origin_day investigation (read-only, no derivation changes)

Follow-up on the negative-origin_day finding from the previous entry.
`src/entities.py` and `src/evaluate.py` untouched -- this was pure
investigation. Wrote `scripts/investigate_d1.py` -> `results/d1_investigation.md`.

**The root cause is now clear and it's benign.** Split by D1 within each
group tells the whole story: the "positive" group is 68.51% D1=0 (median
D1=0), i.e. mostly cards being seen for the very first time within the
dataset's own window, where origin_day trivially equals the transaction's
own day. The "negative" group has median D1=252 and max D1=640 -- these are
transactions from cards whose documented history (D1, "days since first
seen") already exceeds the entire ~year-plus span of TransactionDT covered
by this file. D1 isn't reset to this dataset's own day-zero; it reflects a
card's real prior history, which can predate the collection window
entirely. So negative origin_day isn't corrupted data, it's the arithmetic
consequence of joining an absolute "card age" feature against a
dataset-relative day counter. This matches last entry's guess ("D1 and
TransactionDT don't share one consistent day-zero") but now has the actual
mechanism behind it instead of just the symptom.

**Answering the four questions, briefly (full tables and numbers in
results/d1_investigation.md):**

1. Yes, negative-origin_day rows are a different population, not noise:
   lower fraud rate (1.60% vs 4.33%), much lower identity-record coverage
   (7.35% vs 31.97%), and a heavily skewed ProductCD mix (92.07% ProductCD=W
   vs 66.66%). They look like an older/more-established, lower-risk slice of
   traffic, consistent with "cards with a longer prior history."
2. D1 is not null-heavy in either group (can't be, by construction -- null
   D1 makes origin_day NaN, which is excluded from both groups entirely).
   It IS zero-heavy in the positive group (68.51%) and essentially never
   zero in the negative group (0%) -- this is the actual mechanism above.
3. Purity does NOT hold equally: 99.24% (weighted) for negative-origin_day
   multi-transaction uids vs 96.61% for positive. Per the task's own
   framing, that's the important case -- origin_day's sign is not cosmetic,
   it correlates with how trustworthy the resulting uid's label consistency
   is. Worth carrying forward: cluster features built on positive-origin_day
   (newer-card) uids are somewhat noisier than ones built on
   negative-origin_day (established-card) uids.
4. Collision check on the 20 largest uids: card2/card3/card5 are constant
   (1 distinct value) across literally all 20 -- expected, since they're
   sub-attributes of the same physical card as card1, not independent
   signals, so this adds nothing. P_emaildomain is where the real signal
   is: it varies (2-3 distinct values) in **10 of the 20** largest uids.
   card1+addr1+origin_day is genuinely merging multiple distinct people in
   half of the largest clusters -- confirmed by data, not assumed. This is
   "stability without correctness" exactly as the task named it: high label
   purity is not evidence the uid identifies one physical client, only that
   whoever it merges together tends to share a fraud outcome. Said plainly
   in the report; derivation was not touched in response to this.

**Nothing got corrected this time** -- the script ran cleanly on the first
attempt, most likely because the index-alignment gotcha from the previous
session (`df.set_index("TransactionID")` right after calling
`resolve_entities`) was already a known pattern going in, not something
rediscovered here.

## 2026-08-31 — Task 1: graph.py implemented, and a real methodology problem found and fixed

Implemented src/graph.py: LINKAGE_RULES (device_info, addr1_email,
card_bank_addr, each with a one-line rationale), build_entity_graph (hub
guard + edge construction, returns an EntityGraph(graph, excluded_hubs)
dataclass rather than a bare nx.Graph -- signature change from the scaffold
stub, needed to actually report what the hub guard excluded), get_connected_
components, and compute_cluster_features (10 columns: cluster size/txn
count/edge density/velocity/amount CV/burst concentration/email-uid ratio/
prior-fraud share, plus per-uid node degree and email domain count).
tests/test_graph.py: 10 tests, including the required explicit leakage test
(a future, huge-amount, fraud-labeled transaction for an already-clustered
uid must not change that uid's or its cluster's features at all -- verified
by asserting the feature frame is byte-for-byte identical with vs. without
that row).

**What surprised me, and it's a big one:** a 50k-row smoke test before
moving to Task 2 showed 29,211 of 30,185 nodes (96.8%) collapsing into ONE
connected component at the task's specified default, max_degree=1000. That
is not a ring, it's almost the whole active population -- cluster features
computed on it would be nearly constant across nearly every transaction and
carry no discriminating signal. Investigated properly on the full dataset
(524k uid'd rows) with a cheap union-find (no need to materialize edges to
check this): at max_degree=1000, the largest cluster is 127,708 uids (64% of
all 199,070 uids). Root cause: addr1 is a region-level code with only a few
hundred distinct values across the whole file, and card3/card5 are
dominated by one or two values (150.0/226.0 cover the overwhelming
majority) -- so card_bank_addr is nearly a proxy for addr1 alone, not an
independent signal, and even "below-hub" addr1-based groups (100-999
members) bridge transitively into one supercluster once enough of them
overlap.

Swept max_degree on the full dataset: 15 -> largest 77 uids; 20 -> 126;
25 -> 526; 30 -> 919; 35 -> **5,141**; 40 -> 6,625; 45 -> 10,074; 1000 ->
127,708. There's a sharp phase transition between 30 and 35 (a 5.6x jump),
not a gradual one -- this is a real structural property of the linkage
rules on this data, not noise. Also relevant for tractability: my clique-
based edge construction (itertools.combinations per shared-value group) is
O(k^2) per group; at max_degree=1000 a single near-threshold group can
contribute ~500k edges, and the 50k-row smoke test alone produced 6.4M
edges in 19s -- at full scale with the literal default this would likely be
tens of millions of edges and impractically slow. Lowering max_degree fixes
both problems at once.

**Fix:** kept build_entity_graph's own default at max_degree=1000, exactly
as specified in the task -- but run_pipeline.py calls it with max_degree=20,
documented prominently in graph.py's module docstring with the full sweep
so the number isn't just buried in a commit. Verified at full scale with
the real function (not just the union-find estimate): max_degree=20 builds
in 3.2s (199,070 nodes, 65,223 edges), connected components in 0.35s
(largest 126 uids), and compute_cluster_features over all 199,070 uids in
15.8s. This is a graph-construction/methodology decision, not a metric being
tuned -- the ablation lift hasn't been measured yet (that's Task 2) and this
choice was made before seeing any model result, purely from cluster-size
distributions.

**Got wrong and corrected (in the investigation, not in graph.py itself):**
my first attempt at investigating this on the real data returned "0 rows
covered" for every single linkage rule -- looked like a data problem, but it
was the exact same indexing bug class flagged in the entities.py DEVLOG
entry, just recommitted in a throwaway analysis script: I assigned
`df["uid"] = uid` without first calling `df.set_index("TransactionID")`, so
pandas aligned the RangeIndex against uid's TransactionID index and got
NaN almost everywhere. graph.py's own `_prepare()` helper already does this
correctly (that's precisely why it exists as a shared helper rather than
being repeated inline) -- the bug was only in my scratch investigation code,
not in the module being tested.

## 2026-08-31 — Task 2: model.py + run_pipeline.py implemented, ablation produced

model.py: train_baseline_model and train_cluster_model both call the same
private _fit() helper -- literally the same code path, same LGBM_PARAMS
dict (seed=42), same NUM_BOOST_ROUND=300 -- so the only way the two models
can differ is which columns are in X. build_feature_matrix drops isFraud
and a raw "uid" column if present, deliberately: the cluster model gets the
engineered cluster STATISTICS, never the raw uid string, so it can't just
memorize "this exact uid was fraud in training" instead of learning from
the aggregated signal -- that would inflate the ablation for reasons
unrelated to the cluster features actually being tested. 4 tests in
tests/test_model.py, including one that runs both training functions on
identical data and asserts the resulting boosters serialize identically
(`model_to_string()` equal) -- proving "identical treatment" structurally,
not just by eyeballing the params dict.

run_pipeline.py: load_and_prepare -> train_both_models -> evaluate_both_
models -> write_ablation_report, plus main(). Exposed as importable
functions (not just a __main__ script) so Task 3/4's scripts can reuse the
same data/graph/model artifacts without copy-pasting this orchestration.
Graph is built once from train-only data (max_degree=20, per Task 1's
finding) and cluster features are computed once with
as_of=test_df["TransactionDT"].min() -- the first test-period timestamp --
then broadcast from per-uid to per-TransactionID and left-joined onto both
train and test rows. Rows with no uid (~11%, see uid_validation.md) or a
uid with no pre-as_of history simply get NaN cluster columns via that
left-join -- never dropped, never a fabricated zero, exactly as instructed.
Policy.py is NOT called from here (out of scope this session, per
instructions) -- updated run_pipeline's module docstring accordingly since
the original scaffold-stub sequence included an apply_policy step.

**Result on the real full-scale temporal split** (472,432 train /
118,108 test rows, cost_fn=500, cost_fp=5, both illustrative and stated as
such in ablation.md):

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30,078.40 |
| cluster | 0.6322 | 0.5576 | 26,155.72 |

+0.0676 PR-AUC, +0.0785 recall@1%FPR, -3,922.68 cost per 10k. This is not a
small or marginal lift -- reported as-is, nothing was adjusted after seeing
it.

**What surprised me, and it needs Task 3 before anyone should believe it:**
`cluster_prior_fraud_share`'s feature-gain is 558,334 -- about 9x the
*second*-place feature (V258 at 63,550) and roughly 20x most others. That
single feature dominating this heavily is exactly the shape of a leak, even
though the unit-level leakage test in test_graph.py (future fraud-labeled
transaction must not change any feature) already passed. A passing unit
test on a small synthetic case proves the *mechanism* filters as_of
correctly; it doesn't by itself prove there's no leak at the *full-pipeline*
level (e.g. in how as_of is chosen, or in what "prior" means across the
train/test boundary in aggregate). Not fixing or second-guessing this
number now -- Task 3 is specifically the trace-it-and-report-plainly step,
next.

Runtime at full scale, for the record: ~35s to load data + build the graph
+ compute cluster features, ~58s to train both models. Cheap enough that
Task 3/4 can just re-run load_and_prepare()/train_both_models() fresh
rather than needing to cache artifacts to disk.

**Got wrong and corrected:** first cut of build_feature_matrix used
`select_dtypes(include="object")`, which raised a Pandas4Warning in
pandas 3.0.5 about "str" dtype columns being swept in implicitly under
"object" and that becoming an error in a future pandas version. Fixed by
being explicit: `include=["object", "str"]`. Caught by the model.py unit
tests before it ever touched the real 434-column dataset.

## 2026-08-31 — Task 3: adversarial sanity checks -- no leak found, but the lift is concentrated

Wrote scripts/sanity_checks.py, appending a "## Sanity checks" section to
results/ablation.md (not a separate file, since the re-ablation explicitly
needed to land as a second row in the same results table).

**1. Correlation with isFraud (train set):** cluster_prior_fraud_share is
0.7797 -- the only feature over the 0.5 red-flag threshold by a wide
margin (next highest is cluster_velocity at 0.057). This alone doesn't
prove a leak, but it's exactly the shape one looks like, so it drove
section 2.

**2. Traced cluster_prior_fraud_share for a leak -- found none, but only
after fixing my own check.** First attempt picked the first cluster with
*any* test-period fraud row as the demonstration example, and got a false
positive: reported, independently-recomputed, and "if-leaked" values all
came out identical (0.0563), which my comparison logic mislabeled as
"MISMATCH" because it required the leaked value to differ from the
reported one to count as a pass. Looking at *why* they matched: the
example uid already had fraud labels in the *train* period too, so its
per-uid "ever fraud" flag was already 1 before the test-period row existed
-- the test-period fraud couldn't have changed anything even if it had
leaked in, so the example proved nothing either way. That's a test-design
bug, not a pipeline bug, but it would have shipped a false "leak found" if
I'd trusted the first run's verdict text without reading the numbers.
Fixed by computing expected-vs-leaked prior-fraud share for *every*
cluster and picking the one with the largest gap as the example --
guarantees a non-vacuous demonstration instead of hoping for one. Also
added a global check across all clusters, not just the one example.

Real result: **0 mismatches across all 155,579 clusters** between the
pipeline's reported cluster_prior_fraud_share and an independent
recomputation from raw rows with `TransactionDT < as_of` (bypassing
graph.py entirely). The maximally-discriminating concrete example (cluster
#17894, a singleton uid with two fraud transactions, both in the test
period): reported = 0.0000, independently recomputed = 0.0000, what it
would be if test-period rows had leaked in = 1.0000. No leak.

**Why the correlation is still so high without leaking:** CLAUDE.md
already documents that labels "propagate across a card once reported."
cluster_prior_fraud_share is close to a direct measurement of exactly that
dynamic -- "has this persistent card-identity already been caught" --
which is fully legitimate as a *causal, non-leaking* feature, but its
predictive power comes largely from the same noisy label-propagation
process this project already flags as a caveat, not from discovering an
independent abuse signal. Worth remembering when reading the headline
lift number: high but not leaking is not the same as high and clean.

**3. Re-ablation without cluster_prior_fraud_share:**

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30,078.40 |
| cluster (full) | 0.6322 | 0.5576 | 26,155.72 |
| cluster (no prior_fraud_share) | 0.5756 | 0.4951 | 29,317.66 |

Removing that one feature shrinks the lift from +0.0676 to +0.0110 PR-AUC
(84% of the measured lift comes from that single feature), from +0.0785 to
+0.0160 recall@1%FPR, and from -3,922.68 to -760.74 cost per 10k. The
remaining lift is real, small, and driven by the other 9 structural/graph
features (edge density, velocity, burst concentration, email
heterogeneity, cluster size) -- reported as-is, not chased further to make
it look bigger.

**4. Cluster assignment independence from test-period edges: confirmed.**
0 graph nodes found with zero train-period transactions (should be 0, and
was). Concrete example: a uid appearing only in the test period is absent
from the graph and gets fully-null broadcast cluster features, exactly as
expected -- no train-period history means no cluster signal, not a
fabricated one.

**Bottom line for this task:** no leak found anywhere checked. The
headline ablation numbers from Task 2 stand, but the honest reading is
"~84% of the lift is one feature that's legitimate but rides on the
project's own documented noisy-label dynamic; the graph-structural
features alone add a smaller, genuine ~0.011 PR-AUC." Both numbers are now
in results/ablation.md; neither is hidden in favor of the other.

## 2026-08-31 — Task 4: threshold sweep and cost curve

scripts/cost_curve.py sweeps 300 thresholds (dense from 0-0.05, coarser
0.05-1.0, since the base rate is ~3.5% and both models' scores concentrate
near 0), plots both models' cost_per_10k curves, and marks each model's own
cost-minimizing point. Appends recall/FPR at each chosen point to
results/ablation.md.

**Got wrong and corrected (presentation, not numbers):** first version put
both "chosen point" annotations directly on a single full-range [0,1] plot.
Both models' optimal thresholds landed within 0.001 of each other
(0.0095 vs 0.0103) in the steep near-zero region, so the two text labels
overlapped into unreadable text. Fixed by splitting into two panels: the
full [0,1] range for shape, and a zoomed [0,0.05] panel (shaded on the
full-range plot to show where it comes from) with the annotations properly
offset apart. The underlying numbers were correct on the first run; only
the plot was unreadable.

**Chosen operating points:** baseline t=0.0095 (cost 30,009, recall 0.9326,
FPR 0.3813); cluster t=0.0103 (cost 25,923, recall 0.9530, FPR 0.3695). The
cluster model's cost-minimizing point is both cheaper AND has a lower FPR
at a higher recall -- a real, not cherry-picked, improvement here.

**Worth flagging plainly:** the cost-minimizing FPR is 37-38% for both
models. That's a direct, correct consequence of the assumed 100:1
cost_fn:cost_fp ratio (a missed fraud case is assumed 100x worse than one
false alarm, so the cost-optimal policy flags aggressively) -- not a bug,
but it means "cost-optimal" here means stepping up roughly a third of all
legitimate transactions, which is a real business-acceptability question
the illustrative cost ratio is doing all the work to answer. Said this
directly in ablation.md rather than only reporting the flattering
recall/cost numbers next to it.

## 2026-08-31 — Task 5: README results table and Limitations section

Filled in the Result table with the real numbers from all three model
variants (baseline, cluster, cluster minus cluster_prior_fraud_share),
linked to results/ablation.md for the full breakdown, and added a short
paragraph flagging that ~84% of the headline lift is one feature -- didn't
want the table alone to imply an uncomplicated win. Left "Why clusters",
"Data & labels", "Architecture" and "Running it" as empty headers: the task
asked specifically for the results table and a Limitations section, and
filling in the others wasn't asked for and isn't this session's call to
make.

Limitations section covers exactly the four things asked for: label noise
and its direct tie to cluster_prior_fraud_share's dominance; the uid
over-merging finding from d1_investigation.md (P_emaildomain varies in 10
of the 20 largest uids); the ~11% no-uid population and its much higher
fraud rate (11.63% vs 2.46%); and the no-ground-truth caveat that
everything measured here is feature lift on a fraud classifier, not
validated ring-detection accuracy.

Nothing surprising in this task -- it's a writing task pulling together
numbers already computed and verified in Tasks 1-4, not new analysis. No
corrections needed.

---

**Session summary (uid-and-graph branch, this multi-day session):**
graph.py, model.py and run_pipeline.py implemented; a genuine graph-
construction bug (giant-component collapse at the task's specified
max_degree default) found and fixed before it could contaminate anything
downstream; a real ablation run end-to-end on the full dataset; an
adversarial pass that found no leak but did find the lift is concentrated
in one feature, and said so; a cost curve with an honestly-reported
uncomfortable FPR at the "optimal" point; and a README that reflects all of
it, including the parts that don't make the result look as clean as a
single PR-AUC number would suggest. evaluate.py, entities.py, investigator.py
and policy.py were never touched, as instructed.

## 2026-08-31 — Task 1: investigator.py implemented

LLM layer over the Anthropic API (model claude-sonnet-4-6). explain_cluster
builds a flat evidence dict (build_evidence) from a cluster's member rows
of graph.compute_cluster_features output plus their raw transactions, sends
it to the model as a JSON code block (never prose) with a system prompt
whose hard rule is "never state a number not present in this JSON verbatim
or trivially rounded" -- stated explicitly as more important than the rest
of the prompt, since that's the one constraint this session actually checks
programmatically (Task 2, next). prioritize_clusters sorts by a simple,
clearly-labeled heuristic priority_score (prior-fraud share weighted
heaviest, txn volume and burst concentration as tie-breakers) -- explicitly
not a policy decision, no thresholds, no actions.

Graceful degradation: explain_cluster checks for ANTHROPIC_API_KEY before
attempting a call, and wraps the actual API call in a bare
`except Exception` -- either path returns a ClusterExplanation with
source="ungrounded-fallback" and a narrative built by directly formatting
the evidence dict's own values (grounded by construction, since it can't
contain anything that isn't already a dict value). Never raises either way.
6 tests, including one that monkeypatches `_call_anthropic` to raise and
confirms the fallback still returns cleanly.

**Important honest note for Task 2, which measures this module next:**
`ANTHROPIC_API_KEY` is NOT set in this environment. Every explain_cluster
call made in this session will take the fallback path. Task 2's
"groundedness" measurement will therefore be measuring the deterministic
fallback template (which is grounded by construction, since it's built by
formatting the evidence dict directly) -- not the actual LLM's behavior
under the prompt's hard-number rule. This is a real limitation of what this
session can demonstrate, not something to paper over: the LLM-calling code
path (_call_anthropic) is implemented and unit-tested with a mocked
success case, but has never actually been exercised against the real API
in this session. If a key is added later, re-running
scripts/eval_investigator.py is what would produce a real measurement of
the LLM's groundedness under the prompt.

## 2026-08-31 — Task 2: measured the investigator, confirmed the caveat from Task 1

scripts/eval_investigator.py -> results/investigator_eval.md. Selected 30
clusters spanning the risk range (all multi-uid clusters from the train
graph sorted by cluster_prior_fraud_share, 30 evenly-spaced percentile
points -- not just the top 30 riskiest, so low-risk and ambiguous clusters
are represented too, e.g. cluster-13 above with prior_fraud_share=0.0).
Groundedness check extracts every numeric literal from each narrative via
regex and checks it against the cluster's evidence dict, allowing rounding
to 0-4 decimals and percentage form (v cited as v*100).

**Result: 100% groundedness (0 of 360 extracted claims ungrounded), and
this number is exactly as uninformative as flagged in Task 1's DEVLOG
entry.** ANTHROPIC_API_KEY is still not set, so all 30 explanations took
the fallback path (source=ungrounded-fallback for every single one --
confirmed, not assumed, by checking the `sources` counter in the script's
output). The fallback template is grounded by construction (it's built by
directly formatting evidence dict values as `key=value` pairs), so 100%
was the only possible outcome here, not evidence that claude-sonnet-4-6
follows the hard-number-rule prompt. Said this plainly in
investigator_eval.md's first paragraph rather than reporting "100%
groundedness!" as if it validated the LLM.

What this task DID validate, honestly: the selection-and-measurement
*pipeline* itself works end to end -- risk-spanning cluster selection,
evidence assembly, narrative generation, numeric extraction, and grounding
comparison all ran correctly on real data (30 real clusters, real evidence
values, e.g. cluster-29's prior_fraud_share=1.0 with only 3 members and 7
transactions -- a small, high-confidence cluster). If a key is added, this
same script is what would need to be re-run for the number that actually
matters.

No corrections needed this task -- ran cleanly on the first attempt.

## 2026-08-31 — Task 3: policy.py + audit trail

Deterministic decide/apply_policy over allow/step_up/review. Thresholds
are NOT hand-picked: scripts/derive_policy_thresholds.py sweeps the
cluster model's real test-set scores twice, both at cost_fn=500, different
cost_fp: cost_fp=5 for step_up (25,923.31 min cost at t=0.010302 -- this
exactly matches Task 4's existing cost_curve.py output for the cluster
model, a good cross-check that both sweeps agree) and cost_fp=50 for
review (75,625.70 min cost at t=0.184343). The 10x cost_fp multiplier for
review vs step_up is stated as illustrative in policy.py's own docstring,
same caveat as every other cost assumption this project has made. Hard-coded
the two results as STEP_UP_THRESHOLD/REVIEW_THRESHOLD constants with the
script named as provenance, rather than recomputing them at import time.

decide() takes cluster_features but doesn't use it in the threshold
comparison (the policy is deliberately score-only, not cluster-conditioned)
-- documented explicitly in the docstring so this isn't mistaken for a bug,
and the parameter is kept because callers building an audit record want it
alongside the decision. apply_policy is a vectorized batch path proven
equivalent to calling decide() row-by-row (tests/test_policy.py), needed
for performance across the full entity population.

MODEL_VERSION is derived from model.py's own SEED/NUM_BOOST_ROUND constants
(`cluster_seed42_boost300`) rather than a hand-typed string, so it can't
silently drift from what actually trained.

build_audit_record assembles the required fields (transaction_id, uid,
model_version, score, threshold_applied, feature_values, action, reason,
timestamp) as a pure function; scripts/write_audit_sample.py does the
actual scoring/decisioning/writing, producing 200 real records in
results/audit_sample.jsonl from real test-period transactions (131 allow,
63 step_up, 6 review). Left several feature_values fields null in the
sample on purpose rather than filtering them out: some sampled
transactions have a uid with no pre-as_of cluster history (null cluster
features, exactly the "new-in-test-period" case established in Task 2 of
the previous session) and one sampled transaction has no uid at all --
that one still got a real decision (score 0.4364 -> review), which is
correct: the ~11% no-uid population is the highest-risk slice
(uid_validation.md) and must still be scored and decided on, never
silently skipped.

policy.py has no import of investigator.py -- checked two ways in
tests/test_policy.py: statically (parse policy.py's AST, assert no import
mentions "investigator") and behaviorally (apply_policy on identical
scores with ANTHROPIC_API_KEY set vs. unset AND
sys.modules["src.investigator"] simulated unavailable -- identical
DataFrames both times). 8 new tests total, all passing; 47 in the full
suite.

No corrections needed this task -- the two threshold sweeps and the audit
script all ran cleanly on the first attempt, likely because the sweep
methodology was already proven out in Task 4 of the previous session.

## 2026-08-31 — Task 4: calibration -- Brier score looked fine, the region that matters didn't

scripts/calibration.py: reliability curve (quantile-binned, 15 bins --
equal-width bins would dump almost everything into one bin given the
~3.5% base rate) plus Brier score, appended to results/ablation.md. No
retraining, no recalibration -- read-only measurement, as instructed.

**Got wrong and corrected before it shipped:** first pass computed a
single equally-weighted "mean absolute gap across bins" (0.0111) and used
`mean_abs_gap < 0.02` as the sole calibration verdict, concluding
"reasonably well calibrated." Looking at the actual plot before writing
that up: 14 of the 15 bins sit at very low predicted scores (below ~0.09)
where the model is naturally well-calibrated (predicting near-zero for a
mostly-zero outcome is the easy part), and the single highest-score bin --
mean predicted 0.4795, which is above BOTH policy.py thresholds
(STEP_UP_THRESHOLD=0.0103, REVIEW_THRESHOLD=0.1843) -- has an observed
positive rate of only 0.3509, a 0.13 gap. An equally-weighted average
across bins buries that one bad bin under fourteen easy ones. Rewrote the
verdict to be driven by the highest-score bin specifically (the region
closest to where the policy actually operates) rather than the blanket
average, and reported both numbers with the reasoning for why the average
is the wrong one to trust here.

**Verdict, correctly this time: not well calibrated where it matters,
despite a low Brier score (0.0200 vs. 0.0332 for a base-rate-only
predictor).** REVIEW_THRESHOLD should be read as an arbitrary cut on the
score scale, not "we estimate >18.43% abuse risk" -- scores up there
systematically overstate true risk. Noted isotonic/Platt scaling as the
fix, explicitly not implemented (model.py frozen, and the task said not to
retrain).

This is exactly the kind of mistake the whole session is structured to
catch -- a single scalar metric that looks fine in aggregate while hiding
a real problem in the one region that actually matters for the thing being
built. Caught here by looking at the plot before trusting the number, not
by a test (there's no unit test for "did I pick a misleading metric").

## 2026-08-31 — Task 5: latency and scale benchmark

scripts/benchmark.py measures three things separately and appends a
Performance section to ARCHITECTURE.md (previously empty since the
original scaffold): graph construction (2.67s for 472,432 train-period
transactions at max_degree=20 -- consistent with earlier ad-hoc timings
from Task 1's investigation), cluster feature computation (12.45s for
167,111 uids), and per-transaction inline scoring latency over 1,000 real
sampled test transactions (feature-store lookup + single-row
model.predict): p50=32.8ms, p95=37.3ms, p99=39.5ms.

Stated the batch/inline split plainly, as asked: graph construction and
cluster features are batch (never on the request path); scoring is inline,
assuming cluster features are already sitting in a lookup (simulated here
as a plain dict keyed by uid, built once from pipeline_data.cluster_features
-- not a pandas .loc on the full frame, since that's not what a real
feature-store read would look like).

Worth noting honestly: 32-37ms for a single LightGBM prediction is slower
than "just call predict()" might suggest, because the benchmark includes
the realistic cost of assembling a correctly-ordered, correctly-typed
442-column row (432 baseline + 10 cluster columns, several categorical) on
every call, including an explicit column reindex to guarantee exact
alignment with what the model was trained on -- this is genuinely part of
the inline cost, not overhead added by the benchmark, but a real system
would likely keep a pre-allocated schema rather than reassembling it per
request, which this number doesn't capture.

~1B transactions/quarter is ~2117x this benchmark's train set. Named three
concrete changes that scale would require, none implemented (this is a
performance write-up, not new engineering): incremental graph updates
instead of full rebuilds, approximate connected components, and sharding
across machines (with the real problem there being cross-shard edges, not
just storage). Also connected this forward to Task 1 of the previous
session's finding: max_degree's effect on cluster size is a phase
transition, not a smooth curve, which makes "just re-tune it at higher
volume" a riskier proposition than it sounds.

No corrections needed -- ran cleanly on the first attempt, including the
column-reindex safeguard added proactively before running rather than
discovered by a crash.

## 2026-08-31 — Task 6: cluster case studies

scripts/case_studies.py generates investigator explanations for all 1,567
multi-uid clusters in the train graph, ranks by priority_score, and prints
raw detail (per-uid breakdown, shared identifier values, amounts, time
span, fraud labels) for the top 3. results/case_studies.md is the manual
write-up on top of that raw output -- deliberately not templated, since
"does this look like a real ring" is exactly the judgment call a script
shouldn't make for you (said explicitly in the script's own docstring).

All 3 top clusters turned out to be small (2 uids each) and 100% fraud-
labeled -- not cherry-picked, it's a direct consequence of
priority_score weighting cluster_prior_fraud_share at 100x: a small
cluster where every known transaction is fraud maxes that term out, so
small-and-fully-fraudulent clusters dominate this particular ranking.
Worth stating about the ranking itself, not just the 3 clusters it
surfaced.

Checked which specific linkage rule created each cluster's edge by reading
the actual `rules` edge attribute from the graph object (not inferred from
shared-value tables) -- this mattered:
- **Case 1** (`card_bank_addr`, an uncommon card3=223/card5=224 pairing --
  most transactions in this dataset cluster on card3=150/card5=226, so
  this specific combination matching is rarer and more specific than the
  hub-guard's raw uid-count threshold alone would suggest): plausible true
  positive.
- **Case 2** (`device_info`, a specific Samsung build string
  `SM-G950F Build/NRD90M` linking two otherwise-unconnected card+address
  pairs, transactions 4.7 minutes apart, identical $300 amounts, both
  fraud): the strongest case of the three.
- **Case 3** (`device_info`, but the shared value is `"en-gb"` -- reads as
  a locale/language string, not a device fingerprint): flagged explicitly
  as the weakest case, per the task's instruction to include a
  possible-false-positive and say so rather than swap it for a
  cleaner-looking example. The behavioral pattern (3.7 minutes apart,
  identical $150 amounts, both fraud) is genuinely suspicious and mirrors
  Case 2's shape, so it isn't confidently a false positive either --
  reported as ambiguous, with the specific reason the identifier is weaker
  spelled out (a value real unrelated en-GB-locale users could plausibly
  share, only below max_degree=20 by chance in this data slice, not
  because it's actually rare).

This is the first task this session where the deliverable is a judgment
call rather than a number -- worth being honest that "plausible true
positive" and "ambiguous" are reads, not measurements, and the report says
so in its own closing section rather than implying more precision than 3
manually-inspected clusters out of 1,567 can support.

## 2026-08-31 — Task 7 (part 1): wired everything into run_pipeline.py

Moved the artifact-generation logic from scripts/sanity_checks.py,
cost_curve.py, calibration.py, eval_investigator.py, write_audit_sample.py
and benchmark.py into src/run_pipeline.py as reusable functions
(write_sanity_checks, write_cost_curve, write_calibration,
write_investigator_eval, write_audit_sample, write_benchmark), all called
from one main() after a single load_and_prepare()/train_both_models() pair
-- previously each of the 7 artifacts independently re-loaded the 683MB
CSV and re-trained both models, ~90s of pure redundant setup per script.
Each scripts/*.py file is now a thin wrapper delegating to the
corresponding run_pipeline function, kept for standalone use (`python
scripts/cost_curve.py` still works).

**Verification that nothing's numbers changed in the move:** backed up the
committed results/ artifacts, ran the new consolidated `python -m
src.run_pipeline`, and diffed. results/ablation.md: byte-identical.
results/cost_curve.png, results/calibration.png: identical MD5. 
results/investigator_eval.md: one line differs, purely cosmetic wording
("Re-run this script" -> "Re-run this", since the text moved from a
script's own docstring into a shared function that isn't only reachable
from a script anymore) -- zero numbers changed. results/audit_sample.jsonl:
0 of 200 records differ once the timestamp field (which is supposed to
differ -- it's wall-clock time of the run) is excluded from the
comparison.

**Got wrong and corrected before committing:** results/ablation.md's
sections are safe to re-append repeatedly because write_ablation_report
always regenerates the whole file fresh first -- but ARCHITECTURE.md has
no equivalent "start fresh" step, and write_benchmark was a straight
unconditional append. Running the consolidated pipeline once revealed
`ARCHITECTURE.md` had gained a SECOND "## Performance" section (grep count:
2). This is exactly the kind of bug the reproducibility check in part 2 of
this task is supposed to catch -- `make results` needs to be safe to run
more than once, and it wasn't. Fixed with a `_remove_section()` helper that
strips a heading through to the next top-level heading before
re-appending, applied to both the ARCHITECTURE.md path and (defensively)
`_append_to_ablation`'s per-section appends. Verified the fix by running
the full pipeline twice in a row and confirming exactly one "## Performance"
section and one each of the three ablation.md subsections both times, with
identical model metrics both runs.

## 2026-08-31 — Task 7 (part 2): reproducibility verification

Cloned the uid-and-graph branch (`git clone --branch uid-and-graph
--single-branch`) into a fresh temp directory, created a brand-new venv
inside the clone, and `pip install -r requirements.txt` from scratch --
all packages installed cleanly at the exact pinned versions with no
resolver conflicts (same pins verified back in the very first scaffolding
session: pandas 3.0.5, numpy 2.5.0, scikit-learn 1.9.0, lightgbm 4.6.0,
networkx 3.6.1, matplotlib 3.11.1, shap 0.52.0, anthropic 0.120.2,
pytest 9.1.1).

**One step failed and is worth reporting plainly: `make results` does not
run on this machine, because GNU Make isn't installed at all** (`which
make` finds nothing on this Windows/Git-Bash setup). The README's
"Reproduce: `make results`" line assumes a Unix-like environment with make
available (a dev container, CI, macOS/Linux) -- on a bare Windows checkout
without WSL or make installed via a separate package manager, a reviewer
hits this immediately. Since the Makefile's `results` target is exactly
`python -m src.run_pipeline`, that command was run directly instead, which
is what actually verifies the pipeline -- but the `make` wrapper itself
was not exercised, and this is a real gap for a Windows-only reviewer, not
a hypothetical one.

data/ is (correctly) empty in a fresh clone -- CLAUDE.md's "do not commit
anything under data/" rule working as intended, but it also means a fresh
clone cannot run the pipeline until the two CSVs exist locally. Copied
train_transaction.csv and train_identity.csv from the already-downloaded
copy in the main working directory into the clone's data/ rather than
re-running scripts/download_data.sh against Kaggle again -- re-testing the
live Kaggle download wasn't the point of this check (it was already
exercised when the data was first fetched, see the entities.py session's
DEVLOG entry) and would have added a real external dependency (valid
credentials, competition rules accepted, network access) to a check that's
really about the code and environment being reproducible. Worth being
explicit that this means the download step itself was not re-verified here.

With data/ populated: `pytest tests/` -- **47 passed**, matching the
working directory exactly. `python -m src.run_pipeline` -- **completed
successfully**, and every number matched the working directory's committed
results exactly: results/ablation.md byte-identical, results/cost_curve.png
and results/calibration.png identical MD5 hashes, baseline/cluster
PR-AUC/recall/cost all identical to the last decimal printed
(0.5646104419855579 / 0.6322404738191845 for PR-AUC, matching bit for bit).

**Summary: the pipeline itself is fully reproducible from a fresh clone,
fresh venv, and fresh pip install, given the two source CSVs are present.
The one broken step is `make` not existing on this Windows environment** --
not a code bug, but a real onboarding gap between what the README promises
and what a plain Windows checkout can do without additional setup (installing
make, or just running `python -m src.run_pipeline` directly as documented
here). Not fixed this session (installing system-level `make` is outside
what a code change can do, and rewriting the Makefile as something
Windows-native wasn't asked for) -- reported plainly instead, per this
session's own instructions.

## 2026-08-31 — A silent failure hid an entire unmeasured layer, and my own eval reported it as a success

This is the most important bug this project has produced, and it wasn't
caught by any test, any sanity check, or any of the "measure it, don't
just ship it" discipline this session has otherwise followed. It was
caught by the user reading results/investigator_eval.md and noticing that
"ANTHROPIC_API_KEY was set" and "30/30 explanations were
source=ungrounded-fallback" were both true on the same line, and asking
why.

**Root cause, as diagnosed and fixed by the user, not by this session's
own investigation:** the Anthropic API key in use is identity-linked (a
workspace-scoped key), which the API rejects on every single call unless
the request carries an `anthropic-workspace-id` header. investigator.py
never sent that header. Separately, the workspace's balance was zero.
Both are now fixed on the account side; the header support is fixed here.

**Why this is a real bug class, not just a one-off missing header:** this
project's own design -- correctly -- required investigator.py to degrade
gracefully: no API key, a network blip, a rate limit, anything, and
`make results` must still complete rather than crash. That's the right
call, and it's still the right call after this bug. The actual defect was
one specific line: `except Exception: return <fallback>`, with nothing
recording *why*. A graceful degradation path and a silent failure path are
the same code until you ask "does anything downstream know the difference
between 'degraded on purpose' and 'broke and I didn't notice'?" Here,
nothing did. Every one of 30 calls failed with the exact same
authentication error, for an entire session's worth of work (Task 1's
implementation and Task 2's "measurement"), and the code had no way to
surface that -- not a log line, not a counter, not a flag. The fallback
path being *correct* is precisely what made the failure invisible: a
crash would have been noticed in the first five seconds.

**The compounding failure, which is the part worth sitting with:** Task 2
of the earlier session was explicitly framed as "measure the investigator,
don't just ship it" -- and it did report 100% groundedness with a caveat
that the key was unset. That caveat was correct *then*. But this most
recent run had a key set, and the report-writing code
(write_investigator_eval, then in scripts/eval_investigator.py) contained
this: the "at least one explanation used the real LLM path" sentence was
selected by `if not has_key: ... else: "at least one explanation used the
real LLM path"` -- driven by whether a key was *present*, not by whether
`sources` actually contained anything other than `ungrounded-fallback`.
The measurement layer -- the thing built specifically to catch exactly
this kind of problem -- inherited the same category of bug as the thing it
was measuring: it asserted an outcome from a precondition, not from the
actual result. A precondition ("a key was set") is necessary but not
sufficient for success, and reporting code must never conflate the two.
This is the generalizable lesson: any report about whether a fallback path
was needed must be computed from the actual recorded outcome of every
attempt, never inferred from whether success was theoretically possible.

**What was fixed, concretely:**
- `investigator.py`: `_call_anthropic` now reads `ANTHROPIC_WORKSPACE_ID`
  from the environment and passes it as `default_headers=
  {"anthropic-workspace-id": ...}` when set, omitted entirely when unset
  (a non-identity-linked key keeps working unchanged -- verified with a
  test that inspects the actual kwargs a mocked `anthropic.Anthropic` was
  constructed with, for both the header-present and header-absent cases).
- `ClusterExplanation` gained an `error: str | None` field: `None` on a
  real LLM success, `"ANTHROPIC_API_KEY not set"` or `"{ExceptionType}:
  {message}"` on every fallback. `explain_cluster` now also prints
  `[investigator] {cluster_id}: ...` to stderr on every fallback, so a
  `make results` run has a visible trail even if no one reads the written
  report afterward. The fallback *behavior* is unchanged -- still never
  raises, still completes -- only its visibility changed.
- `run_pipeline.write_investigator_eval` rewritten so every claim in
  results/investigator_eval.md is derived from `sources`/`error`, never
  from `has_key`: it now says "0 of 30 explanations used the real LLM
  path" when that's true regardless of whether a key was present, and adds
  a "Fallback errors encountered" section listing every distinct error
  string and how many clusters hit it. Groundedness is now reported
  separately for LLM-sourced vs. fallback-sourced narratives (a mix would
  otherwise dilute or inflate the number that actually matters), with an
  explicit "there is nothing here that measures claude-sonnet-4-6's actual
  behavior" line when n_llm=0, and a defensive check that flags (rather
  than silently trusts) an ungrounded claim ever showing up in a fallback
  narrative, since that would itself be a bug given the template is
  grounded by construction.
- 7 new tests: `error` populated correctly for both fallback causes and
  left `None` on success, the stderr log line contains the cluster id and
  the real exception type/message, and the workspace header is present
  only when `ANTHROPIC_WORKSPACE_ID` is set. 54 tests passing overall.
- README.md documents `ANTHROPIC_WORKSPACE_ID` under a new "Running it ->
  Environment variables" section, including the specific symptom (looks
  like it's working, 100% fallback) so a future reader recognizes this
  failure mode immediately instead of re-discovering it the way this
  session did.

**What this entry cannot honestly claim:** I do not have
`ANTHROPIC_API_KEY` or `ANTHROPIC_WORKSPACE_ID` available in my own tool
execution environment -- checked Bash, PowerShell, persisted Windows
user/machine environment variables, and for a `.env` file; none were
found. Re-running `scripts/eval_investigator.py` here still produces 0 of
30 real LLM explanations, now *correctly* reported as
`ANTHROPIC_API_KEY not set` rather than incorrectly implying success. The
user has verified a direct API call succeeds in their own environment, but
I have not personally produced or witnessed a real groundedness
measurement against actual LLM output, and I am not reporting one. That
measurement still needs to happen in an environment where the credentials
are actually reachable -- this entry fixes the mechanism for measuring it
honestly; it does not itself contain that measurement.

## 2026-08-31 — The real measurement: 100% groundedness on actual LLM output

The user ran `scripts/eval_investigator.py` themselves, in an environment
where `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` are both genuinely
set and working, and shared the resulting results/investigator_eval.md.
This is the measurement the previous two entries explicitly said had not
happened yet -- recorded here as soon as it existed, not reconstructed or
approximated.

**Result: 30 of 30 explanations used the real LLM path (source=llm,
confirmed from the actual `sources` counter, not inferred). 0 of 182
extracted numeric claims were ungrounded -- groundedness rate 100.00%.**
Zero fallback explanations this run, so unlike every previous run, there's
no fallback-vs-LLM split to worry about diluting the number: all 182
claims are real claude-sonnet-4-6 output.

Spot-checked the three example narratives by hand against their evidence
dicts rather than just trusting the automated checker's own arithmetic:
cluster-13 cites edge density 1.0, email-uid ratio 0.1429, 1 distinct email
domain, 2 distinct product codes, amt_cv 1.0557, prior_fraud_share 0.0,
burst_concentration 0.0909 -- every one matches its evidence dict value
exactly. cluster-29 cites prior_fraud_share 1.0, edge_density 1.0, 2
product codes, 2 email domains, email_uid_ratio 0.6667 -- same, all exact
matches. The narratives also read as genuinely reasoned rather than just
regurgitating the evidence dict in prose: cluster-13's read ("structural
linkage and email concentration are the most concerning signals... though
the absence of prior fraud history and low burst activity leave the risk
assessment somewhat ambiguous") explicitly weighs signals against each
other rather than asserting a flat verdict, which is exactly the kind of
grounded-but-not-mechanical behavior the system prompt's hard rule was
meant to allow room for.

**What this does and doesn't establish.** It closes the loop the last two
entries left open: the investigator layer has now actually been exercised
against the real model, not just unit-tested against mocks, and the
specific hard-number-grounding constraint this session was built around
held on every single claim it produced. It does not establish that
claude-sonnet-4-6 will always stay grounded -- this is one run, 30
clusters, 182 claims, not a guarantee, and the honest posture from here is
"measured clean on the one real run we have," not "proven safe." If this
is re-run at higher volume or on a different cluster selection, the
groundedness check should keep running every time, not be treated as a
box now permanently checked.

This also means the bug fixed two entries ago is now confirmed fixed
end-to-end, not just plausible in theory: the workspace-id header was the
actual blocker, and with it in place the same code path that produced 30
silent failures now produces 30 real, grounded explanations.

## 2026-09-01 — Task 1+2: Streamlit dashboard (app.py) + dark theme

Committed together rather than split: the CSS is embedded directly in
app.py (there's no separate "logic" version that ever existed without
styling), so artificially splitting them into two commits would mean
reconstructing a fake unstyled intermediate just for git history's sake,
which helps no one. Both tasks are genuinely done, documented together
honestly.

Three tabs: Review queue (cluster list ranked by investigator.py's
priority_score, selectable, with a detail panel showing the LLM narrative,
evidence table, member uids, a transaction timeline, and the policy
decision + threshold), Model performance (ablation table, cost curve,
calibration -- all parsed from results/ablation.md at runtime), Audit
trail (results/audit_sample.jsonl, filterable by action). Dark theme via
.streamlit/config.toml (near-black #0D0D0F base) plus custom CSS injected
via st.markdown for anything the theme config can't reach (fonts,
monospace numerics, risk badges).

**Real problems found and fixed while building this, in the order they
were hit:**

1. **Queue-building took minutes, not seconds.** First version looped
   `investigator.build_evidence` + `_priority_score` + `policy.decide`
   over every one of the ~1,500+ multi-uid clusters with test-period
   activity, each iteration doing a `.loc` fancy-index lookup against a
   ~199k-row cluster_features table. Caught by literally watching it hang
   for minutes with the server still alive and the log not moving.
   Rewrote as two passes: a fully vectorized pass (groupby `.count()`,
   `.map()` against a Series -- no per-cluster Python callback) to find a
   bounded candidate pool of 400 by a cheap proxy
   (`cluster_prior_fraud_share`, the dominant term in the real priority
   formula anyway), then the exact, non-vectorizable investigator/policy
   calls only on that bounded pool. Seconds instead of minutes, same
   numbers for every cluster that ends up in the displayed queue.

2. **A visibly empty bordered box between the policy-decision line and
   the LLM narrative.** Root cause: `st.markdown('<div class="panel">')`
   opened in one call, with the label/content/closing `</div>` issued in
   *separate* `st.markdown` calls after it. Each `st.markdown` call is its
   own independent HTML fragment in Streamlit -- an unclosed `<div>` in
   one call does not wrap content from a later call, so the "open" div
   rendered as its own empty, styled, bordered rectangle, and the actual
   content became unrelated sibling elements that happened to look
   plausible next to it. Fixed by building the whole panel's HTML as one
   f-string and issuing it in a single `st.markdown` call. Checked the
   rest of the file for the same pattern (grep for every `st.markdown`
   call with raw HTML) -- this was the only instance.

3. **A hardcoded number, caught by looking at my own screenshot.** The
   Model performance tab's caption said "roughly 84% of the headline
   PR-AUC lift (+0.0676 full vs. +0.0110 without it)" as a literal string
   -- correct today, but exactly the kind of thing this session's own
   instruction ("never invent a number... every figure must be read from
   results/ at runtime") was written to prevent, since it would go
   silently stale if the underlying numbers ever changed. Fixed to compute
   the share and both deltas from the already-parsed ablation table at
   render time.

4. **Streamlit's rerun-vs-runOnSave behavior isn't what I assumed.**
   While testing, I edited app.py *while a capture run was already
   executing* against the previous version, expecting the running
   subprocess to keep using the old in-memory code (`runOnSave=false` in
   config.toml). It didn't: a widget-triggered rerun (clicking a queue
   row) picked up the edited file mid-run, producing a screenshot with
   the new styling despite being launched before the edit. Streamlit
   re-reads the script from disk on every rerun regardless of
   `runOnSave` -- that setting only controls whether the file watcher
   *automatically* triggers a rerun without user interaction (showing a
   "File change / Rerun" banner instead). Didn't affect correctness here,
   but worth knowing: don't trust that a long-running dev-server subprocess
   is still running the code you started it with.

Chose a hand-rolled HTML table (`render_html_table`) over `st.dataframe`
for the evidence/ablation/audit tables specifically to get monospace,
right-aligned, consistent-decimal numerics -- `st.dataframe`'s grid
(glide-data-grid) renders to canvas and can't be reached by CSS at the
per-cell level. `st.dataframe` is used only for the one place true
row-selection interactivity is required (the queue), with a pandas Styler
for the risk-colored action column (`.style.map`, confirmed it renders
correctly rather than assumed).

Verified live with a real running server + Playwright screenshots at each
step (not just "it compiles") -- all three tabs, row selection, the
narrative/policy attribution, the action-column coloring, and the
error/empty states were each checked against an actual screenshot before
moving on, per this session's own instruction to test UI changes in a
browser rather than claim success from type-checking alone.

## 2026-09-01 — Task 3: screenshots

scripts/capture_screenshots.py launches `streamlit run app.py` as a
subprocess on a scratch port, waits for the one-time pipeline load, and
uses Playwright (headless Chromium) to capture the review queue, a
cluster detail view (first queue row selected), and the model performance
tab into docs/, then tears the subprocess down. Not wired into `make
results` or requirements.txt -- this is a one-off authoring tool for the
README, not something the pipeline or dashboard needs at runtime;
playwright/chromium were installed ad hoc for this and for live-testing
app.py during Task 1+2, documented in the script's own docstring rather
than silently added as a project dependency.

Screenshots are real: same server, same real data, same real LLM/fallback
behavior as every other verification this session -- not mocked or
hand-assembled.

No corrections needed for the script itself -- it worked on the first
run (the styling fix from Task 1+2 was still in flight when this was
first written, so the first capture predated it slightly; a second run
after that edit landed produced the final, correct screenshots, confirmed
by inspecting all three images before committing).

## 2026-09-01 — Task 4: README rewrite -- results first

Restructured per the session's spec: one-sentence description, results
table with the 84%-lift disclosure stated before the reader gets to
celebrate the headline number, both reproduce commands (`python -m
src.run_pipeline` and `make results`, with the Windows caveat about make
not being present stated plainly rather than assumed away), setup
(venv, data download, both Anthropic env vars with the workspace-id
failure mode named explicitly so a future reader recognizes the symptom
instead of re-discovering it), an architecture summary that states the
ML/LLM/policy separation as a structural fact with its own verification
(AST check + behavioral test), screenshots, and all seven limitations
the task asked for expanded (label noise's interaction with the dominant
feature, uid over-merging, the ~11% unresolvable rows and their higher
fraud rate, max_degree=20 and what it excludes, calibration
overconfidence of 0.13, no ground truth, and the groundedness result
being one run of 30 clusters, not a permanent guarantee).

Every number in it was copied from an already-verified results/ artifact
or DEVLOG entry, not recomputed or re-derived for this rewrite -- nothing
new to check here beyond making sure the transcription matched the
source exactly, which it does (diffed by eye against results/ablation.md,
results/investigator_eval.md, results/uid_validation.md, and
results/d1_investigation.md while writing).

## 2026-09-01 — Task 5: ARCHITECTURE.md completion

Wrote the pipeline overview, the batch/inline split (with the actual
reasoning for why each stage is on the side it's on, not just a label),
the linkage rules table with their rationale (pulled verbatim from
graph.py's own LinkageRule dataclasses, not re-explained from memory),
the causal feature computation section (what CLAUDE.md's causality rule
means concretely and how it was verified against real data, not just
argued from the code), and the ML/LLM/policy boundary (the same
structural separation documented in policy.py's own docstring, restated
here with the dashboard's UI-level enforcement added since that's new
this session).

**Caught and fixed a real duplication problem before committing, not
after:** first draft included a "## Performance numbers" section with
hand-typed approximate figures (graph build ~2.5s, cluster features ~13s,
scoring p50 ~32ms) pulled from earlier terminal output. Two problems:
(1) `run_pipeline.write_benchmark` already generates a "## Performance"
section with exact, fresh numbers on every `make results` run and
appends it to this exact file -- my hand-typed version was duplicating
that, with rounder, staler numbers standing next to the real ones after
the next pipeline run. (2) Worse: I'd named my section exactly
"## Performance" (matching write_benchmark's `_remove_section` target),
which meant the next `make results` run would have silently deleted my
own paragraph along with the numbers it was replacing, not just left a
harmless duplicate. Verified this precisely rather than guessing, by
calling `_remove_section` directly against the file's actual content and
confirming it returned the text unchanged after the fix. Fixed by
removing every hand-copied benchmark figure and renaming the section
"## Performance notes" -- a short frame for what's below, not a second
source of truth for numbers that are supposed to come from a live
benchmark. Applied the same fix to a smaller duplicate in the batch/inline
table's summary line.

The max_degree sweep figures (126 uids at 20, the phase transition at
30->35, 64% at 1000) are quoted directly from graph.py's own module
docstring, not re-derived -- that file is frozen and the sweep was a
one-time empirical finding baked into it, which is a meaningfully
different kind of number from a live benchmark timing: it won't drift
between pipeline runs the way wall-clock latency will, so quoting it here
isn't the same category of risk the Performance-section fix was about.

## 2026-09-01 — Task 6: final verification

**Fresh clone, fresh venv, fresh install.** Cloned uid-and-graph into a new
temp directory, created a brand-new venv, `pip install -r
requirements.txt` from scratch (now including streamlit and its
dependency tree from Task 1) -- all packages installed cleanly at the
exact pinned versions, no resolver conflicts. Copied the already-downloaded
CSVs into the clone's data/ rather than re-running the Kaggle download,
same reasoning as every previous reproducibility check this project has
done: re-testing the live download isn't the point of this check, and it
adds an external dependency (credentials, network) to something that's
really about code/environment reproducibility.

**Pipeline: fully reproducible.** `pytest` -- 54 passed, matching exactly.
`python -m src.run_pipeline` -- completed successfully. Every number
matched the working directory's committed results exactly:
results/ablation.md byte-identical, results/cost_curve.png and
results/calibration.png identical MD5 hashes, baseline/cluster PR-AUC
matching to the last printed decimal (0.5646104419855579 /
0.6322404738191845). results/audit_sample.jsonl: 0 of 200 records differ
once the (expectedly different) timestamp is excluded.
results/investigator_eval.md legitimately differs -- the clone has no
`ANTHROPIC_API_KEY`, so it correctly reports 0 of 30 explanations using
the real LLM path, honestly labeled as such, not a bug.

**Dashboard: renders from the clone with the original working directory
absent.** Launched `streamlit run app.py` from inside the clone directory
on a separate port, confirmed via Playwright screenshots that the review
queue (same clusters, same priorities, same colors), and the audit trail
tab (same 200 records) render identically to the working directory's own
dashboard. Nothing in app.py references the original repo path -- it only
ever uses `Path(__file__).resolve().parent`, which is exactly why this
worked.

**Git hygiene: clean.** `git ls-files data/` shows only `.gitkeep`; no
`.csv`/`.pkl` tracked anywhere; no `.env` tracked; no filename matching
credential/secret/token/key patterns; a repo-wide grep for an
Anthropic-API-key-shaped string (`sk-ant-...`) found nothing tracked.
`.streamlit/` contains only `config.toml` (no `secrets.toml`).

**What failed, reported plainly, same as the last time this exact check
was run:** `make` is still not installed on this Windows/Git-Bash
environment. `make results` as literally written in the README does not
run here -- the underlying command it wraps
(`python -m src.run_pipeline`) does, and that's what both this check and
the README's Reproduce section actually rely on. Not a regression and not
fixed this session (installing system `make` isn't a code change); the
README states this plainly rather than assuming `make` is universally
available, per Task 4.

## 2026-09-01 -- Task 7: queue-level precision@k, and a real null result

**What was built.** `scripts/eval_queue.py` -> `results/queue_eval.md`.
Every reported number up to now (results/ablation.md) is transaction-level
PR-AUC -- a proxy for "does the model rank individual transactions well,"
not "would an analyst working a queue of flagged *clusters* actually find
real abuse near the top." This closes that gap: on the test split only
(no retraining, no feature recomputation -- reuses
`run_pipeline.load_and_prepare()`/`train_both_models()` exactly like every
other results/*.md artifact), restricted to multi-uid clusters (2+
members, the same population case_studies.md and app.py use) with at
least one test-period transaction, for K in [10, 25, 50, 100]:
precision@k, fraud transactions surfaced, workload (test transactions an
analyst would review), and the resulting efficiency ratio -- computed for
two rankings. Ranking A is the actual system ordering
(`investigator.build_evidence` + `investigator._priority_score`, the exact
same call app.py's `build_cluster_queue` makes, over each cluster's full
train+test history). Ranking B is a deliberately dumb baseline: clusters
ranked by the mean of the *baseline* model's (txn features only, zero
cluster signal anywhere in training) per-transaction score across a
cluster's test-period members -- no cluster topology, no
`cluster_prior_fraud_share`, nothing graph-derived at all.

**What surprised me.** The baseline beat the priority ranking at 3 of the
4 K values (K=100 tied), not narrowly: precision@10 is 0.40 for the
baseline vs. 0.20 for priority; precision@50 is 0.10 vs. 0.06. I expected
the system's own ranking to at least match a naive baseline, since it's
built from features that measurably lift transaction-level PR-AUC
(results/ablation.md). It doesn't, and thinking about *why* is more
useful than the number itself: `_priority_score` weights
`cluster_prior_fraud_share` 100x -- "has this cluster's history ever
included fraud" -- which is a backward-looking, largely static signal per
cluster. A cluster whose members were flagged long ago and haven't done
anything since sits at the top of the priority queue indefinitely. The
baseline's mean transaction score has no memory like that -- it only
reacts to what a cluster's members are doing *right now*, in the test
window, which is exactly the thing precision@k on *test-period* fraud is
measuring. The same feature that drives 84% of the PR-AUC lift
(README.md's "read the third row before the first one" caveat) is the
one dragging the cluster-priority ordering down here. Same feature, two
different jobs, and it's good at one and not the other.

**What the numbers actually say.** Base rate: only 9 of 494 qualifying
clusters (1.8%) contain any test-period fraud at all -- both rankings are
working against a genuinely sparse target, and every precision@k number
has to be read against that 1.8%, not in isolation (both easily clear it,
for what that's worth). Reported the loss as-is in both
results/queue_eval.md and the new README section -- no reweighting of
`_priority_score`, no swapping which model backs the baseline, no
after-the-fact justification softened into the verdict text. This is a
legitimate finding about where the cluster-graph apparatus does and
doesn't help: it was built and validated against transaction-level
PR-AUC, and that lift does not transfer to being a better cluster-queue
ordering than just averaging a plain transaction classifier's scores.
`policy.py`/`evaluate.py` and every existing results/*.md number are
untouched -- this is a new, additive measurement, not a revision of the
submission's headline.

## 2026-09-01 -- Task 1: queue_eval.md was technically correct and still misleading

**What was corrected.** The previous entry's report (results/queue_eval.md)
stated a true fact -- the baseline beats the priority ranking at 3 of 4
K values -- in a way that asserted more confidence than 9 positive
clusters can support, and it buried the actually-good news. Fixed in
`scripts/eval_queue.py` (not hand-edited into the .md -- the report is
still generated, never hand-typed, same discipline as every other
results/*.md file here): every precision@k figure now carries its
absolute count (e.g. "0.4000 (4/10)"), its lift over the 1.8% base rate
(e.g. "22.0x"), and a parallel recall@k column (of the 9 fraud-containing
clusters, how many appear in the top K). The Verdict section now states a
noise caveat explicitly -- a gap of 3 clusters or fewer, which is true at
all 4 K values tested here, is called out as indistinguishable from
chance with only 9 positives -- before restating the null finding, more
precisely worded: the priority score is behind or tied at 4 of 4 K
values, *and* the sample can't confirm that's a real gap. Neither half
alone was the honest statement.

**What changed, concretely.** Precision@10 (0.40 baseline / 0.20
priority) now reads as 22x / 11x base rate, respectively, and the report
now says plainly that both rankings clearing an order of magnitude over
random is the headline finding, not something to skip past on the way to
the A-vs-B comparison. That comparison hadn't changed -- the underlying
numbers are identical to Task 7's run -- only how honestly the report
described what those numbers could and couldn't support.

**What surprised me.** How easy it was to write a "reported honestly"
paragraph last session that was still one-sided: I didn't invent
anything or hide the loss, but by leading with "the baseline beats the
priority ranking" without immediately pairing it with "and that gap is 2
clusters out of 9, which is noise," the framing did the reader's
statistical reasoning for them, in the wrong direction. Getting a null
result reported honestly turned out to require more than not lying about
the direction of the result -- it required stating the uncertainty
around it in the same sentence, not a paragraph later. Fixed the
`_results_table` header to derive the "9" from `total_positives` at
runtime rather than a literal I'd typed while drafting the wording,
before it could become the next hardcoded-number bug this project has
already hit twice.

## 2026-09-01 -- Task 2: five orderings, none of them confidently better

**What was built.** `scripts/eval_priority_variants.py` ->
results/priority_variants.md. `_priority_score`'s 100x weight on
`cluster_prior_fraud_share` was picked by hand, so this evaluates it
against four alternatives on the identical 494-cluster/9-positive
population as results/queue_eval.md (imported from scripts/eval_queue.py,
not recomputed): (a) the shipped priority score, (b) mean transaction
score (queue_eval.md's baseline), (c) priority score with
`cluster_prior_fraud_share` deleted from the evidence dict before calling
`investigator._priority_score` -- not a reimplementation of the formula,
the same frozen function with one term zeroed by omission, (d) max
(instead of mean) transaction score per cluster, and (e) mean score
multiplied by the cluster's full component size.

**What the numbers say.** No variant wins outright. The best precision@k
at each K is a different variant every time: (d) at K=10 and K=50, (b) at
K=25, and a four-way tie including (a) itself at K=100. (c) -- dropping
the 100x term entirely -- is not obviously better OR worse than (a); at
K=50 it's tied with (b), at K=100 it's the worst of the five. The spread
between the best and worst variant's fraud-cluster count at any given K
is 1-4 clusters, computed directly from each table's counts (not
guessed) -- the same magnitude as the noise callout in queue_eval.md's
Task 1 fix.

**What surprised me.** I expected (d), max-in-cluster, to be a clear
improvement over (b), mean-in-cluster -- the intuition being that a
single very suspicious transaction should out-signal an average diluted
by a cluster's other, unremarkable members. It IS ahead at K=10 and
K=50, but ties or loses at K=25 and K=100, and the swings are exactly the
size the 9-positive sample can produce by chance. I did not expect
"obviously more sensible aggregation choice" to still land inside the
noise floor -- it's a reminder that with this few positives, even a
directionally well-motivated variant can't be told apart from the others
on this test split alone.

**What this means for the project.** Reported as a menu of untested
candidates for future work, explicitly not adopted: `investigator.py` and
`policy.py` are unmodified, and results/priority_variants.md says plainly
that recommending any one of these would need a less severe temporal
split or a longer test window to get enough positive clusters to
distinguish them. The honest conclusion of Tasks 1 and 2 together: this
project's queue-ordering evaluation is fraud-count-limited, not
idea-limited -- there are several plausible orderings, and not enough
positives on this test split to say which, if any, is actually better.

## 2026-09-01 -- Task 3: the headline lift swings a lot, and the residual lift isn't even reliably positive

**What was built.** `scripts/eval_stability.py` -> results/stability.md.
results/ablation.md's +0.0676 PR-AUC lift comes from exactly one temporal
boundary (last 20% held out). This re-runs the entire ablation -- fresh
entity graph, fresh causal cluster features, fresh baseline/cluster/
trimmed models, nothing cached -- at 4 rolling split points (60/70/80/90%
through sorted TransactionDT), importing only entities.py/graph.py/
model.py/evaluate.py (all frozen, none modified) plus
run_pipeline.broadcast_cluster_features (not frozen, reused as-is). Took
408s for all 4 splits; the time-budget fallback (drop to 3 splits) wasn't
needed.

**Consistency check first, since it's the cheapest way to catch a bug:**
the 80% split point in this script reproduced results/ablation.md's exact
numbers -- baseline 0.5646, cluster 0.6322, lift +0.0676, trimmed
0.5756 -- to 4 decimal places, both times this script was run. That's the
same train/test boundary computed independently by a different script, so
this is real evidence the split/feature/training logic here matches
run_pipeline.py's, not just an assumption.

**What the numbers say.** The full cluster-feature lift swings from
+0.0284 (60% split) to +0.0676 (80% split) -- mean +0.0516, spread 0.0393,
which is most of the mean itself. results/ablation.md's headline number
is one sample from a noisy distribution, not a fixed property of the
feature set, and this report says so plainly rather than treating +0.0676
as *the* number.

**What surprised me, and it's the more important finding of the two:**
the *residual* lift -- PR-AUC gain from the structural/graph features
alone, with `cluster_prior_fraud_share` removed -- isn't just noisy, it
changes sign: -0.0044 at 60%, +0.0186 at 70%, +0.0110 at 80%, -0.0013 at
90% (mean +0.0060, spread 0.0230 -- larger than the mean in absolute
terms). README.md currently calls this residual "a smaller but genuine
residual lift on their own," a characterization that rests entirely on
the single 80% split. Across 4 splits it is not reliably positive at all
-- it looks like noise centered near zero, not a small-but-real
structural signal. results/stability.md states this plainly and flags it
against README's existing wording, but does not edit README's Result/
Limitations prose itself -- that's an editorial call about the project's
main narrative, out of scope for a script whose deliverable is the
stability report, not a rewrite of the headline story.

**Why this matters for the project as a whole.** Between this and Tasks
1-2, the pattern across this whole session is the same: every number this
project has treated as a stable property of the model turns out to be
more fragile once measured more than once -- the cluster-queue priority
score didn't beat a naive baseline (Task 1), no simple alternative
ordering confidently beat it either (Task 2), and now the "genuine"
non-prior-fraud lift doesn't reliably survive a different temporal
boundary (Task 3). None of these are contradictions or bugs -- they're
what you find when you go looking for a second measurement instead of
resting on the first one.

## 2026-09-01 -- Task 4: mapping to production, without pretending to know Razorpay's internals

**What was built.** A "Mapping to production" section in README.md, two
paragraphs, no code. First: what IEEE-CIS's columns are analogous to in a
real Indian payments stack -- `card1` to a tokenized instrument,
`DeviceInfo` to a client-SDK device fingerprint, `addr1` to a pincode/
region code, `P_emaildomain` to a customer contact identity -- and what
this dataset has no equivalent for: a VPA handle, the UPI PSP a
transaction routed through, a COD flag, merchant category, a shipping
address distinct from billing. Second: what changes under UPI --
there's no chargeback mechanism, so the label source this project trains
on (chargeback-reported, card-propagating) has no direct analog; the
nearest substitutes (merchant-reported fraud, RTO/return-abuse signals)
come from a different actor on a different timeline, which changes both
label latency and label noise.

**What I was careful about.** The section opens by stating plainly that
this mapping is illustrative of a generic Indian payments stack, not a
description of Razorpay's actual systems -- nothing here claims internal
knowledge this project doesn't have. No code, no new claims about the
pipeline's own numbers, and nothing in src/ touched.

## 2026-09-01 -- Task 5: measuring stability invalidated a claim I had already published in the README

**What happened.** Task 3's stability check (results/stability.md) found
that the residual lift -- PR-AUC gained from the structural/graph
features alone, with `cluster_prior_fraud_share` removed -- changes sign
across the 4 rolling splits (mean +0.0060, spread 0.0230: negative at
60%/90%, positive at 70%/80%). I flagged that finding plainly in
results/stability.md's own text when I wrote it, and then, in the same
session, left README.md's Result section still asserting "the remaining
structural/graph features ... contribute a smaller but genuine residual
lift on their own" -- a sentence I had written earlier this project,
based on exactly one split, that my own later measurement had already
shown didn't hold. I didn't go back and fix it until asked to. That gap
-- computing the correction and then not immediately propagating it to
the document making the claim -- is the mistake worth recording here, not
the original single-split claim itself (which was reasonable given what
was known when it was written).

**What was corrected, precisely.** README.md's Result section and
results/ablation.md's Results section (both, per the request -- this
claim was published in two places, not one) now state, every time the
+0.0676 single-split lift is presented: the full lift's cross-split mean
(+0.0516) and spread (0.0393, directionally consistent but a large swing
relative to the mean), and the residual lift's cross-split mean (+0.0060)
and spread (0.0230) *with the sign flip stated explicitly, not
paraphrased into "noisy."* The "smaller but genuine residual lift"
sentence is gone; it's replaced with "does not show a reliable lift on
this dataset at this sample size." The +0.0676 and 0.5756 single-split
figures themselves are untouched -- they're real, reproducible numbers
from a real run, not the thing that was wrong. What was wrong was
presenting them without the range that shows how much they move.

**Limitations section addition.** Added a plain opening statement to the
existing "Label noise, and its interaction with the dominant feature"
bullet: `cluster_prior_fraud_share` is backward-looking fraud history,
powerful but partly circular in a chargeback-labelled dataset, since it
partly measures label propagation across a card rather than
independently discovering abuse. The bullet already said something like
this in more hedged language ("close to a direct measurement," "partly
learning to reproduce the label-generation process"); this adds the
direct version on top, plus a new closing sentence connecting it to the
stability finding -- since the other cluster features don't show a
reliable lift, this one backward-looking, partly-circular feature is now
carrying essentially all of the reliably-measured lift, not just the
largest share of it. That's a stronger and more precise claim than the
bullet made before this session.

**No pipeline files touched, nothing re-run.** Per the explicit
instruction: the correction is the finding, not a reason to go looking
for a better split. results/ablation.md and README.md were edited
directly with already-computed, already-published numbers from
results/stability.md -- the same pattern this project already uses
elsewhere for prose that cites another report's figures (e.g. README's
uid-validation numbers), not a new "hand-invented number" exception.

## 2026-09-01 -- Task 6: renamed abuse-ring-sentinel to not-strangers

**What was renamed.** The dashboard header (`st.title`), the browser-tab
title (`st.set_page_config(page_title=...)`), app.py's own module
docstring title, and README.md's H1 all said "abuse-ring-sentinel" --
changed to "not-strangers" to match the actual repo name. The dashboard
subtitle line (`st.caption(...)`) was left untouched, per instruction.

**What was deliberately NOT touched, and why.** Grepped the whole repo
case-insensitively for "sentinel" first, rather than assuming where it
appeared. Two categories of match were left alone: (1) app.py's
`sentinel-table` CSS class name (6 occurrences) -- an internal styling
hook, never rendered as visible text, so it isn't a "user-facing"
occurrence of the name; renaming it would touch more lines for no visible
change. (2) `tests/test_entities.py`'s "stringified sentinel" and
`results/investigator_eval.md`'s "credit abuse rings" (inside a captured,
real LLM narrative) -- both use "sentinel"/"abuse ring" as the ordinary
English/CS terms they are, unrelated to the project's old brand name;
"abuse ring" is this project's own domain vocabulary (see CLAUDE.md's
first line), not something to rename. ARCHITECTURE.md, results/*.md, and
a repo-root pyproject.toml/setup.py/package.json were all checked and
none exist or reference the old name.

**Screenshots.** Re-ran scripts/capture_screenshots.py (Playwright +
headless Chromium against a temporary `streamlit run app.py`) to
regenerate all 3 docs/ PNGs. Verified the queue screenshot directly --
header now reads "not-strangers", subtitle unchanged, rest of the queue
UI identical. No pipeline files touched; 54/54 tests still pass.

## 2026-09-01 -- README restructure: same content, four minutes not fifteen

**What changed.** README.md was accurate but front-loaded with density --
the Result section alone ran to 4 paragraphs before a reader ever saw the
tool. Restructured to the requested shape: Result now ends after the
table plus exactly 4 sentences (headline lift, the 84%/cluster_prior_
fraud_share fact, the cross-split holds/doesn't-hold correction, one link
line); Screenshots moved up to right after Result, before any caveats;
Queue-level evaluation cut to 3 sentences; Reproduce/Setup compressed to
commands with one line of context each; Architecture's prose cut to 3
sentences on the ML/LLM/policy separation; all 7 Limitations bullets
became a bold one-line claim plus at most 2 sentences; Mapping to
production cut to 1 paragraph. New Troubleshooting section at the bottom
holds the identity-linked-key setup detail that used to live in Setup.
967 words end to end now (was denser and harder to scan, not shorter in
a way that lost anything -- see below).

**Nothing was deleted, only relocated -- verified, not assumed.** Every
paragraph cut from README landed in a results/ file: the two long
Result-section paragraphs (the 84% trace, the single-split-vs-cross-
split correction) and the "dominant feature is backward-looking, partly
circular" limitations bullet moved verbatim into a new "## Detailed
findings" section in results/ablation.md; the uid-over-merges bullet
into results/d1_investigation.md; the ~11%-no-uid bullet into
results/uid_validation.md; the clusters-have-no-ground-truth bullet into
results/case_studies.md; the groundedness-is-one-run bullet into
results/investigator_eval.md. The max_degree and calibration bullets
weren't duplicated anywhere -- checked sentence-by-sentence first, and
ARCHITECTURE.md and results/ablation.md's own Graph construction/
Calibration sections already state a superset of what README said, so
those two just got a link instead of a second copy. Diffed every numeric
token between the old and new README (`comm` on sorted number lists) to
confirm nothing changed value -- the only real deltas were numbers now
appearing once in compressed prose instead of twice (once in prose, once
in the still-present table), which is compression, not loss.

**What surprised me:** how much of the "full detail" this task asked to
relocate turned out to already exist in the target file, just phrased
differently -- ARCHITECTURE.md's max_degree sweep and ablation.md's
Calibration section were already supersets of README's own summaries.
The restructure mostly needed subtraction from README, not new writing
elsewhere.

No frozen pipeline files or app.py touched; 54/54 tests still pass. Not
committed -- this task didn't ask for one, and after the branch mixup
earlier this session, committing is now something I wait to be asked for
rather than assume.

## 2026-09-01 -- Task 1: two topology features, added without moving the existing numbers by a single digit

**What was built.** `graph.compute_cluster_features` gained `k_core_number`
(this uid's k-core index -- how deep the densest subgraph containing it
goes) and `star_ratio` (max node_degree among a cluster's active members /
cluster size -- close to 1 for a hub-and-spoke device farm, but ALSO close
to 1 for a same-size mutual clique by this formula alone, which is why the
docstring pairs it with the already-existing cluster_edge_density to tell
the two shapes apart). Both are gated behind a new `include_topology=False`
parameter -- default False, so every existing call site, run_pipeline.py
included, gets exactly the same 10 columns computed exactly the same way,
with zero code changes needed there.

**The real risk in this task wasn't the feature math, it was
`model.build_feature_matrix`.** It does `X.join(cluster_features)` with no
fixed column list -- if compute_cluster_features had simply always
returned 12 columns, the standard pipeline's "cluster model" would have
silently trained on 2 extra features next time anyone ran
`python -m src.run_pipeline`, without touching model.py, run_pipeline.py,
or any frozen file. Checked this BEFORE writing any feature code, not
after. The `include_topology` flag is the fix: existing behavior isn't
preserved by hoping nothing downstream changes, it's preserved because the
new columns are structurally absent unless asked for.

**Verification, not assumption:** ran the full pipeline afterward.
baseline/cluster/trimmed PR-AUC (0.5646/0.6322/0.5756), recall
(0.4791/0.5576/0.4951), and cost (30078.40/26155.72/29317.66) all matched
the required numbers exactly. 57/57 tests pass (54 existing + 3 new:
absent-unless-requested, a star/clique/singleton fixture proving why
star_ratio alone can't distinguish shape, and an explicit as_of-leakage
test for both new features).

**An unrelated bug this verification run surfaced, fixed in the same
sitting:** re-running the pipeline regenerates results/ablation.md and
results/investigator_eval.md from scratch, and both files had hand-added
correction sections from earlier sessions (ablation.md's "Detailed
findings", the single-split-vs-cross-split callouts; investigator_eval.md's
"Limitation, stated for README.md") that write_ablation_report /
write_investigator_eval know nothing about -- the regeneration silently
deleted them. This is exactly the "generated file gets hand-edited, next
regeneration wipes it" bug class this project already has one precedent
for (the ARCHITECTURE.md near-miss). Caught it because this task's own
hard constraint forced a real pipeline re-run rather than trusting the
diff would be clean; restored both sections afterward (fixing
investigator_eval.md's claim count to the actual 176 this run produced,
not the stale 182 it had before, since that number is regenerated by
definition and can't be pinned). Not fixed at the root (making
run_pipeline.py's generators aware of hand-added sections is a real
design change, out of scope for a task about graph.py) -- flagged here so
the next person who hand-edits a results/*.md file knows this can happen,
and the next full pipeline re-run should be treated as an opportunity to
check for this, not just the ablation numbers.

## 2026-09-01 -- Task 2: topology features made row 4 slightly WORSE, not better

**What was built.** `scripts/eval_topology.py` -> results/ablation_topology.md
(new file, results/ablation.md untouched). Four rows on the same single
80% split as results/ablation.md: (1) baseline, (2) + original cluster
features -- both read straight off the existing, already-verified
`run_pipeline.load_and_prepare()`/`train_both_models()`, not retrained --
(3) + original cluster features minus `cluster_prior_fraud_share`, and (4)
+ original + the two new topology features (`k_core_number`, `star_ratio`),
also minus `cluster_prior_fraud_share`. Rows 3-4 train fresh models via
`model.train_cluster_model` (frozen) on modified feature sets; every
metric comes from `evaluate.py`'s frozen functions.

**What the numbers say.** Row 4 vs. row 3 -- the comparison that isolates
whether topology adds anything with the dominant confound already
removed from both sides -- is PR-AUC **-0.0045**, recall@1%FPR **-0.0015**,
cost per 10k **+833.56** (worse). Not a rounding-noise tie: topology made
this single split slightly worse, not better. Feature importances confirm
why -- `star_ratio` ranks 113th of 443 features (gain 948.0), `k_core_number`
228th (gain 187.0), both far below the aggregates they were meant to
supplement (`cluster_amt_cv` gain 28,458.9, `cluster_velocity` 27,438.4).
Reported exactly as computed; no retry, no alternate formula, no threshold
adjustment to flip the sign.

**What this means so far.** On this one split, richer topology features
did not rescue the residual-lift story -- if anything they mildly hurt
it. That is one split's answer, though; Task 3 re-runs this same
comparison across the 4 rolling splits results/stability.md already used,
which is the test that actually matters for whether this is real or
single-split noise (the same lesson Task 3 of the earlier stability
session taught about the aggregate-only lift itself). Not drawing the
final conclusion here -- that's what the stability re-run is for.

## 2026-09-01 -- Task 3: topology still flips sign -- this is the session's answer

**What was built.** `scripts/eval_stability_topology.py` ->
results/stability_topology.md. Same 4 rolling splits as
results/stability.md (60/70/80/90%, fresh graph/features/models at each,
nothing cached), extended to also train a topology-augmented trimmed
model (original + `k_core_number` + `star_ratio`, minus
`cluster_prior_fraud_share`) at every split. Took 890s for all 4 splits
-- roughly 2.2x results/stability.md's own 408s, expected since this
trains 4 models per split instead of 3.

**Consistency check first:** the "without topology" column reproduces
results/stability.md's exact numbers at all 4 splits -- -0.0044 (60%),
+0.0186 (70%), +0.0110 (80%), -0.0013 (90%), byte-for-byte. Same script
family, same methodology, independently re-derived and matching --
real evidence this run is measuring the same thing, not just an
assumption.

**The answer, plainly: topology does not rescue the residual lift.**
With topology added, the lift is -0.0020 (60%), +0.0135 (70%), +0.0065
(80%), -0.0104 (90%) -- mean +0.0019, spread 0.0239. It still changes
sign at the exact same two splits (negative at 60% and 90%, positive at
70% and 80%) as the aggregate-only version (mean +0.0060, spread 0.0230).
Not a close call resolved narrowly either way -- the sign pattern is
identical, and the mean is if anything smaller with topology than
without it.

**Did this change my belief about the project's core claim?** Yes,
concretely. Before this session I'd have described the residual lift as
"probably real but underpowered to prove" -- a hedge that assumed a
richer feature set might tip it into being reliably positive. It didn't.
Combined with Task 2's single-split finding (topology made things
slightly worse, not better) and the two topology features' low feature
importance (star_ratio 113th of 443, k_core_number 228th), I now read
this as a real, structural finding rather than a measurement gap: on
this dataset, at this sample size, cluster topology beyond
`cluster_prior_fraud_share` does not carry a detectable abuse signal that
two different feature-engineering attempts (aggregates, then shape) could
surface. `cluster_prior_fraud_share` -- backward-looking, partly
circular fraud history -- is carrying essentially the entire reliably-
measured lift in this project, and that was true before this session and
remains true after two honest attempts to find something else. No
feature definition, threshold, or max_degree was tuned to reach this
conclusion; it's what both attempts actually found.

## 2026-09-01 -- Task 4: strengthened the Limitations bullet, not softened

**What was built.** Since Task 3 found topology does NOT rescue the
residual lift, README.md's "dominant feature is backward-looking, partly
circular fraud history" Limitations bullet was strengthened, not
softened: it now states explicitly that a second attempt with topology
features (k-core depth, hub-vs-clique shape) was made and still didn't
produce a stable effect across splits, framed as a finding about this
dataset and sample size, not a bug -- exactly the framing the task asked
for on a negative result. Links added to results/ablation.md and the new
results/stability_topology.md. results/stability.md's own numbers
(mean +0.0060, spread 0.0230, sign flip) are untouched and still the
number this bullet's cross-reference ultimately traces back to -- nothing
about this session hid or replaced that finding, it added a second,
consistent one on top of it.

**Whether this changed my belief, stated directly (repeating Task 3's
answer since it's the same belief-update, just now reflected in the
document a reader actually sees):** yes. Two independent feature-
engineering attempts -- aggregates, then topology -- have now failed to
find a reliable non-prior-fraud signal in this dataset's cluster
structure. That's no longer "underpowered, might still be real" in my own
read; it's a fairly settled negative result for this specific dataset and
sample size, and README.md now says so as plainly as the Result section
and Limitations bullet format (bold headline, two sentences, link out)
allow.

## 2026-09-01 -- Tasks 1-5: the dashboard shows a ring now, not just a table of numbers describing one

**What was built.** Three new files (`dashboard_theme.py`,
`dashboard_graph.py`, `dashboard_attribution.py`) plus changes to app.py.
Cluster detail now renders, in order, above the LLM narrative: (1) a real
plotly network graph of the cluster's actual entity-graph subgraph --
nodes sized by transaction count, colored by whether that uid carried a
fraud-labelled transaction, edges colored by which linkage rule
(device/addr+email/card+bank+addr) created them, with a legend, hover
tooltips, and click-to-pin node stats; clusters over 60 members get
sampled to their 60 highest-transaction-count members with a visible
note, never silently; (2) a MODEL-badged score-attribution block -- a
SHAP TreeExplainer (built once, `st.cache_resource`, never per selection)
against the real transaction whose score actually drove the cluster's
queue action, a top-12 contribution bar chart, a 0-1 threshold-position
chart reading `STEP_UP_THRESHOLD`/`REVIEW_THRESHOLD` live from
`src/policy.py` (never hardcoded), and the transaction-vs-cluster SHAP
split -- the project's central claim (clusters carry signal beyond the
raw transaction) made visible per decision, not just in aggregate
ablation numbers; (3) an off-by-default contrast toggle that renders a
second, real "allow"-decided cluster of the closest available size next
to the flagged one, using the identical graph-rendering code for both
sides. Added `plotly==5.24.1` to requirements.txt (matplotlib alone can't
give real hover/click; verified pip could actually resolve it before
committing to the dependency).

**What broke, and what I did before assuming anything.** Before writing
any UI code, I checked whether `model.build_feature_matrix`'s
`X.join(cluster_features)` (unconditional, no fixed column list) would
matter here -- it doesn't, since this session touches no frozen file and
adds no columns to `compute_cluster_features`; that check was really a
carry-over habit from the topology session, confirmed irrelevant here and
moved on. The real find was upstream of my new code entirely: I could not
get Playwright to select a cluster in the (pre-existing, untouched-by-me)
queue table no matter what I clicked, and checked
`docs/screenshot_detail.png` from a *previous* session's supposedly-
successful capture -- it also shows the unselected placeholder. This
row-selection interaction had never actually worked, in any session,
including the one that generated that screenshot. Root cause, found by
testing rather than assuming: `st.dataframe(styled_queue, ...)` had no
explicit `key=`, and a fresh `Styler` object is constructed on every
rerun; Streamlit apparently can't recognize that as "the same widget"
across the rerun the click itself triggers, so it reset the selection
before `render_cluster_detail` ever ran. Added `key="cluster_queue_df"`.
Verified the fix directly: a raw coordinate click plus polling for the
detail pane's own content (not a fixed sleep) now reliably reaches
"Score attribution," a real SHAP chart, and a real network graph. This
was a genuine, previously-undiscovered bug blocking the dashboard's core
interactivity, not something introduced this session -- fixed because
nothing in Tasks 1-3 would have been reachable by a real user otherwise.

**Verification, not assumption.** Before touching the browser at all,
wrote a plain-Python (no Streamlit, no Playwright) script exercising
every new function against the real trained pipeline: graph rendering on
the actual largest cluster (171 members, correctly sampled to 60 with a
note) and an actual 2-member cluster (no sampling, correct degenerate-case
values); SHAP computation on a real queue cluster's real driving
transaction, confirming `sum(shap_values) + expected_value` equals the
model's own raw-margin prediction exactly; the transaction/cluster split
summing back to `shap_row.sum()`; and `find_contrast_cluster` finding an
exact same-size (27-uid) real "allow" cluster. All of this passed before
a single Playwright click was attempted -- separating "is the logic
correct" from "did the click land," which turned out to be two entirely
different problems.

**Task 4 (coherence pass).** Read every tab as a first-time reviewer:
added one-line what/why captions to the Model performance and Audit
trail tabs (neither had one); clarified that the queue's `action` column
is policy.py's decision on the cluster's highest-scored transaction, not
the priority ranking itself (the two can and do disagree); added an
explicit POLICY badge and framing sentence to the Audit trail tab, which
showed real policy decisions with no source attribution at all; labeled
`score` as "predicted fraud probability, 0-1" once, prominently, rather
than repeating a units caveat on every table; added a "($)" unit to the
transaction timeline's amount axis; and reworded the Evidence panel's
label so it can't be confused with the new MODEL score-attribution block
above it (both were, before this session, just "the drove-the-score
numbers," which is now two different, clearly-attributed things).

**Task 5.** Regenerated all three docs/ screenshots with the fixed
`scripts/capture_screenshots.py` (same key-selection fix, plus polling
instead of a fixed sleep, plus a taller 2200px viewport so the graph and
MODEL attribution block fit in one shot). Caught one more timing bug on
the first re-run: the queue screenshot was captured empty -- the "loaded"
text appears before the canvas grid finishes painting its rows, a
separate async step. Added a short settle delay before that specific
capture. Re-ran and verified all three visually.

54/54 -> 57/57 tests still pass (57 is the current, correct baseline;
this session's task description assumed 54, which was the count before
the topology session added 3 -- noted rather than silently treated as a
mismatch). No file under src/ or results/ touched -- confirmed by diff,
not assumption -- so the pipeline's reported numbers are unchanged.

## 2026-09-01 — Live replay tab (new, additive-only)

Built a fourth tab, `dashboard_replay.py` + `render_replay_tab` in app.py:
replays real held-out test-split transactions in their actual
TransactionDT order, showing the pipeline exactly as it would run in
production. Read `app.py` fully first per the task's instruction; no
frozen file (`evaluate.py`, `entities.py`, `graph.py`, `model.py`,
`investigator.py`, `policy.py`) and no existing tab or helper function
was touched -- confirmed by `git diff app.py`, which shows only new
imports, one new function, one new tab-wiring line, and a docstring
update describing four tabs instead of three.

**Precompute discipline.** `build_replay_sequence` runs once under
`st.cache_data`: one vectorized `cluster_model.predict()` call, one
`policy.apply_policy()` call, and a single plain-Python pass building
per-transaction frames (score, action, new graph nodes/edges, running
cluster-max-score bookkeeping). Per-frame rendering only ever indexes
into that precomputed list -- no feature recomputation, no graph rebuild,
no model call happens on Play/Pause/rerun. The node layout is a single
`nx.spring_layout` computed once over every uid the window will ever
reveal (benchmarked at 7.3s for 1446 nodes), so revealing nodes over time
never moves an already-placed one.

**Window choice, verified against real data before writing any UI code.**
A throwaway diagnostic script found 10 real REVIEW_THRESHOLD crossings
across the full 118,108-row test split, and that the first 2000
chronological test transactions (matching the task's own suggested cap)
contain 2 of them -- clusters 769 and 7255. Checked whether either
crossing is a multi-member "group forming" story, since that's the payoff
moment the task describes: both are single-revealed-member crossings,
driven by that uid's cluster having a high `cluster_prior_fraud_share`
from training, not a build-up. A larger window exists with a genuine
5-member build-up (cluster 241, crossing at test-index 30512), but I kept
the smaller/faster 2000-transaction window and said so honestly in the
tab's caption instead of reaching for a bigger window to manufacture a
better-looking story -- this is itself a live illustration of this
project's own repeatedly-documented finding that `cluster_prior_fraud_share`
dominates the score, not a shortcoming to hide.

**A real bug, not just cleanup, found while re-reading my own draft
before committing.** `build_incremental_figure` initialized every node's
`any_fraud` to `False` and never updated it, so the "carried a
fraud-labelled txn" legend entry was dead -- every non-highlighted node
rendered as the same color regardless of its real label. The held-out
test split (carved from train_transaction.csv, unlike the never-released
Kaggle `test_transaction.csv`) does carry real `isFraud` labels, so this
was fixable with real data, not synthesis: added `is_fraud` to each
precomputed frame and folded it into `any_fraud` as frames are revealed.
Re-verified after the fix: 91 of 2000 replayed transactions carry a real
positive label, and the node trace now actually uses both risk colors.

**Verification, driven not eyeballed** (explicitly called out by the task,
after last session's queue-selection bug survived specifically because
nobody drove the UI). First, a plain-Python logic script against the real
trained pipeline (no browser): monotonic counters, correct
step_up/review firing points, correct highlight-linewidth transition
exactly at the crossing frame, both narratives real (`source="llm"`).
Then Playwright, twice -- once before the `is_fraud` fix and once after,
since the fix changed frame contents: load the app, switch tabs, click
Play, poll to the end of the 2000-transaction window, confirm both
Cluster 769 and Cluster 7255 narratives render, click Reset, confirm the
counters and review panel clear, and confirm zero console/page errors
throughout. Both full runs passed. (Minor Playwright-side friction, not
app bugs: the speed slider's react-aria overlay blocks a direct
`.click()`, so the test just runs at the default 15-txns/tick speed
instead; `get_by_role("button", name="Reset")` collides with Plotly's own
"Reset axes" modebar button and needs `exact=True`.)

57/57 tests still pass; `git diff --stat -- src/ results/` is empty.

## 2026-09-01 — README screenshots refreshed, three new views added

`scripts/capture_screenshots.py` now produces five images instead of
three: the existing queue and performance shots (regenerated fresh, so
the queue screenshot's tab bar now correctly shows Live replay as a
fourth tab), plus cluster 74986's entity graph on its own (a real 3-uid
ring, one node fraud-coloured -- picked because a 3-node ring reads
clearly, unlike a 27-member cluster's dense blob), the SHAP
score-attribution panel for that same cluster (transaction-vs-cluster
split included), and the Live replay tab mid-playback, paused just after
cluster 769's real REVIEW_THRESHOLD crossing so its LLM narrative panel
is visible.

Two things worth recording because they weren't obvious going in:

- **The detail view's content sits inside a fixed-height inner
  container.** `page.screenshot(full_page=True)` produced byte-identical
  output to `full_page=False` at the same viewport -- confirmed by
  diffing, not assumed -- meaning anything below the Evidence panel
  (the Transaction timeline chart) was never actually being captured by
  the prior script either, viewport scroll capture just silently didn't
  reach it. Worked around by cropping one 1700x2200 capture with PIL at
  measured y-boundaries for the graph and SHAP sub-images, rather than
  trying to scroll or full-page-capture further down.
- **The replay tab's scored-feed table clipped its rightmost (action)
  column at 1700px width**, cutting a word in the narrative below it
  mid-crop on a first attempt at a fixed-pixel fix (caught by re-reading
  the cropped image before committing, not by assuming the crop was
  safe). Fixed properly: widen to 2000px only for that one tab, then
  auto-detect the true content bounding box (both the tallest row and
  the widest column with real content) from the actual pixels instead of
  hardcoding a crop rectangle, so it isn't fragile to the narrative's
  length or the feed table's column widths varying run to run.

Row-selection is still the fragile part (see the module docstring):
cluster 74986 is reached by a raw pixel click at a canvas offset found
empirically, not by any DOM lookup, since the queue is a canvas-rendered
grid with no queryable cell text. Two capture runs this session found
the row-selection log line's own `heading.inner_text()` check racy (it
reported "no heading appeared" once even though the actual screenshot
came out correct) -- the screenshot itself, not that log line, is the
real check, and was verified visually both times.

No frozen pipeline file touched; only `scripts/capture_screenshots.py`,
`docs/*.png`, and `README.md` changed. 57/57 tests pass.
