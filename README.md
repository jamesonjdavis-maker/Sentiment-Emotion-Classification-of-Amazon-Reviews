# Review Sentiment Audit: Amazon Gift Card Reviews

This project tests how well an AI language model (LLM) can read an Amazon gift card review and tell whether the reviewer was happy, unhappy, or in between. The model only sees the review's **title and text**, never its star rating. Afterwards, the star rating is used as the "correct answer" to check the model's work. The model also names the reviewer's main emotion, and that is compared with a simpler word-counting method. Everything is shown in an interactive dashboard.

![Dashboard overview, balanced 3-class run](screenshots/dashboard_overview.png)

## The short version

- **Telling happy from unhappy reviews is easy for the model.** On an even mix of 50 good and 50 bad reviews, it agreed with the star rating 99.0% of the time. Most of those reviews were very clear-cut (5-star or 1-star), though.
- **Middle-of-the-road reviews are where it struggles.** When 3-star reviews are added as their own "neutral" group, agreement drops to **71.3%**, because **3-star reviews mostly don't get their own class: 30 of 50 were labeled negative.** Reading them, most 3-star reviews are complaints, so the model is often reacting to what people wrote rather than the score they gave.
- **A random sample hides this problem.** 84% of all reviews are 5 stars. In a random 100, there were no 3-star reviews at all, so the weakness never showed up.
- **The two emotion methods rarely agree** (16.0% of the time). The word-counting method misses most complaints, because words like "scam" and "refund" aren't in its word list, and the word "gift" throws it off.

---

## Key terms

| Term | Meaning here |
|---|---|
| **Agreement / accuracy** | The share of reviews where the model's label matches the star rating. |
| **Correct answer** | The label the star rating implies. With 2 classes: 4 or more stars is positive, otherwise negative. With 3 classes: 4 to 5 stars positive, 3-star neutral, 1 to 2 stars negative. |
| **Baseline** | The score you would get by always guessing the most common answer. A model is only useful if it beats this. |
| **Balanced sample** | An equal number of reviews from each class, so the rare classes get tested properly. A **random** sample instead mirrors the real (very lopsided) data. |
| **Balanced accuracy** | The average hit rate across classes. Unlike plain accuracy, it can't be inflated by one large class. |
| **Recall** (hit rate) | Of the reviews that truly belong to a class, how many the model found. |
| **Precision** | Of the reviews the model put in a class, how many truly belong there. |
| **95% range** | The range the true value most likely falls in. With only 50 to 100 reviews, these ranges are wide. |
| **Confusion matrix** | A table of what the stars say against what the model said. The diagonal is agreement; everything else shows which way the mistakes go. |

---

## Data

- **Source:** Amazon Reviews '23, *Gift Cards* category, by Hou, Li, He, Yan, Chen & McAuley, *Bridging Language and Items for Retrieval and Recommendation* (2024), McAuley Lab, UC San Diego.
  Dataset page: <https://amazon-reviews-2023.github.io> · File: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz>
- **Size:** 152,410 reviews written between 2008-08-06 and 2023-09-06.
- **What I kept:** the star `rating` (the correct answer, never shown to the model), the `title` and `text` (the only things the model reads), plus a few extra details such as whether it was a verified purchase, helpful votes, and the date. 49 reviews have empty text; every review has a title.
- **The data is very lopsided:**

| Stars | Reviews | Share |
|---|---:|---:|
| 5 stars | 128,248 | 84.1% |
| 4 stars | 6,692 | 4.4% |
| 3 stars | 3,271 | 2.1% |
| 2 stars | 1,873 | 1.2% |
| 1 star | 12,326 | 8.1% |

88.5% of all reviews are 4 or 5 stars, and only 2.1% are 3 stars. Any random sample inherits this imbalance.

- **Emotion word list:** the NRC Word-Emotion Association Lexicon (EmoLex) v0.92, by Mohammad & Turney (2013), <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>. The script downloads it automatically. It is **not included in this repository** because its terms ask that it not be redistributed, and the script checks the file's fingerprint (SHA-256) before using it.

## How it works

1. **The prompt** (`sentiment.py`) gives the model a review's title and text, plus rules for tricky cases (for example, if the title and text disagree, the text wins). The model must answer with a fixed label so the answer can be read automatically.
2. **The scoring script** (`score_batch.py`) picks a sample of reviews with a fixed random seed (42), sends each one to the model, then compares the answers with the star ratings.
3. **The word-list script** (`emotion_lexicon.py`) counts emotion words in each review and picks the emotion with the most matches.
4. **The dashboard generator** (`build_dashboard.py`) turns the saved results into `dashboard.html`.

| Setting | Value |
|---|---|
| Model endpoint | The class's OpenAI-compatible endpoint (`http://dobolyi.com:9001/v1`), called with the `openai` Python library |
| Model | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` |
| Settings | temperature 0 (least random); the model's "thinking" step turned off; answers restricted to the allowed labels |
| Unreadable answers | 0 in every run |

## Results

| Run | Classes | Sample | Agreement with rating (95% range) | Baseline | Balanced accuracy |
|---|---|---|---|---|---|
| Step 2, random | 2 | 100 random | **98.0%** (93.0 to 99.4%) | 89.0% | 94.9% |
| Step 2, balanced | 2 | 50 / 50 | **99.0%** (94.6 to 99.8%) | 50.0% | 99.0% |
| Step 6, random | 3 | 100 random | **89.0%** (81.4 to 93.7%) | 89.0% | 89.8% *(no neutral reviews in sample)* |
| Step 6, balanced | 3 | 50 / 50 / 50 | **71.3%** (63.6 to 78.0%) | 33.3% | 71.3% |

Asking the model for an emotion as well (Step 5) changed **0 of 200** of its sentiment answers, so the Step 2 results still stand.

---

## Findings

### 1. Why did the lopsided run look so accurate, and what did equal sampling change?

The random sample of 100 looks like the real data: **89 positive, 11 negative, and zero 2-star or 3-star reviews**. A "model" that ignored the text and always said "positive" would already score **89.0%**. So the impressive 98.0% is really only 9 points better than guessing. The negative side was tested on just 11 reviews, so its 90.9% hit rate is uncertain (the 95% range is 62 to 98%).

Testing with equal numbers from each class showed two different things:

- **With two classes, the warning in the assignment did *not* hold up.** I expected the lopsided result to flatter the model. It didn't: on 50 positive and 50 negative reviews the model scored **99.0%** and caught **50/50** negatives. Telling positive from negative really is easy for this model. The imbalance made the *baseline* look high, not the model.
- **With three classes, balancing exposed the real weakness.** The random 3-class run scores **89.0%, exactly the always-positive baseline**, because it has **no neutral reviews at all**, so it can't measure how the model handles them. The only sign of trouble is 9 positive reviews (7 five-star, 2 four-star) wrongly called neutral. With 50 reviews per class, agreement drops to **71.3%**, and the model finds only **30.0%** (15/50; 95% range 19 to 44%) of the neutral reviews.

**Lesson:** a random sample of this data can look excellent while never testing the hardest class.

**A second, hidden imbalance.** "Balanced" here means an equal number *per class*, picked at random from the whole file, so the imbalance survives *inside* each class. In both balanced runs, the 50 positives are 49 five-star and 1 four-star, and the negatives are mostly 1-star (39 of 50, the rest 2-star, plus 4 three-star in the 2-class run). So the 99.0% score mostly tests the clearest, most extreme reviews; milder 2-star and 4-star reviews are barely tested. Sampling an equal number per *star level* would be the next step.

![Lopsided 2-class run with the baseline note](screenshots/dashboard_lopsided_2class.png)

![Random 3-class run: the neutral class is simply missing](screenshots/dashboard_random_3class.png)

### 2. Where do the mistakes go?

Confusion matrix for the balanced 3-class run (rows are what the stars say; columns are what the model said):

| Stars say ↓ / Model says → | Positive | Neutral | Negative | Total |
|---|---:|---:|---:|---:|
| **Positive** | **45** | 5 | 0 | 50 |
| **Neutral** | 5 | **15** | 30 | 50 |
| **Negative** | 1 | 2 | **47** | 50 |

- **Neutral reviews get pushed down to negative, not the other way around.** 30 of 50 three-star reviews (60%) were labeled negative, but only 2 of 50 negative reviews (4%) were called neutral.
- **So the model says "negative" too often:** 77 labels vs. 50 true negatives. It finds nearly all real negatives (hit rate **94.0%**), but only **61.0%** of its "negative" labels are right. It says "neutral" too rarely: 22 labels vs. 50.
- **Positive reviews are mostly right** (hit rate **90.0%**). All of its 5 misses went to neutral, and they are short, flat reviews such as *"It's a gift card....,what more is there to say ?"* (5 stars) and *"Gas card."* (5 stars).
- **Most of the 30 three-star reviews labeled negative are actually complaints:** a missing gift message, a dented card, a $6.95 loading fee, a card that won't work with Apple or Google Wallet. The model's emotion labels agree: every one got a negative emotion (anger 18, disgust 8, sadness 4). The reviewers wrote negative text but gave a middle score. So many of these "mistakes" are really a gap between what people *write* and how they *rate*. The model judges the text, and 3-star text in this category is mostly unhappy.
- **The 2-class balanced run shows the same thing:** all 4 three-star reviews it included were labeled negative.

![Filtered to the 30 three-star reviews labeled negative](screenshots/dashboard_filter_neutral_to_negative.png)

Some 2-class "mistakes" look more like wrong star ratings than model errors: a 1-star review that says *"Not much you can say other than it works"*, and a 5-star review that is just a joke about the *House of the Dead* video game.

### 3. How do the LLM's emotions and the word list's emotions differ, and why?

Balanced 3-class run (150 reviews):

| | LLM | Word list (NRC) |
|---|---|---|
| Most common answers | anger 56, joy 39, trust 31, disgust 14 | joy 21, anticipation 19, trust 16 |
| No single answer | n/a | **none 30** (no emotion words found) · **tie 51** (several emotions tied, all from the same word) |

- **They rarely agree: 16.0% exact.** Even counting a match whenever the LLM's emotion is one of the word list's tied choices, they agree only 31.3% of the time.
- **They disagree most on unhappy reviews.** Exact agreement is 34% for positive reviews but **6% for neutral and 8% for negative**. On the 50 negative reviews, the LLM says *anger* 38 times; the word list says *anger* twice.
- **Why they differ:**
  1. **Complaint words are missing from the word list.** "scam", "refund", "card", "amazon", "cheated", "useless" and "ripoff" have no emotion attached. For example, a 1-star review titled *"ZERO BALANCE!!!"* that reads *"Fandango card was empty!"* contains no listed words at all, while the LLM reads it as anger. 4 of the 50 negative reviews are like this.
  2. **Everyday product words mislead it.** The word list links "gift" to anticipation, joy *and* surprise, and "gift" matched 143 times across 68 of the 150 reviews. 36 of the 51 ties involve "gift", and 22 ties are exactly its anticipation/joy/surprise set. "money" is linked to five emotions at once. So the word list mostly measures what the product is, not how the reviewer feels.
  3. **It ignores context.** It can't handle "not happy", sarcasm, or situations that imply anger without emotional words (like a card that didn't activate).
  4. **Reviews are very short** (median 8 words), so there is little to count: 30 of 150 matched nothing.
- **A caution about the LLM:** there is no "correct answer" for emotions, so neither method can be scored. The LLM's emotions are at least consistent with its sentiment labels: *joy* appears only on reviews it called positive (39); *anger*, *disgust*, *sadness* and *fear* only on reviews it called negative (56, 14, 6, 1); and *trust* is split between neutral (19) and positive (12). Part of that consistency comes from the prompt's own definitions and from both labels coming in the same answer.

![Emotion comparison](screenshots/dashboard_emotions.png)

### 4. Problems I ran into, and how I fixed them

| Where | Problem | Fix |
|---|---|---|
| Model | The class model "thinks" before answering. By default it spent 169 hidden tokens reasoning (saved in `results/report_supporting_numbers.txt`), so short replies came back **empty**, and its reasoning could mention both labels and confuse the code reading the answer. | Turned the thinking step off and strip any leftover thinking text. Replies went from 1.24 s to 0.11 s. |
| Model answers | Free-form answers can come back in the wrong format. | Restricted the model to a fixed list of allowed answers: 0 unreadable replies in every run. |
| Sampling | The random sample had no 2-star or 3-star reviews, so some averages couldn't be calculated and came out as "not a number", which would have broken the dashboard. | Averages now use only the classes that are present and say which one is missing; the dashboard shows "none in this sample" instead of a misleading 0%. |
| Word list | My first rule for breaking ties couldn't choose between emotions triggered by the *same* word, and quietly fell back to alphabetical order. That handed about 20% of reviews to "anticipation". | Those cases are now labeled `tie`, and agreement is reported both strictly and loosely. |
| Charts | Very small values (the 1.2% 2-star bar, a single review, a 4-star row with only 1 review) could shrink to nothing on screen, the problem the assignment warns about. | Every non-zero bar has a minimum width, and I measured the drawn bars in the browser to confirm (see below). |
| Charts | On phones, two percentage labels ("33%", "30%") were wider than their bars and spilled over. The cause: the bars got only 70 px of 309, because the note beside them ("n=50 · should be Neutral") took 149 px. | On narrow screens the note now sits under the bar (bar now 167 px wide), and labels only show when they fit. |
| Charts | My first fix for the labels measured the wrong thing and hid **every** label, and the check still passed. I only caught it because the check also counts how many labels are showing (it said 0). | The fix now measures the text itself, and the check confirms each hidden label really wouldn't fit. |
| Colors | Gray for "neutral" failed a colorblind-safety check; yellow and pink were too close to orange. | Blue, aqua and orange pass every check in light and dark mode. |
| Layout | On phones the run tabs broke across lines, a filter button wrapped, and some reviews showed raw `<br />` code. | Tabs scroll sideways, buttons stay on one line, and the code is converted to normal line breaks. |
| Screenshots | Screenshots came out blank after scrolling. | Added a link option that shows just one section of the page, then captured that. |
| Repeatability | Re-running the model on the same reviews at temperature 0 still changed 1 of 150 sentiment labels and 5 of 150 emotions (a normal side effect of shared computing servers). | All published numbers come from the saved results, and the re-run difference is reported openly (see *Repeatability* below). |
| Rebuilding | The star chart needed the raw data, which isn't in the repository. | The generator saves a small summary file (`results/dataset_summary.json`), so the dashboard rebuilds from a fresh copy. |
| Working with the agent | At first the agent only wrote scripts for me to run (a preference it had saved from an earlier project), and an early answer about "why do we need an API" was confusing. The agent does the work; the endpoint does the classifying. | Told it to run everything itself, and clarified the difference between the agent (writes and runs code) and the endpoint (labels the reviews). |
| Working with the agent | The agent's own drafts had mistakes that only checks caught: the alphabetical tie-break above, a sentence saying "gift" appeared in 143 *reviews* (it is 143 *matches* in 68 reviews), and the imbalance inside each class, which was missed until I asked for a full audit. | Had every number in this report recomputed from the saved files: `check_readme_numbers.py` re-derives them all, and `check_repeatability.py` re-tests that the model never sees the rating and that results repeat. |

**Numbers check:** `verify_dashboard.js` compares every number shown on the dashboard with the saved results, inside the browser: **429 checks, 0 mismatches** across all 4 runs.

**Chart check:** the same script measures every bar at desktop, tablet and phone widths (1280, 768 and 375 px). It confirms that every non-zero bar is at least 3 px wide, bar lengths match their values, labels fit, and nothing spills out of its box: **0 failures at all three widths** (228 / 228 / 221 checks). Record: `results/dashboard_check.txt`.

---

## The dashboard

`dashboard.html` is a single file: open it in any browser, with no internet connection needed. Tabs at the top switch between the four runs; it opens on the balanced 3-class run.

- The headline agreement with its 95% range, the guessing baseline, balanced accuracy, and a plain-language note about the imbalance.
- Charts of the star ratings in the full data, what the model called each star level, how many reviews each class should have vs. how many the model gave it, a clickable confusion matrix, and the hit rate for each class.
- The LLM's emotions next to the word list's, with agreement by class and the most-matched words.
- A table of every review, with filters (agrees or not, class, model label, stars, emotion, text search) and a live count. Clicking a chart filters the table, and clicking a row shows the full review, the model's exact reply, and the words the word list matched.
- Light and dark themes. All colors are set in one place at the top of the file, so they are easy to change.
- Links can open a filtered view, e.g. `dashboard.html?truth=NEUTRAL&pred=NEGATIVE#150_balanced_s42_c3_emo`.

![Dark theme](screenshots/dashboard_dark.png)

## How to reproduce it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # endpoint address, key and model name
python data_loader.py           # Step 0: download and read the data (about 12 MB)
python spot_check.py            # Step 1: quick test of the prompt
python score_batch.py                                          # Step 2: random 100, 2 classes
python score_batch.py --balanced                               # Step 2: 50 / 50
python score_batch.py --emotion && python score_batch.py --emotion --balanced   # Step 5: add emotion
python score_batch.py --classes 3 --emotion                    # Step 6: random 100, 3 classes
python score_batch.py --classes 3 --emotion --balanced --n 150 # Step 6: 50 per class
python emotion_lexicon.py       # Step 5: word-list emotions (downloads the word list the first time)
python build_dashboard.py       # builds dashboard.html
python report_numbers.py        # extra figures quoted in this report
```

If a run is interrupted, re-running it picks up where it stopped, and a finished run reuses its saved answers instead of asking the model again.

**Repeatability** (`python check_repeatability.py`, saved in `results/repeatability_check.txt`):

- **The same reviews are picked every time.** Re-drawing each run with its seed (42) gives exactly the same reviews for all six saved runs.
- **Every published number can be rebuilt exactly.** Recalculating from the saved results reproduces the same files, byte for byte.
- **Asking the model again gives almost, but not exactly, the same answers.** Re-sending all 150 reviews of the balanced 3-class run gave 149 of 150 identical sentiment labels and 145 of 150 identical emotions (accuracy 72.0% instead of the saved 71.3%, well within its 95% range). This is normal for shared computing servers. That is why every number here comes from the saved results.
- **The model never sees the rating.** The same check records every message sent to the model: each one contains only the title and text, and the instructions never mention stars or ratings.

**Number audit:** `python check_readme_numbers.py` recalculates every number in this README from the saved results and confirms it matches (`results/readme_number_check.txt`).

## Files

| What | File |
|---|---|
| The prompt | `sentiment.py` (version 1: two classes · version 2: plus emotion · version 3: three classes plus emotion) |
| Scoring script | `score_batch.py` (uses `llm_client.py` and `data_loader.py`) |
| Word-list emotion script | `emotion_lexicon.py` |
| Dashboard generator | `build_dashboard.py` and `dashboard_template.html` |
| Raw output of one balanced run | `results/batch_150_balanced_s42_c3_emo.csv` (every review, its correct answer, the model's answer and emotion, the model's exact reply, and the word-list emotion) |
| Final dashboard | `dashboard.html` |
| Supporting checks and records | `spot_check.py`, `verify_dashboard.js`, `report_numbers.py`, `check_repeatability.py`, `check_readme_numbers.py`, `results/*_metrics.json`, `results/*_report.txt`, `ISSUES_LOG.md` |

The raw data and word list (`data/`) and the private settings file (`.env`) are not included.

## Limitations

- The samples are small (100 to 150 reviews), so the numbers for each class have wide 95% ranges.
- The balanced samples are balanced by class, not by star level, so 4-star reviews (1 per run) and 2-star reviews are barely tested.
- The star rating is treated as the truth, but some ratings clearly don't match their own review text.
- Only one model and one setting were tested, and the prompt's rules (for example, "'ok' is neutral") affect how neutral reviews are labeled.
- Emotions have no correct answer, so the comparison shows where the two methods disagree, not which one is right.
