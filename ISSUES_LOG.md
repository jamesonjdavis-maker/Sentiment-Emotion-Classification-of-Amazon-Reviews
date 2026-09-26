# Issues log (feeds report question 4)

| Step | Issue | Fix |
|---|---|---|
| 1 | Class model (Qwen3.6-35B-A3B) is a *thinking* model: by default it spent ~170 tokens reasoning before answering, so a 5-token cap returned an empty reply (every review would have been UNPARSEABLE). Reasoning text could also contain both labels, fooling a naive regex. | Disabled thinking via `chat_template_kwargs: {enable_thinking: false}`, strip any `<think>` block before parsing; answer now returns in ~0.1 s. |
| 2 | Random 100 contained 0 two-star and 0 three-star reviews and only 11 negatives — 89% accuracy was available for free. | Report "always POSITIVE" baseline + balanced accuracy; added a 50/50 balanced run. |
| 3 | Run tabs wrapped into a broken pill at phone width. | Tabs scroll horizontally, no wrap. |
| 3 | "✓ Agrees" filter label broke onto two lines at narrow widths. | `white-space: nowrap` on segmented-control buttons. |
| 3 | Review text contains raw `<br />` HTML, which showed literally in the table. | Replaced with spaces in previews, newlines in the expanded view. |
| 3 | In-app browser pane screenshots came back blank when emulating a desktop viewport. | Used headless Chrome full-page capture for screenshots instead. |
| 3 | A 1-in-89 error segment risks rendering at ~0 px wide. | `min-width: 3px` on bar segments (and 4px on the class-mix strip). |
| 5 | Word-list tie-break "earliest word wins" could not separate emotions triggered by the *same* word ("gift" → anticipation/joy/surprise) and silently fell back to alphabetical order, handing 20–24 reviews per run to "anticipation". | Unbreakable ties are now labeled `tie` with the tied set kept in `lex_top`; agreement reported strictly and leniently. |
| 5 | Parsing free-text JSON from the model risks malformed replies. | Used the endpoint's enum-constrained `json_schema` response format — the server can only emit valid labels/emotions (0 unparseable). |
| 6 | The random-100 3-class run contains **zero** 3★ reviews, so neutral recall was 0/0 → balanced accuracy and macro-F1 came out `NaN`, and `NaN` is not valid JSON (would break the dashboard's data load). | Balanced accuracy / macro-F1 now average only the classes present and name the absent class; builder converts NaN → null; dashboard shows "none in this sample" hatching instead of a fake 0%. |
| 7 | Neutral class was first drawn in gray; the colorblind/chroma validator failed it ("reads as gray"). Yellow and pink alternatives failed the normal-vision separation check next to orange. | Neutral = aqua (blue / aqua / orange passes every pairwise check in light and dark). |
| 7 | Dashboard needed the full-file star distribution, but `data/` is git-ignored, so a fresh clone could not rebuild it. | Builder writes `results/dataset_summary.json` (committed) and reads it when the raw data is absent. |
| 7 | Tiny chart elements (the 1.2% 2★ bar, a 1-of-49 segment, a 4★ row with n=1) risk rendering at 0 px. | `min-width: 3px` on every non-zero bar/segment; verified by measuring rendered widths at 1280px and 375px (0 collapsed). |
| 7 | Balanced runs labeled the baseline "Always say Positive" although all classes are equal. | Label switches to "Always guess one class" on balanced runs. |
