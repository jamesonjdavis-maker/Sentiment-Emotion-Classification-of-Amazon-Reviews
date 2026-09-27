// In-browser check that the dashboard shows exactly what the scoring saved.
// Open dashboard.html, paste this into the DevTools console (or run it through an
// agent's browser tool). For every run it compares each displayed number with the
// embedded metrics JSON / saved rows, checks every filter's live count and rows,
// and measures rendered bar widths so no non-zero value collapses to 0 px.
(() => {
  const out = [];
  const p1 = x => (x * 100).toFixed(1) + '%';
  const px = el => el.getBoundingClientRect().width;
  for (const r of DATA.runs) {
    selectRun(r.id);
    const m = r.metrics, L = r.meta.labels;
    const chk = (name, shown, want) =>
      out.push(`${r.id} | ${name}: page=${shown} saved=${want} ${String(shown) === String(want) ? 'OK' : 'MISMATCH'}`);

    // headline
    chk('accuracy', $('#heroFig').textContent, p1(m.accuracy));
    chk('balanced accuracy', $('#tBal').textContent, p1(m.balanced_accuracy));
    chk('majority baseline', $('#tBase').textContent, p1(m.majority_baseline ?? m.always_positive_baseline));
    chk('mismatches', +$('#tMiss').textContent, r.rows.filter(x => !isTrue(x.correct)).length);

    // confusion matrix + per-class recall
    document.querySelectorAll('#cm .cell').forEach(c =>
      chk(`matrix ${c.dataset.truth}->${c.dataset.pred}`, +c.dataset.count, m.confusion_matrix[c.dataset.pred][c.dataset.truth]));
    L.filter(l => m.per_class[l].support).forEach(l =>
      chk('recall ' + l, document.querySelector(`[data-class-recall="${l}"]`).textContent, p1(m.per_class[l].recall)));

    // step 7: full-file stars, model label by star, rating vs model counts
    document.querySelectorAll('#starDist .lane').forEach(el => {
      chk(`file ${el.dataset.star}★ count`, +el.dataset.count, DATA.dataset.rating_counts[el.dataset.star]);
      const bar = el.querySelector('.bar');
      if (+el.dataset.count > 0) chk(`file ${el.dataset.star}★ bar visible (>=3px)`, px(bar) >= 3, true);
    });
    document.querySelectorAll('#starPred .inseg').forEach(el => {
      chk(`${el.dataset.star}★ -> ${el.dataset.label}`, +el.dataset.count, m.star_by_pred[el.dataset.star][el.dataset.label]);
      chk(`${el.dataset.star}★ -> ${el.dataset.label} segment visible (>=3px)`, px(el) >= 3, true);
    });
    document.querySelectorAll('#rvm .lane').forEach(el => {
      const l = el.dataset.class;
      chk(`rating says ${l}`, +el.dataset.support, m.per_class[l].support);
      chk(`model says ${l}`, +el.dataset.predicted, m.per_class[l].predicted);
      el.querySelectorAll('.bar').forEach((b, i) => {
        const v = i ? +el.dataset.predicted : +el.dataset.support;
        if (v > 0) chk(`${l} ${i ? 'model' : 'rating'} bar visible (>=3px)`, px(b) >= 3, true);
      });
    });

    // honest-imbalance caveat on balanced runs
    if (r.meta.sampling === 'balanced') {
      const pos = r.rows.filter(x => x.truth === 'POSITIVE'), neg = r.rows.filter(x => x.truth === 'NEGATIVE');
      chk('within-class caveat', $('#withinMix').textContent,
          `${pos.filter(x => x.rating === 5).length} of ${pos.length} positives are 5-star and ${neg.filter(x => x.rating === 1).length} of ${neg.length} negatives are 1-star`);
    }

    // step 5: emotions
    if (m.emotion) {
      const e = m.emotion;
      chk('emotion exact agreement', $('#eAgree').textContent, p1(e.agreement));
      chk('emotion lenient agreement', $('#eLenient').textContent, p1(e.agreement_lenient));
      chk('word list none', $('#eNone').textContent, `${e.n_lexicon_none} of ${e.n}`);
      chk('word list tie', $('#eTie').textContent, `${e.n_lexicon_unbreakable_tie} of ${e.n}`);
      document.querySelectorAll('#emoBars .lane').forEach(el => {
        const c = el.dataset.emo;
        if (el.dataset.llm !== '') chk(`LLM ${c}`, +el.dataset.llm, e.llm_counts[c]);
        chk(`word list ${c}`, +el.dataset.lex, e.lex_counts[c]);
        chk(`LLM ${c} (vs rows)`, el.dataset.llm === '' ? 'n/a' : r.rows.filter(x => x.llm_emotion === c).length, el.dataset.llm === '' ? 'n/a' : +el.dataset.llm);
        chk(`word list ${c} (vs rows)`, r.rows.filter(x => x.lex_emotion === c).length, +el.dataset.lex);
      });
    }

    // step 4: filters (live count and exact rows)
    const cnt = () => +$('#liveCount').textContent;
    const wrong = r.rows.filter(x => !isTrue(x.correct));
    applyFilter({ match: 'wrong' });
    chk('filter mismatched count', cnt(), wrong.length);
    state.shown = 1e9; renderTable();
    chk('filter mismatched rows', [...document.querySelectorAll('#tbody tr.main')].map(t => +t.dataset.id).sort().join(','),
        wrong.map(x => x.review_id).sort().join(','));
    applyFilter({ match: 'correct' }); chk('filter agreeing count', cnt(), r.rows.length - wrong.length);
    for (const t of L) for (const p of L) { applyFilter({ truth: t, pred: p }); chk(`filter cell ${t}->${p}`, cnt(), m.confusion_matrix[p][t]); }
    for (const s of [1, 2, 3, 4, 5]) { applyFilter({ stars: String(s) }); chk(`filter ${s}★`, cnt(), r.rows.filter(x => x.rating === s).length); }
    if (m.emotion) {
      applyFilter({ emoAgree: 'same' }); chk('filter emotions agree', cnt(), Math.round(m.emotion.agreement * m.emotion.n));
      applyFilter({ emo: 'anger' }); chk('filter LLM anger', cnt(), m.emotion.llm_counts.anger);
    }
    applyFilter({ q: 'scam' }); chk('filter search "scam"', cnt(), r.rows.filter(x => (x.title + ' ' + x.text).toLowerCase().includes('scam')).length);
    // "none in this sample" placeholders keep bar height (caught a CSS class collision once)
    document.querySelectorAll('.stack.empty').forEach(el => chk('placeholder height <= 18px', el.getBoundingClientRect().height <= 18, true));
  }
  selectRun(DATA.default_run);
  const bad = out.filter(x => x.includes('MISMATCH'));
  return `${out.length} checks, ${bad.length} mismatches\n` + (bad.length ? bad.join('\n') + '\n---\n' : '') + out.join('\n');
})();

// ── Geometry check (Step 7: "bars given zero width by a label/positioning interaction") ──
// Run once per viewport width (e.g. 1280 / 768 / 375). For every chart on every run:
//  * each non-zero bar/segment is >= 3px wide (never collapsed),
//  * bar lengths are proportional to their values (within 1.5px, except where the 3px floor applies),
//  * labels drawn inside stacked segments fit inside them,
//  * value labels next to bars stay inside their card, and the page has no horizontal overflow.
(() => {
  const out = [], px = el => el.getBoundingClientRect();
  const chk = (id, name, ok, detail = '') => out.push(`${id} | ${name}: ${ok ? 'OK' : 'FAIL'} ${detail}`);
  const within = (a, b, tol = 1.5) => Math.abs(a - b) <= tol;
  let shown = 0, hidden = 0;
  for (const r of DATA.runs) {
    selectRun(r.id);
    // plain bar charts: all bars in one chart share one scale
    for (const chart of ['#starDist', '#rvm', '#emoBars']) {
      const root = $(chart); if (!root || root.offsetParent === null) continue;
      const bars = [...root.querySelectorAll('.bar')].map(b => ({ b, v: parseFloat(b.style.width.split('*')[1]) }));
      const card = root.closest('.card').getBoundingClientRect();
      const maxW = Math.max(...bars.map(x => px(x.b).width));
      bars.forEach(({ b, v }, i) => {
        const w = px(b).width, want = v * maxW;
        if (v > 0) chk(r.id, `${chart} bar ${i} not collapsed`, w >= 3, `${w.toFixed(1)}px`);
        if (want >= 3) chk(r.id, `${chart} bar ${i} proportional`, within(w, want), `${w.toFixed(1)} vs ${want.toFixed(1)}px`);
        const label = b.nextElementSibling;
        if (label) chk(r.id, `${chart} label ${i} inside card`, px(label).right <= card.right - 8, `${px(label).right.toFixed(0)} <= ${(card.right - 8).toFixed(0)}`);
      });
    }
    // 100% stacked rows (model label by star) + hit-rate tracks: segments ∝ counts
    const stacks = [...document.querySelectorAll('#starPred .stack:not(.empty)')].map(s => [s, [...s.children].map(c => +c.dataset.count)]);
    const tracks = [...document.querySelectorAll('#bars .track')].map(t => [t, [...t.children].map(c => +(c.getAttribute('style').match(/flex:(\d+)/)[1]))]);
    for (const [row, counts] of [...stacks, ...tracks]) {
      const segs = [...row.children], n = counts.reduce((a, b) => a + b, 0);
      const usable = px(row).width - 2 * (segs.length - 1);          // 2px gaps
      const floor = counts.filter(c => usable * c / n < 3).length;   // segments lifted to 3px
      segs.forEach((s, i) => {
        const w = px(s).width, want = usable * counts[i] / n;
        chk(r.id, `stack segment (${counts[i]}/${n}) not collapsed`, w >= 3, `${w.toFixed(1)}px`);
        if (!floor) chk(r.id, `stack segment (${counts[i]}/${n}) proportional`, within(w, want), `${w.toFixed(1)} vs ${want.toFixed(1)}px`);
        // measure the text itself (scrollWidth equals clientWidth whenever text fits, so it can't tell)
        const tw = el => { const g = document.createRange(); g.selectNodeContents(el); return g.getBoundingClientRect().width; };
        if (s.textContent.trim()) { shown++; chk(r.id, `in-bar label "${s.textContent}" fits`, tw(s) + 8 <= s.clientWidth + 0.5, `${tw(s).toFixed(1)}+8 <= ${s.clientWidth}`); }
        else if (s.dataset.lbl) {   // a hidden label must be one that genuinely would not fit
          hidden++; s.textContent = s.dataset.lbl; const need = tw(s) + 8, have = s.clientWidth; s.textContent = '';
          chk(r.id, `hidden label "${s.dataset.lbl}" really too wide`, need > have, `${need.toFixed(1)} > ${have}`);
        }
      });
    }
  }
  selectRun(DATA.default_run);
  chk('page', 'no horizontal overflow', document.documentElement.scrollWidth <= innerWidth, `${document.documentElement.scrollWidth} <= ${innerWidth}`);
  const bad = out.filter(x => x.includes('FAIL'));
  return `viewport ${innerWidth}px: ${out.length} geometry checks, ${bad.length} failures (in-bar labels shown ${shown}, hidden ${hidden})\n` + bad.join('\n');
})();
