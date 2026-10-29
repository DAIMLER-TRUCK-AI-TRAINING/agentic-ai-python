/* Agentic AI Systems Engineering — shared helpers: theme toggle, SVG builder, tween, log, terminal, self-quiz */

(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('aai-viz-theme'); } catch (e) {}
  if (saved) document.documentElement.setAttribute('data-theme', saved);
})();

function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute('data-theme') ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('aai-viz-theme', next); } catch (e) {}
}

const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const node = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) {
    if (k === 'text') node.textContent = attrs[k];
    else node.setAttribute(k, attrs[k]);
  }
  // An explicit fill on text must beat the theme rule `svg text { fill: var(--text) }`;
  // set it after the loop so a `style` key cannot overwrite it.
  if (tag === 'text' && attrs && attrs.fill) node.style.fill = attrs.fill;
  if (parent) parent.appendChild(node);
  return node;
}

const PCOLORS = ['var(--p0)', 'var(--p1)', 'var(--p2)', 'var(--p3)', 'var(--p4)', 'var(--p5)'];

const sleep = ms => new Promise(r => setTimeout(r, ms));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Animate an SVG element from (x1,y1) to (x2,y2) via transform. */
function tween(node, x1, y1, x2, y2, ms) {
  ms = reducedMotion ? 1 : (ms || 600);
  return new Promise(resolve => {
    const t0 = performance.now();
    function frame(t) {
      const p = Math.min(1, (t - t0) / ms);
      const e = p < .5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      node.setAttribute('transform', `translate(${x1 + (x2 - x1) * e},${y1 + (y2 - y1) * e})`);
      if (p < 1) requestAnimationFrame(frame); else resolve();
    }
    requestAnimationFrame(frame);
  });
}

function logLine(logEl, html, highlight) {
  const d = document.createElement('div');
  d.innerHTML = html;
  if (highlight) d.className = 'hl';
  logEl.prepend(d);
  while (logEl.children.length > 40) logEl.lastChild.remove();
}

/* Terminal helper: appends a "$ prompt cmd" line plus optional output lines, then scrolls down. */
function termRun(termEl, cmd, outLines, isErr) {
  const empty = termEl.querySelector('.term-empty');
  if (empty) empty.remove();
  const cmdLine = document.createElement('div');
  cmdLine.className = 'term-line';
  cmdLine.innerHTML = `<span class="prompt">$</span> <span class="cmd">${cmd}</span>`;
  termEl.appendChild(cmdLine);
  (outLines || []).forEach(line => {
    const d = document.createElement('div');
    d.className = 'term-line ' + (isErr ? 'err' : 'out');
    d.innerHTML = line;
    termEl.appendChild(d);
  });
  while (termEl.children.length > 160) termEl.removeChild(termEl.firstChild);
  termEl.scrollTop = termEl.scrollHeight;
}

/*
 * Self-quiz. questions: [{ q, options: [...], answer: index, explain }]
 */
function renderQuiz(container, questions) {
  const root = typeof container === 'string' ? document.getElementById(container) : container;
  root.classList.add('quiz');
  let answered = 0, correct = 0;
  const score = document.createElement('div');
  score.className = 'quiz-score';

  function updateScore() {
    const done = answered === questions.length;
    score.innerHTML = `<span><b>${correct}</b> / ${questions.length} correct` +
      (done ? ` — ${correct === questions.length ? 'excellent, you’ve got this topic.' :
        correct >= questions.length - 1 ? 'nearly there. Review the explanation you missed.' :
        'revisit the visualization above and try again.'}` : ` · ${questions.length - answered} left`) +
      `</span>`;
    const reset = document.createElement('button');
    reset.textContent = 'Reset quiz';
    reset.onclick = build;
    score.appendChild(reset);
  }

  function build() {
    answered = 0; correct = 0;
    root.innerHTML = '';
    questions.forEach((item, qi) => {
      const box = document.createElement('div');
      box.className = 'q';
      box.innerHTML = `<div class="q-title">${qi + 1}. ${item.q}</div>`;
      const opts = document.createElement('div');
      opts.className = 'q-opts';
      const explain = document.createElement('div');
      explain.className = 'q-explain';
      item.options.forEach((text, oi) => {
        const b = document.createElement('button');
        b.className = 'q-opt';
        b.innerHTML = `<span class="k">${String.fromCharCode(65 + oi)}</span><span>${text}</span>`;
        b.onclick = () => {
          const buttons = opts.querySelectorAll('button');
          buttons.forEach(x => x.disabled = true);
          buttons[item.answer].classList.add('correct');
          const ok = oi === item.answer;
          if (!ok) b.classList.add('wrong');
          answered++; if (ok) correct++;
          explain.innerHTML = `<b>${ok ? 'Correct.' : 'Not quite.'}</b> ${item.explain}`;
          explain.classList.add('show');
          updateScore();
        };
        opts.appendChild(b);
      });
      box.appendChild(opts);
      box.appendChild(explain);
      root.appendChild(box);
    });
    root.appendChild(score);
    updateScore();
  }
  build();
}

/* Module 1 — softmax over logits with an optional temperature (attention rows, next-token sampling). */
function softmax(logits, temp) {
  const t = temp === undefined ? 1 : temp;
  const finite = logits.filter(Number.isFinite);
  const max = Math.max(...finite);
  const ex = logits.map(z => Number.isFinite(z) ? Math.exp((z - max) / t) : 0);
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / sum);
}

/* Make an SVG node keyboard-operable: role=button, tabindex, aria-label, click + Enter/Space. */
function clickable(node, fn, label) {
  node.setAttribute('role', 'button');
  node.setAttribute('tabindex', '0');
  if (label) node.setAttribute('aria-label', label);
  node.style.cursor = 'pointer';
  node.addEventListener('click', fn);
  node.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(e); }
  });
  return node;
}

/* Wire a .seg group whose buttons carry data-v: keeps aria-pressed in sync and calls fn(value). */
function bindSeg(id, fn) {
  const seg = document.getElementById(id);
  const buttons = seg.querySelectorAll('button');
  buttons.forEach(b => b.addEventListener('click', () => {
    buttons.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    fn(b.dataset.v);
  }));
}

/* Wire an on/off button: flips aria-pressed and calls fn(isOn). */
function bindToggle(id, fn) {
  const b = document.getElementById(id);
  b.addEventListener('click', () => {
    const on = b.getAttribute('aria-pressed') !== 'true';
    b.setAttribute('aria-pressed', String(on));
    fn(on);
  });
}

/* Module 2 — HTML escaping, money, syntax-coloured JSON for payload panes, SVG text wrapping. */
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function money(x, dp) {
  const d = dp === undefined ? (Math.abs(x) < 10 ? 2 : 0) : dp;
  return '$' + x.toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d });
}

/* Render a JS value as indented JSON HTML. Keys listed in hl get a highlight; short values stay on one line. */
function jsonHtml(v, hl, ind) {
  hl = hl || []; ind = ind || '';
  const key = k => `<span class="j-key${hl.includes(k) ? ' j-hl' : ''}">${esc(JSON.stringify(k))}</span>`;
  const leaf = x => x === null || typeof x !== 'object'
    ? `<span class="${typeof x === 'string' ? 'j-str' : 'j-lit'}">${esc(JSON.stringify(x))}</span>` : null;
  const inline = x => leaf(x) ?? (Array.isArray(x) ? '[' + x.map(inline).join(', ') + ']'
    : '{' + Object.keys(x).map(k => key(k) + ': ' + inline(x[k])).join(', ') + '}');
  if (leaf(v) !== null) return leaf(v);
  if (JSON.stringify(v).length <= 60 && !Object.keys(v).some(k => hl.includes(k))) return inline(v);
  const pad = ind + '  ';
  if (Array.isArray(v)) return '[\n' + v.map(x => pad + jsonHtml(x, hl, pad)).join(',\n') + '\n' + ind + ']';
  return '{\n' + Object.keys(v).map(k => pad + key(k) + ': ' + jsonHtml(v[k], hl, pad)).join(',\n') + '\n' + ind + '}';
}

/* Split text into lines of at most n characters on word boundaries (SVG text does not wrap). */
function wrapText(s, n) {
  const out = []; let line = '';
  String(s).split(' ').forEach(w => {
    if (line && (line + ' ' + w).length > n) { out.push(line); line = w; } else line = line ? line + ' ' + w : w;
  });
  if (line) out.push(line);
  return out;
}

/* Module 3 — ranked-list retrieval metrics. rel: array of booleans in rank order (true = relevant). */
function precisionAtK(rel, k) { return k ? rel.slice(0, k).filter(Boolean).length / k : 0; }
function recallAtK(rel, k, totalRelevant) { return totalRelevant ? rel.slice(0, k).filter(Boolean).length / totalRelevant : 0; }
function reciprocalRank(rel, k) { const i = rel.slice(0, k === undefined ? rel.length : k).indexOf(true); return i < 0 ? 0 : 1 / (i + 1); }
