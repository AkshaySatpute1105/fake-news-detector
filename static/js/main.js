(() => {
  const $ = id => document.getElementById(id);
  const text = $('articleText'), detectBtn = $('detectBtn'), clearBtn = $('clearBtn');
  const errorBox = $('errorBox');
  const panels = { empty: $('emptyState'), loader: $('loader'), result: $('result') };
  const HISTORY_KEY = 'verinews_history';
  let history = [];
  try { history = JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || []; } catch (e) {}

  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const show = name => Object.entries(panels).forEach(([k, el]) => el.classList.toggle('d-none', k !== name));
  const showError = msg => { errorBox.textContent = msg; errorBox.classList.remove('d-none'); };
  const hideError = () => errorBox.classList.add('d-none');

  text.addEventListener('input', () => {
    const t = text.value.trim();
    $('wordCount').textContent = t ? t.split(/\s+/).length : 0;
    $('charCount').textContent = `${text.value.length} / 50000`;
    hideError();
  });

  detectBtn.addEventListener('click', async () => {
    hideError();
    const article = text.value.trim();
    if (!article) return showError('Please paste a news article first.');
    if (article.split(/\s+/).length < 20) return showError('Article is too short. Enter at least 20 words.');

    detectBtn.disabled = true;
    detectBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analyzing...';
    $('details').classList.add('d-none');
    show('loader');
    const started = Date.now();
    try {
      const res = await fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: article })
      });
      const data = await res.json().catch(() => ({ error: 'Unexpected server response.' }));
      // keep the loader visible briefly so the animation is noticeable
      await new Promise(r => setTimeout(r, Math.max(0, 600 - (Date.now() - started))));
      if (!res.ok) { show('empty'); return showError(data.error || 'Request failed.'); }
      render(data);
      addHistory(article, data);
    } catch (err) {
      show('empty');
      showError('Could not reach the server. Check that Flask is running.');
    } finally {
      detectBtn.disabled = false;
      detectBtn.innerHTML = '<i class="bi bi-search"></i> Detect News';
    }
  });

  clearBtn.addEventListener('click', () => {
    text.value = ''; text.dispatchEvent(new Event('input'));
    hideError(); show('empty'); $('details').classList.add('d-none'); text.focus();
  });

  function render(d) {
    const real = d.label === 'Real News';
    $('verdict').className = 'verdict ' + (real ? 'real' : 'fake');
    $('verdictIcon').className = 'bi ' + (real ? 'bi-patch-check-fill' : 'bi-exclamation-octagon-fill');
    $('verdictText').textContent = d.label;
    $('confText').textContent = d.confidence.toFixed(2) + '%';
    $('pFake').textContent = d.probabilities.fake.toFixed(2);
    $('pReal').textContent = d.probabilities.real.toFixed(2);
    const bar = $('confBar');
    bar.className = 'progress-bar ' + (real ? 'real' : 'fake');
    bar.style.width = '0%';
    show('result');
    requestAnimationFrame(() => requestAnimationFrame(() => bar.style.width = d.confidence + '%'));

    // keywords
    const max = Math.max(...d.keywords.map(k => Math.abs(k.weight)), 1e-9);
    $('keywords').innerHTML = d.keywords.map(k =>
      `<span class="kw ${k.weight > 0 ? 'kw-real' : 'kw-fake'}" title="stem: ${esc(k.stem)}">${esc(k.word)}</span>`).join('')
      || '<span class="text-muted small">No known keywords found.</span>';
    $('kwBars').innerHTML = d.keywords.map(k => {
      const cls = k.weight > 0 ? 'real' : 'fake';
      return `<div class="kw-row"><span class="name">${esc(k.word)}</span>
        <div class="track"><div class="fill ${cls}" style="width:${Math.abs(k.weight) / max * 100}%"></div></div>
        <span class="text-muted">${k.pushes}</span></div>`;
    }).join('');

    // preprocessing steps
    const p = d.preprocessing, c = p.counts;
    const tok = arr => arr.map(t => `<span class="token">${esc(t)}</span>`).join('');
    const items = [
      ['Original input', `${p.original_words} words`, `<div class="step-text">Your pasted text (${p.original_words} words)</div>`],
      ['1. Lowercasing', '', `<div class="step-text">${esc(p.lowercased)}</div>`],
      ['2. Punctuation removal', '', `<div class="step-text">${esc(p.no_punctuation)}</div>`],
      ['3. Tokenization', `${c.tokens} tokens`, `<div class="step-text">${tok(p.tokens)}</div>`],
      ['4. Stop-word removal', `${c.no_stopwords} left`, `<div class="step-text">${tok(p.no_stopwords)}</div>`],
      ['5. Porter stemming', `${c.stemmed} stems`, `<div class="step-text">${tok(p.stemmed)}</div>`],
    ];
    $('stepsAcc').innerHTML = items.map((it, i) => `
      <div class="accordion-item">
        <h2 class="accordion-header"><button class="accordion-button ${i ? 'collapsed' : ''}" type="button" data-bs-toggle="collapse" data-bs-target="#s${i}">
          ${it[0]}<span class="ms-auto me-2 small text-muted fw-normal">${it[1]}</span></button></h2>
        <div id="s${i}" class="accordion-collapse collapse ${i ? '' : 'show'}" data-bs-parent="#stepsAcc"><div class="accordion-body">${it[2]}</div></div>
      </div>`).join('');
    $('details').classList.remove('d-none');
  }

  function addHistory(article, d) {
    history.unshift({
      snippet: article.slice(0, 90) + (article.length > 90 ? '…' : ''),
      label: d.label, confidence: d.confidence,
      time: new Date().toLocaleTimeString()
    });
    history = history.slice(0, 50);
    try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch (e) {}
    renderHistory();
  }

  function renderHistory() {
    $('historyEmpty').classList.toggle('d-none', history.length > 0);
    $('historyBody').innerHTML = history.map((h, i) => `
      <tr><td>${history.length - i}</td><td class="small">${esc(h.snippet)}</td>
      <td><span class="badge ${h.label === 'Real News' ? 'badge-real' : 'badge-fake'}">${h.label}</span></td>
      <td>${h.confidence.toFixed(2)}%</td><td class="small text-muted">${h.time}</td></tr>`).join('');
  }

  $('clearHistory').addEventListener('click', () => {
    history = []; sessionStorage.removeItem(HISTORY_KEY); renderHistory();
  });
  renderHistory();
})();
