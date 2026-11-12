import './style.css';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';

hljs.registerLanguage('python', python);

document.querySelectorAll('pre code').forEach(function (el) { hljs.highlightElement(el); });

document.querySelectorAll('.copy').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var code = btn.closest('.card').querySelector('code');
    var text = code.innerText;
    function done(ok) { btn.textContent = ok ? 'Tersalin' : 'Pilih teks'; btn.classList.toggle('ok', ok); setTimeout(function () { btn.textContent = 'Salin'; btn.classList.remove('ok'); }, 1400); }
    function fallback() { var r = document.createRange(); r.selectNodeContents(code); var s = getSelection(); s.removeAllRanges(); s.addRange(r); done(false); }
    try { navigator.clipboard.writeText(text).then(function () { done(true); }, fallback); } catch (e) { fallback(); }
  });
});

var q = document.getElementById('q'), count = document.getElementById('count'), empty = document.getElementById('empty');
var panels = document.querySelectorAll('.panel'), navs = document.querySelectorAll('nav.idx'), tabs = document.querySelectorAll('.tab');
var active = 'visualisasi';
document.querySelectorAll('.card').forEach(function (c) { c.dataset.search = (c.dataset.search + ' ' + c.innerText).toLowerCase(); });

function filter() {
  var panel = document.getElementById('panel-' + active);
  var cards = panel.querySelectorAll('.card');
  var terms = q.value.toLowerCase().trim().split(/\s+/).filter(Boolean), shown = 0;
  cards.forEach(function (c) { var ok = terms.every(function (t) { return c.dataset.search.indexOf(t) !== -1; }); c.classList.toggle('hide', !ok); if (ok) shown++; });
  panel.querySelectorAll('.sec').forEach(function (s) { s.classList.toggle('hide', !s.querySelector('.card:not(.hide)')); });
  count.textContent = terms.length ? shown + ' cocok' : cards.length + ' kartu';
  empty.style.display = shown ? 'none' : 'block';
}
function setTab(name) {
  if (!document.getElementById('panel-' + name)) name = 'visualisasi';
  active = name;
  panels.forEach(function (p) { p.hidden = p.dataset.tab !== name; });
  navs.forEach(function (n) { n.hidden = n.dataset.tab !== name; });
  tabs.forEach(function (t) { var on = t.dataset.tab === name; t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1; });
  q.placeholder = name === 'cleaning' ? 'Cari: fillna, outlier, duplikat, to_numeric…' : 'Cari: boxplot, heatmap, unstack, <br>…';
  try { localStorage.setItem('cheat-tab', name); } catch (e) {}
  filter();
}
function route() {
  var h = decodeURIComponent(location.hash.slice(1));
  if (document.getElementById('panel-' + h)) { setTab(h); window.scrollTo(0, 0); return; }
  var el = h && document.getElementById(h);
  var p = el && el.closest('.panel');
  if (p) { setTab(p.dataset.tab); el.scrollIntoView(); return; }
  var saved = null; try { saved = localStorage.getItem('cheat-tab'); } catch (e) {}
  setTab(saved || 'visualisasi');
}
tabs.forEach(function (t, i) {
  t.addEventListener('click', function () { if (location.hash === '#' + t.dataset.tab) route(); else location.hash = t.dataset.tab; });
  t.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    var n = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); n.click();
  });
});
window.addEventListener('hashchange', route);
// A deep link (#dist, #c-out, ...) is scrolled again once fonts and layout have settled.
window.addEventListener('load', function () {
  var h = decodeURIComponent(location.hash.slice(1));
  var el = h && !document.getElementById('panel-' + h) && document.getElementById(h);
  if (el && el.closest('.panel')) el.scrollIntoView({ behavior: 'instant', block: 'start' });
});
q.addEventListener('input', filter);
document.addEventListener('keydown', function (e) {
  if (e.key === '/' && document.activeElement !== q) { e.preventDefault(); q.focus(); }
  if (e.key === 'Escape' && document.activeElement === q) { q.value = ''; filter(); q.blur(); }
});
route();
