const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const BASE = process.env.SATQUERY_URL || 'http://127.0.0.1:8765/';
const FIX = __dirname + '/fixtures/';
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => { if (!/fonts\.googleapis\.com/.test(e.message)) errors.push('jsdomError: ' + e.message); });  // web fonts are not loadable in CI sandboxes
vc.on('error', e => errors.push('console.error: ' + e));
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitFor(win, fn, ms = 15000) { const t = Date.now(); while (Date.now() - t < ms) { try { if (fn()) return true; } catch (e) {} await sleep(100); } return false; }
function check(name, ok, extra = '') { console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? ' — ' + extra : '')); if (!ok) process.exitCode = 1; }

(async () => {
  const dom = await JSDOM.fromURL(BASE, { runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(win) { win.fetch = (u, o) => fetch(new URL(u, BASE), o); } });
  const win = dom.window, doc = win.document;
  check('API pill online', await waitFor(win, () => doc.querySelector('#apiPill').classList.contains('ok')), doc.querySelector('#apiText').textContent);

  const file = (p, type) => new win.File([fs.readFileSync(p)], p.split('/').pop(), { type });
  const setFile = async (slotIdx, f) => {
    const input = doc.querySelectorAll('#slots input[type=file]')[slotIdx];
    Object.defineProperty(input, 'files', { value: [f], configurable: true });
    input.dispatchEvent(new win.Event('change'));
    await waitFor(win, () => { const d = doc.querySelectorAll('#slots .drop')[slotIdx]; return d && d.classList.contains('filled') && d.textContent.includes(f.name); });
  };
  // bitemporal flow
  doc.querySelector('#modeSeg button[data-mode=bitemporal]').click();
  check('run disabled before upload', doc.querySelector('#runBtn').disabled === true);
  await setFile(0, file(FIX + 'a.png', 'image/png')); await setFile(1, file(FIX + 'b.png', 'image/png'));
  check('run enabled after two uploads', doc.querySelector('#runBtn').disabled === false);
  doc.querySelector('#query').value = 'What changed between these two dates, and where did the change occur?';
  doc.querySelector('#runBtn').click();
  check('answer rendered', await waitFor(win, () => doc.querySelector('.answer')), (doc.querySelector('.answer') || {}).textContent);
  const tabs = [...doc.querySelectorAll('.tabs button')].map(b => b.textContent);
  check('all tabs present', ['Overview', 'Evidence', 'Quality', 'Sensitivity', 'Layers', 'Methodology', 'Report'].every(t => tabs.includes(t)), tabs.join(', '));
  const goTab = async name => { [...doc.querySelectorAll('.tabs button')].find(b => b.textContent === name).click(); await sleep(50); };
  await goTab('Evidence');
  const cells = doc.querySelectorAll('svg rect.cell');
  check('grid cells rendered (4x4)', cells.length === 16, String(cells.length));
  const clickCell = id => [...doc.querySelectorAll('svg rect.cell')].find(r => r.querySelector('title').textContent.startsWith(id)).dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  clickCell('grid_2'); await sleep(50);
  check('inspector shows region', /grid_2/.test(doc.querySelector('#work').textContent) && /analysed pixels/.test(doc.querySelector('#work').textContent));
  [...doc.querySelectorAll('.btn.small')].find(b => b.textContent === 'Zoom to cell').click(); await sleep(50);
  check('zoom applies transform', /scale\(/.test(doc.querySelector('.viewer-inner').style.transform), doc.querySelector('.viewer-inner').style.transform);
  for (const t of ['Quality', 'Sensitivity', 'Layers', 'Methodology', 'Report', 'Overview']) { await goTab(t); check('tab renders: ' + t, doc.querySelector('#work').textContent.length > 200); }
  await goTab('Layers'); [...doc.querySelectorAll('.layer-btns button')].find(b => b.textContent === 'Before / after slider').click(); await sleep(50);
  check('before/after slider renders', !!doc.querySelector('.compare'));
  await goTab('Report');
  const dls = [...doc.querySelectorAll('.dl button')].map(b => b.textContent);
  check('export buttons', ['Download JSON (full result)', 'Markdown', 'Regions CSV', 'Sensitivity CSV', 'Open printable report'].every(x => dls.includes(x)), dls.join(' | '));

  // capability gate renders explanation
  doc.querySelector('#modeSeg button[data-mode=single]').click();
  await setFile(0, file(FIX + 'a.png', 'image/png'));
  doc.querySelector('#query').value = 'Calculate NDVI for this image';
  doc.querySelector('#runBtn').click();
  check('unavailable capability explained', await waitFor(win, () => /What is missing/.test(doc.querySelector('#work').textContent)) && /multispectral bands/.test(doc.querySelector('#work').textContent));
  check('no answer card invented for blocked query', !doc.querySelector('.answer'));

  // empty query
  doc.querySelector('#query').value = ''; doc.querySelector('#runBtn').click(); await sleep(100);
  check('empty query message', /Enter a question/.test(doc.querySelector('#work').textContent));

  // GeoTIFF check inputs
  await setFile(0, file(FIX + 'g.tif', 'image/tiff'));
  doc.querySelector('#checkBtn').click();
  check('geotiff metadata shown', await waitFor(win, () => /EPSG:32644/.test(doc.querySelector('#work').textContent)));

  // STAC
  doc.querySelector('.navtabs button[data-view=stac]').click();
  const ta = doc.querySelector('#work textarea');
  ta.value = JSON.stringify({ type: 'Feature', id: 'X1', collection: 'sentinel-2-l2a', bbox: [78, 17, 79, 18], properties: { datetime: '2024-01-01T00:00:00Z', 'eo:cloud_cover': 3 }, assets: { t: { href: 'https://x.invalid/t.jpg', type: 'image/jpeg' } } });
  [...doc.querySelectorAll('#work .btn.primary')].find(b => b.textContent === 'Inspect').click();
  check('stac inspected', await waitFor(win, () => /Nothing was fetched/.test(doc.querySelector('#work').textContent)));

  doc.querySelector('.navtabs button[data-view=analysis]').click();
  console.log(errors.length ? 'ERRORS:\n' + errors.join('\n') : 'no console/runtime errors');
  if (errors.length) process.exitCode = 1;
  win.close();
})();
