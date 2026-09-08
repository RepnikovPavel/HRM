import { defs } from './registry.js';

let DATA = { runs: [] };
const listeners = new Set();

export function getData() { return DATA; }

export function onData(fn) {
  listeners.add(fn);
  if (DATA.runs.length) fn(DATA);
}

export function ctxFor(tile) {
  return {
    state: tile.state || {},
    setState(patch) {
      tile.state = Object.assign({}, tile.state, patch);
      saveLayout();
    },
  };
}

let uidCounter = 1;
function uid() { return 'w' + (uidCounter++) + Date.now().toString(36); }

const board = document.getElementById('board');
let layout = null;
let saveTimer = 0;

function saveLayout() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const tiles = [...board.querySelectorAll('.tile')].map(el => ({
      id: el.dataset.id, type: el.dataset.type,
      x: parseFloat(el.style.left), y: parseFloat(el.style.top),
      w: el.offsetWidth, h: el.offsetHeight,
      state: el._tile.state || {},
    }));
    localStorage.setItem('mviz_layout', JSON.stringify(tiles));
  }, 200);
}

let zTop = 10;

function addTile(type, pos) {
  const def = defs.find(d => d.type === type);
  if (!def) return;
  const tile = {
    id: uid(), type,
    x: pos && pos.x != null ? pos.x : 40 + board.querySelectorAll('.tile').length * 30,
    y: pos && pos.y != null ? pos.y : 40 + board.querySelectorAll('.tile').length * 30,
    w: pos && pos.w || def.defaultSize.w,
    h: pos && pos.h || def.defaultSize.h,
    state: pos && pos.state || def.defaultState(),
  };
  const el = document.createElement('div');
  el.className = 'tile';
  el.dataset.id = tile.id;
  el.dataset.type = type;
  el.style.left = tile.x + 'px';
  el.style.top = tile.y + 'px';
  el.style.width = tile.w + 'px';
  el.style.height = tile.h + 'px';
  el.style.zIndex = ++zTop;
  el._tile = tile;

  const head = document.createElement('div');
  head.className = 'tile-head';
  const ttl = document.createElement('span');
  ttl.className = 'ttl';
  ttl.textContent = def.title;
  const xBtn = document.createElement('button');
  xBtn.className = 'x';
  xBtn.textContent = '✕';
  xBtn.title = 'закрыть';
  xBtn.addEventListener('click', () => { el.remove(); saveLayout(); });
  head.append(ttl, xBtn);

  const body = document.createElement('div');
  body.className = 'tile-body';

  const rs = document.createElement('div');
  rs.className = 'tile-rs';

  el.append(head, body, rs);
  board.appendChild(el);

  el.addEventListener('pointerdown', () => {
    el.style.zIndex = ++zTop;
    for (const t of board.querySelectorAll('.tile')) t.classList.remove('focus');
    el.classList.add('focus');
  });

  head.addEventListener('pointerdown', e => {
    if (e.target === xBtn) return;
    const sx = e.clientX - tile.x, sy = e.clientY - tile.y;
    head.setPointerCapture(e.pointerId);
    const mv = ev => {
      tile.x = Math.max(0, ev.clientX - sx);
      tile.y = Math.max(0, ev.clientY - sy);
      el.style.left = tile.x + 'px';
      el.style.top = tile.y + 'px';
    };
    const up = () => {
      head.removeEventListener('pointermove', mv);
      head.removeEventListener('pointerup', up);
      saveLayout();
    };
    head.addEventListener('pointermove', mv);
    head.addEventListener('pointerup', up);
  });

  rs.addEventListener('pointerdown', e => {
    e.stopPropagation();
    const sw = el.offsetWidth, sh = el.offsetHeight;
    const sx = e.clientX, sy = e.clientY;
    rs.setPointerCapture(e.pointerId);
    const mv = ev => {
      el.style.width = Math.max(260, sw + ev.clientX - sx) + 'px';
      el.style.height = Math.max(180, sh + ev.clientY - sy) + 'px';
    };
    const up = () => {
      rs.removeEventListener('pointermove', mv);
      rs.removeEventListener('pointerup', up);
      tile.w = el.offsetWidth; tile.h = el.offsetHeight;
      saveLayout();
    };
    rs.addEventListener('pointermove', mv);
    rs.addEventListener('pointerup', up);
  });

  def.mount(body, ctxFor(tile));
  saveLayout();
  return el;
}

const wpickerBtn = document.getElementById('wpicker-btn');
const wpickerMenu = document.getElementById('wpicker-menu');
wpickerBtn.addEventListener('click', () => {
  if (wpickerMenu.style.display === 'block') {
    wpickerMenu.style.display = 'none';
    return;
  }
  wpickerMenu.innerHTML = '';
  for (const d of defs) {
    const b = document.createElement('button');
    b.innerHTML = d.title + '<span class="hint">' + d.hint + '</span>';
    b.addEventListener('click', () => {
      wpickerMenu.style.display = 'none';
      addTile(d.type);
    });
    wpickerMenu.appendChild(b);
  }
  wpickerMenu.style.display = 'block';
});
window.addEventListener('pointerdown', e => {
  if (!wpickerMenu.contains(e.target) && e.target !== wpickerBtn)
    wpickerMenu.style.display = 'none';
}, true);

async function refresh() {
  try {
    const resp = await fetch('/api/series');
    DATA = await resp.json();
    for (const fn of listeners) fn(DATA);
  } catch (e) {}
}

try {
  layout = JSON.parse(localStorage.getItem('mviz_layout') || 'null');
} catch (e) { layout = null; }

if (layout && layout.length) {
  for (const t of layout) addTile(t.type, t);
} else {
  addTile('lines', { x: 20, y: 20, w: 660, h: 430 });
  addTile('candles', { x: 700, y: 20, w: 720, h: 500 });
}

refresh();
setInterval(refresh, 60000);
