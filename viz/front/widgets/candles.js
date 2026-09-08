import { onData } from '../shell/shell.js';

const TF_LADDER = [
  ['15м', 9e5], ['30м', 18e5], ['1ч', 36e5], ['2ч', 72e5], ['4ч', 144e5],
];
const TIME_STEPS = [9e5, 18e5, 36e5, 72e5, 144e5, 288e5, 432e5, 864e5,
  1728e5, 3456e5, 6912e5];
const RIGHT_OFF = 0.1;
const MAGNET_PX = 8;
const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
const FIB_COLORS = ['#8b9bb4', '#2ea88a', '#4f8ef7', '#b46ff7', '#e8c14d',
  '#f7a24f', '#e5484d'];
const TOOLBAR_W = 34;

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function fmtPrice(v, step) {
  const dec = step >= 1 ? 2 :
    Math.min(8, Math.max(0, Math.ceil(-Math.log10(step))));
  return v.toLocaleString('ru-RU',
    { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function fmtVol(v) {
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(2) + 'K';
  if (v >= 1) return v.toFixed(0);
  return v.toLocaleString('ru-RU', { maximumFractionDigits: 4 });
}

function fmtT(ms) {
  const h = ms / 36e5;
  if (h < 1) return Math.round(ms / 6e4) + 'm';
  return (h < 10 ? h.toFixed(1) : h.toFixed(0)) + 'h';
}

function fmtDurH(ms) {
  const h = Math.abs(ms) / 36e5;
  return (h < 10 ? h.toFixed(1) : h.toFixed(0)) + 'ч';
}

function fmtSigned(v) {
  return (v >= 0 ? '+' : '') + v.toFixed(2);
}

function cssVar(el, name, fallback) {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

let cssInjected = false;

function injectCss() {
  if (cssInjected) return;
  cssInjected = true;
  const st = document.createElement('style');
  st.textContent = `
    .candles-hdr { display: flex; align-items: center; gap: 4px;
      padding: 3px 6px; background: var(--bg2); font-size: 12px;
      border-bottom: 1px solid var(--border); position: relative; z-index: 8; }
    .candles-sel { background: var(--bg3); color: var(--fg); font-size: 12px;
      border: 1px solid var(--border); border-radius: 5px;
      padding: 2px 4px; max-width: 220px; }
    .candles-root { position: absolute; left: 0; right: 0; top: 26px;
      bottom: 0; overflow: hidden; }
    .candles-root canvas { display: block; width: 100%; height: 100%;
      cursor: crosshair; touch-action: none; }
    .candles-tf { display: flex; gap: 2px; margin-left: 4px; }
    .candles-tf button { border: 0; background: none; color: var(--fg2);
      cursor: pointer; padding: 3px 6px; border-radius: 5px; font-size: 12px; }
    .candles-tf button:hover { background: var(--bg3); color: var(--fg); }
    .candles-tf button.active { background: var(--accent); color: #fff; }
    .candles-mag { border: 0; background: none; color: var(--fg2);
      cursor: pointer; padding: 3px 6px; border-radius: 5px; margin-left: 2px;
      display: inline-flex; align-items: center; }
    .candles-mag:hover { background: var(--bg3); color: var(--fg); }
    .candles-mag.active { background: var(--accent); color: #fff; }
    .candles-golive { position: absolute; right: 76px; bottom: 32px; z-index: 5;
      background: var(--bg3); color: var(--accent);
      border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
      font-size: 12px; cursor: pointer; user-select: none; display: none; }
    .candles-golive:hover { background: var(--accent); color: #fff; }
    .candles-toolbar { position: absolute; left: 0; top: 0; bottom: 0;
      z-index: 6; width: 34px; box-sizing: border-box; display: flex;
      flex-direction: column; align-items: center; gap: 2px;
      background: var(--bg2); border-right: 1px solid var(--border);
      padding: 4px 0; }
    .candles-toolbar button { border: 0; background: none; color: var(--fg2);
      cursor: pointer; width: 26px; height: 26px; border-radius: 5px;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-style: italic; }
    .candles-toolbar button svg { display: block; }
    .candles-toolbar button:hover { background: var(--bg3); color: var(--fg); }
    .candles-toolbar button.active { background: var(--accent); color: #fff; }
    .candles-indmenu { position: absolute; left: 38px; top: 8px; z-index: 7;
      background: var(--bg2); border: 1px solid var(--border);
      border-radius: 7px; padding: 6px 8px; display: flex;
      flex-direction: column; gap: 4px; }
    .candles-indmenu label { display: flex; gap: 6px; align-items: center;
      color: var(--fg); font-size: 12px; cursor: pointer; white-space: nowrap; }
    .candles-indmenu input { accent-color: var(--accent); }
  `;
  document.head.appendChild(st);
}

export const def = {
  type: 'candles',
  title: 'Свечи: exact accuracy',
  hint: 'OHLC по таймфрейму, выбор модели',
  defaultSize: { w: 720, h: 500 },
  defaultState: () => ({ tf: 36e5, run: 0, magnet: false, drawings: [],
    indicators: [] }),

  mount(body, ctx) {
    injectCss();
    body.style.position = 'relative';
    const hdr = document.createElement('div');
    hdr.className = 'candles-hdr';
    const sel = document.createElement('select');
    sel.className = 'candles-sel';
    hdr.appendChild(sel);
    body.appendChild(hdr);
    const root = document.createElement('div');
    root.className = 'candles-root';
    const canvas = document.createElement('canvas');
    const goLiveEl = document.createElement('div');
    goLiveEl.className = 'candles-golive';
    goLiveEl.textContent = '►|';
    goLiveEl.title = 'к последней свече';
    root.append(canvas, goLiveEl);
    body.appendChild(root);
    const g2d = canvas.getContext('2d');

    const S = {
      w: 0, h: 0,
      runs: [],
      runIdx: ctx.state.run || 0,
      viewFrom: 0, viewTo: 0,
      stepMs: ctx.state.tf || 36e5,
      init: false,
      followTail: true,
      mouse: null,
      hover: false,
      magnet: !!ctx.state.magnet,
      yAuto: true, yScale: 1, yShift: 0,
      dragging: false, dragKind: 'pan', dragX: 0, dragFrom: 0, dragY: 0,
      rafPending: false,
      cacheKey: null,
      entry: null,
      tfButtons: new Map(),
      tool: null,
      fsm: 'idle',
      placing: null,
      dragAnchor: null,
      selected: -1,
      temp: null,
      drawingsByRun: (ctx.state.drawingsByRun &&
        typeof ctx.state.drawingsByRun === 'object') ? ctx.state.drawingsByRun
        : (Array.isArray(ctx.state.drawings) && ctx.state.drawings.length
          ? { [ctx.state.run || 0]: ctx.state.drawings } : {}),
      drawings: [],
      indicators: Array.isArray(ctx.state.indicators) ?
        ctx.state.indicators : [],
      indOpen: false,
      yRange: { min: 0, max: 1 },
      buf: { cap: 0 },
      toolButtons: new Map(),
      pairBuf: [null, null],
      colors: {
        bg: '#141b28', bg2: '#1b2434', bg3: '#232e42', fg: '#d5dbe7',
        fg2: '#7d8699', up: '#2ea88a', down: '#e5484d',
        accent: '#4f8ef7', border: '#2b3650',
      },
    };
    for (const k of Object.keys(S.colors)) {
      S.colors[k] = cssVar(body, '--' + k, S.colors[k]);
    }
    S.drawings = (S.drawingsByRun[S.runIdx] ||= []);

    function layout() {
      const tbW = TOOLBAR_W;
      const priceAxisW = 64;
      const timeAxisH = 22;
      const plotW = Math.max(0, S.w - tbW - priceAxisW);
      const plotH = Math.max(0, S.h - timeAxisH);
      const hasInd = S.indicators.includes('rsi') ||
        S.indicators.includes('macd');
      const volH = 0;
      const indH = hasInd ? Math.round(plotH * 0.25) : 0;
      const priceH = Math.max(0, plotH - indH);
      return { tbW, priceAxisW, timeAxisH, plotW, plotH, volH, priceH, indH };
    }

    function onResize() {
      const r = root.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      S.w = Math.max(0, r.width);
      S.h = Math.max(0, r.height);
      canvas.width = Math.round(S.w * dpr);
      canvas.height = Math.round(S.h * dpr);
      g2d.setTransform(dpr, 0, 0, dpr, 0, 0);
      requestDraw();
    }
    const resizeObs = new ResizeObserver(onResize);
    resizeObs.observe(root);

    const toolbar = document.createElement('div');
    toolbar.className = 'candles-toolbar';
    root.appendChild(toolbar);
    const svgWrap = (inner) => '<svg width="13" height="13" viewBox="0 0 16 16"' +
      ' fill="none" stroke="currentColor" stroke-width="1.6"' +
      ' stroke-linecap="round" stroke-linejoin="round">' + inner + '</svg>';
    const TOOL_DEFS = [
      ['cursor', 'курсор', '<path d="M4 2l9 6-4 1-2 4z"/>'],
      ['measure', 'линейка (или shift + 2 клика)',
        '<path d="M2.5 13.5L13.5 2.5M5 12l1.6 1.6M8 9l1.6 1.6M11 6l1.6 1.6"/>'],
      ['pen', 'карандаш',
        '<path d="M2.5 13.5c2.5-5 3 0.5 5-3.5s2.5 2 6-3.5"/>'],
      ['trend', 'тренд-линия',
        '<line x1="2.5" y1="13" x2="13.5" y2="3"/>' +
        '<circle cx="2.5" cy="13" r="1.5" fill="currentColor" stroke="none"/>' +
        '<circle cx="13.5" cy="3" r="1.5" fill="currentColor" stroke="none"/>'],
      ['hline', 'горизонталь',
        '<line x1="2" y1="8" x2="14" y2="8"/>' +
        '<circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none"/>'],
      ['fibo', 'фибоначчи',
        '<line x1="2" y1="3.5" x2="14" y2="3.5"/>' +
        '<line x1="2" y1="8" x2="14" y2="8"/>' +
        '<line x1="2" y1="12.5" x2="14" y2="12.5"/>'],
    ];
    let cursorBtn = null;
    for (const [name, title, icon] of TOOL_DEFS) {
      const b = document.createElement('button');
      b.innerHTML = svgWrap(icon);
      b.title = title;
      b.addEventListener('click', () => {
        if (name === 'cursor') setTool(null);
        else setTool(S.tool === name ? null : name);
      });
      toolbar.appendChild(b);
      if (name === 'cursor') {
        cursorBtn = b;
        b.classList.add('active');
      } else {
        S.toolButtons.set(name, b);
      }
    }
    const indBtn = document.createElement('button');
    indBtn.textContent = 'fx';
    indBtn.title = 'индикаторы';
    indBtn.addEventListener('click', () => toggleIndMenu(!S.indOpen));
    toolbar.appendChild(indBtn);
    const indMenu = document.createElement('div');
    indMenu.className = 'candles-indmenu';
    indMenu.style.display = 'none';
    root.appendChild(indMenu);
    const IND_DEFS = [
      ['sma', 'SMA (20)'], ['ema', 'EMA (50)'], ['bb', 'Bollinger (20, 2)'],
      ['rsi', 'RSI (14)'], ['macd', 'MACD (12, 26, 9)'],
    ];
    for (const [key, label] of IND_DEFS) {
      const lab = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = S.indicators.includes(key);
      cb.addEventListener('change', () => {
        const i = S.indicators.indexOf(key);
        if (cb.checked && i < 0) S.indicators.push(key);
        if (!cb.checked && i >= 0) S.indicators.splice(i, 1);
        ctx.setState({ indicators: S.indicators });
        requestDraw();
      });
      lab.append(cb, document.createTextNode(label));
      indMenu.appendChild(lab);
    }

    function toggleIndMenu(open) {
      S.indOpen = open;
      indMenu.style.display = open ? '' : 'none';
      indBtn.classList.toggle('active', open);
    }

    function onWinPointerDown(e) {
      if (!S.indOpen) return;
      if (indMenu.contains(e.target) || indBtn.contains(e.target)) return;
      toggleIndMenu(false);
    }
    window.addEventListener('pointerdown', onWinPointerDown, true);

    function setTool(name) {
      S.tool = name;
      S.placing = null;
      S.fsm = name ? 'toolActive' : 'idle';
      for (const [k, b] of S.toolButtons) b.classList.toggle('active', k === name);
      if (cursorBtn) cursorBtn.classList.toggle('active', !name);
      requestDraw();
    }

    function curRun() {
      return S.runs[S.runIdx] || null;
    }

    function getEntry() {
      const r = curRun();
      if (!r) return null;
      const key = S.runIdx + ':' + S.stepMs + ':' + r.series.length;
      if (S.cacheKey === key && S.entry) return S.entry;
      const step = S.stepMs;
      let mx = 0;
      for (const p of r.series) {
        const t = p[0] * 36e5;
        if (t > mx) mx = t;
      }
      const n = Math.floor(mx / step) + 1;
      const e = {
        t0: 0, n, tail: 0,
        o: new Float64Array(n), h: new Float64Array(n),
        l: new Float64Array(n), c: new Float64Array(n),
        nTick: new Float64Array(n),
      };
      for (const p of r.series) {
        const i = Math.floor(p[0] * 36e5 / step);
        const val = p[1];
        if (e.nTick[i] === 0) {
          e.o[i] = val;
          e.h[i] = val;
          e.l[i] = val;
        } else {
          if (val > e.h[i]) e.h[i] = val;
          if (val < e.l[i]) e.l[i] = val;
        }
        e.c[i] = val;
        e.nTick[i]++;
        const end = i * step + step;
        if (end > e.tail) e.tail = end;
      }
      S.cacheKey = key;
      S.entry = e;
      return e;
    }

    function tailEnd() {
      const e = getEntry();
      return e ? e.tail : -Infinity;
    }

    function setViewportInitial() {
      const te = tailEnd();
      if (!isFinite(te) || te <= 0) return;
      const span = Math.min(150 * S.stepMs, te);
      S.viewTo = te + span * 0.05;
      S.viewFrom = S.viewTo - span;
      if (S.viewFrom < 0) S.viewFrom = 0;
      S.followTail = true;
    }

    function setTf(tf) {
      if (tf === S.stepMs) return;
      S.stepMs = tf;
      ctx.setState({ tf });
      S.cacheKey = null;
      for (const [v, b] of S.tfButtons) b.classList.toggle('active', v === tf);
      const te = tailEnd();
      if (isFinite(te)) {
        let anchor = S.followTail ? te : (S.viewFrom + S.viewTo) / 2;
        const half = 75 * tf;
        S.viewFrom = anchor - half;
        S.viewTo = anchor + half;
        if (S.viewFrom < 0) {
          S.viewTo -= S.viewFrom;
          S.viewFrom = 0;
        }
      }
      requestDraw();
    }

    function goLive() {
      if (S.viewTo <= S.viewFrom) return;
      const te = tailEnd();
      if (!isFinite(te)) return;
      const span = S.viewTo - S.viewFrom;
      S.followTail = true;
      S.viewTo = te;
      S.viewFrom = te - span;
      requestDraw();
    }

    function xy2tp(x, y, L) {
      const msPerPx = (S.viewTo - S.viewFrom) / (L.plotW * (1 - RIGHT_OFF));
      const cy = Math.min(Math.max(y, 0), L.priceH);
      return {
        t: Math.round((S.viewFrom + x * msPerPx) / S.stepMs) * S.stepMs,
        p: S.yRange.max - cy / L.priceH * (S.yRange.max - S.yRange.min),
      };
    }

    function figP2y(L) {
      return (p) => L.priceH - (p - S.yRange.min) /
        (S.yRange.max - S.yRange.min) * L.priceH;
    }

    function distSeg(px, py, x1, y1, x2, y2) {
      const dx = x2 - x1, dy = y2 - y1;
      const len2 = dx * dx + dy * dy;
      let u = len2 ? ((px - x1) * dx + (py - y1) * dy) / len2 : 0;
      u = Math.min(1, Math.max(0, u));
      return Math.hypot(px - (x1 + u * dx), py - (y1 + u * dy));
    }

    function hitTest(x, y, L) {
      if (!S.drawings.length) return null;
      const msPerPx = (S.viewTo - S.viewFrom) / (L.plotW * (1 - RIGHT_OFF));
      const t2x = (t) => (t - S.viewFrom) / msPerPx;
      const p2y = figP2y(L);
      if (S.selected >= 0 && S.selected < S.drawings.length) {
        const d = S.drawings[S.selected];
        if (d.a) {
          for (let ai = 0; ai < d.a.length; ai++) {
            if (Math.hypot(x - t2x(d.a[ai].t), y - p2y(d.a[ai].p)) <= 8) {
              return { di: S.selected, ai };
            }
          }
        }
      }
      for (let di = S.drawings.length - 1; di >= 0; di--) {
        const d = S.drawings[di];
        if (d.kind === 'hline') {
          if (Math.abs(y - p2y(d.a[0].p)) <= 5) return { di, ai: -1 };
        } else if (d.kind === 'pen') {
          if (!d.pts || !d.pts.length) continue;
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (const q of d.pts) {
            const qx = t2x(q.t), qy = p2y(q.p);
            if (qx < x0) x0 = qx;
            if (qx > x1) x1 = qx;
            if (qy < y0) y0 = qy;
            if (qy > y1) y1 = qy;
          }
          if (x >= x0 - 5 && x <= x1 + 5 && y >= y0 - 5 && y <= y1 + 5) {
            return { di, ai: -1 };
          }
        } else if (d.a.length >= 2) {
          if (distSeg(x, y, t2x(d.a[0].t), p2y(d.a[0].p),
            t2x(d.a[1].t), p2y(d.a[1].p)) <= 5) return { di, ai: -1 };
        }
      }
      return null;
    }

    function persistDrawings() {
      ctx.setState({ drawingsByRun: S.drawingsByRun });
    }

    function pushDrawing(d) {
      S.drawings.push(d);
      S.selected = S.drawings.length - 1;
      persistDrawings();
    }

    function commitPlacing() {
      const d = S.placing.kind === 'pen'
        ? { kind: 'pen', pts: S.placing.a }
        : { kind: S.placing.kind, a: S.placing.a };
      S.placing = null;
      if (d.kind === 'pen' && d.pts.length < 2) {
        setTool(null);
        requestDraw();
        return;
      }
      if (d.kind === 'measure') S.temp = d;
      else pushDrawing(d);
      setTool(null);
      requestDraw();
    }

    function onPointerDown(e) {
      const L = layout();
      const x = e.offsetX - L.tbW;
      const inPlot = x >= 0 && x <= L.plotW &&
        e.offsetY >= 0 && e.offsetY < L.plotH;
      if (x > L.plotW && e.offsetY < L.plotH && !S.tool) {
        S.dragging = true;
        S.dragKind = 'y';
        S.dragY = e.offsetY;
        canvas.setPointerCapture(e.pointerId);
        return;
      }
      if (inPlot && S.viewTo > S.viewFrom && L.priceH > 0) {
        const tp = xy2tp(x, e.offsetY, L);
        if (e.shiftKey && S.fsm !== 'placing') {
          S.placing = { kind: 'measure', a: [tp], cur: tp, drag: false };
          S.fsm = 'placing';
          requestDraw();
          return;
        }
        if (S.tool === 'hline') {
          pushDrawing({ kind: 'hline', a: [tp] });
          setTool(null);
          return;
        }
        if (S.tool === 'pen' && S.fsm === 'toolActive') {
          S.placing = { kind: 'pen', a: [tp], cur: tp, drag: true,
            px: x, py: e.offsetY };
          S.fsm = 'placing';
          canvas.setPointerCapture(e.pointerId);
          requestDraw();
          return;
        }
        if (S.tool && S.fsm === 'toolActive') {
          S.placing = { kind: S.tool, a: [tp], cur: tp, drag: false };
          S.fsm = 'placing';
          requestDraw();
          return;
        }
        if (S.fsm === 'placing' && S.placing && !S.placing.drag &&
          (S.tool || S.placing.kind === 'measure')) {
          S.placing.a.push(tp);
          commitPlacing();
          return;
        }
        if (!S.tool) {
          if (S.temp) { S.temp = null; requestDraw(); }
          const hit = hitTest(x, e.offsetY, L);
          if (hit) {
            S.selected = hit.di;
            if (hit.ai >= 0) {
              S.fsm = 'draggingAnchor';
              S.dragAnchor = hit;
              canvas.setPointerCapture(e.pointerId);
            }
            requestDraw();
            return;
          }
          if (S.selected >= 0) { S.selected = -1; requestDraw(); }
        }
      }
      S.dragging = true;
      S.dragKind = 'pan';
      S.dragX = x;
      S.dragY = e.offsetY;
      S.dragFrom = S.viewFrom;
      canvas.setPointerCapture(e.pointerId);
    }

    function onPointerMove(e) {
      const L = layout();
      const x = e.offsetX - L.tbW;
      S.mouse = { x, y: e.offsetY };
      if (L.priceH > 0 && S.viewTo > S.viewFrom) {
        if (S.fsm === 'placing' && S.placing) {
          const tp = xy2tp(x, e.offsetY, L);
          S.placing.cur = tp;
          if (S.placing.kind === 'pen' &&
            Math.hypot(x - S.placing.px, e.offsetY - S.placing.py) >= 4) {
            S.placing.a.push(tp);
            S.placing.px = x;
            S.placing.py = e.offsetY;
          }
          requestDraw();
          return;
        }
        if (S.fsm === 'draggingAnchor' && S.dragAnchor) {
          const d = S.drawings[S.dragAnchor.di];
          if (d && d.a[S.dragAnchor.ai]) {
            d.a[S.dragAnchor.ai] = xy2tp(x, e.offsetY, L);
          }
          requestDraw();
          return;
        }
      }
      if (S.dragging && S.dragKind === 'y') {
        if (L.priceH > 0) {
          const dy = e.offsetY - S.dragY;
          S.dragY = e.offsetY;
          S.yScale = Math.min(100, Math.max(0.05,
            S.yScale * Math.exp(dy * 3 / L.priceH)));
          S.yAuto = false;
        }
        requestDraw();
        return;
      }
      if (!S.dragging) {
        canvas.style.cursor =
          (x > L.plotW && e.offsetY < L.plotH) ? 'ns-resize' : '';
      }
      if (S.dragging && S.viewTo > S.viewFrom) {
        if (L.plotW > 0) {
          const span = S.viewTo - S.viewFrom;
          const msPerPx = span / (L.plotW * (1 - RIGHT_OFF));
          const dt = (S.dragX - x) * msPerPx;
          S.viewFrom = S.dragFrom + dt;
          S.viewTo = S.viewFrom + span;
          S.followTail = S.viewTo >= tailEnd() - S.stepMs * 2;
          const dy = e.offsetY - S.dragY;
          if (dy !== 0 && L.priceH > 0) {
            S.dragY = e.offsetY;
            S.yAuto = false;
            S.yShift += dy * (S.yRange.max - S.yRange.min) / L.priceH;
          }
        }
      }
      requestDraw();
    }

    function onPointerUp(e) {
      if (S.fsm === 'placing' && S.placing && S.placing.drag) {
        commitPlacing();
      } else if (S.fsm === 'draggingAnchor') {
        S.fsm = 'idle';
        S.dragAnchor = null;
        persistDrawings();
      }
      S.dragging = false;
      if (canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
      }
    }

    function onPointerLeave() {
      S.mouse = null;
      S.dragging = false;
      requestDraw();
    }

    function onDblClick(e) {
      const L = layout();
      if (e.offsetX - L.tbW > L.plotW && e.offsetY < L.plotH) {
        S.yAuto = true;
        S.yScale = 1;
        S.yShift = 0;
        requestDraw();
      }
    }

    function onKeyDown(e) {
      if (!S.hover) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' ||
        t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if (e.key === 'Escape') {
        if (S.fsm === 'placing') {
          S.placing = null;
          S.fsm = S.tool ? 'toolActive' : 'idle';
          requestDraw();
        } else if (S.selected >= 0) {
          S.selected = -1;
          requestDraw();
        } else if (S.tool) {
          setTool(null);
        } else if (S.indOpen) {
          toggleIndMenu(false);
        }
        return;
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && S.selected >= 0) {
        e.preventDefault();
        S.drawings.splice(S.selected, 1);
        S.selected = -1;
        persistDrawings();
        requestDraw();
        return;
      }
      const i = Number(e.key) - 1;
      if (!(i >= 0 && i < TF_LADDER.length)) return;
      e.preventDefault();
      setTf(TF_LADDER[i][1]);
    }

    function onWheel(e) {
      e.preventDefault();
      if (S.viewTo <= S.viewFrom) return;
      const { plotW } = layout();
      if (plotW <= 0) return;
      const span = S.viewTo - S.viewFrom;
      if (e.shiftKey) {
        const msPerPx = span / (plotW * (1 - RIGHT_OFF));
        const dt = (e.deltaY !== 0 ? e.deltaY : e.deltaX) * msPerPx * 1.2;
        S.viewFrom += dt;
        S.viewTo += dt;
      } else {
        const frac = Math.min(1, Math.max(0,
          (e.offsetX - TOOLBAR_W) / (plotW * (1 - RIGHT_OFF))));
        const tAt = S.viewFrom + span * frac;
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        let newSpan = span * factor;
        newSpan = Math.min(4000 * 36e5, Math.max(S.stepMs * 5, newSpan));
        S.viewFrom = tAt - newSpan * frac;
        S.viewTo = S.viewFrom + newSpan;
      }
      S.followTail = S.viewTo >= tailEnd() - S.stepMs * 2;
      requestDraw();
    }

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('dblclick', onDblClick);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    goLiveEl.addEventListener('click', goLive);
    root.addEventListener('pointerenter', () => { S.hover = true; });
    root.addEventListener('pointerleave', () => { S.hover = false; });
    window.addEventListener('keydown', onKeyDown);

    function requestDraw() {
      if (S.rafPending) return;
      S.rafPending = true;
      requestAnimationFrame(() => {
        S.rafPending = false;
        draw();
      });
    }

    function draw() {
      const L = layout();
      goLiveEl.style.display =
        (!S.followTail && S.runs.length) ? '' : 'none';
      g2d.fillStyle = S.colors.bg;
      g2d.fillRect(0, 0, S.w, S.h);
      if (S.w < 40 || S.h < 40) return;
      if (!S.runs.length) {
        drawCentered('нет данных');
        return;
      }
      if (S.viewTo <= S.viewFrom || L.plotW <= 0 || L.priceH <= 0) return;
      const step = S.stepMs;
      const entry = getEntry();
      if (!entry) {
        drawCentered('нет данных');
        return;
      }
      const span = S.viewTo - S.viewFrom;
      const msPerPx = span / (L.plotW * (1 - RIGHT_OFF));
      const t2x = (t) => (t - S.viewFrom) / msPerPx;
      let i0 = Math.max(0, Math.floor((S.viewFrom - entry.t0) / step));
      let i1 = Math.min(entry.n - 1, Math.ceil((S.viewTo - entry.t0) / step));
      let agg = null;
      if (i1 > i0 && (i1 - i0) > L.plotW * 1.5) {
        agg = aggregateColumns(entry, i0, i1, step, t2x, L.plotW);
      }
      let pMin = Infinity, pMax = -Infinity;
      if (agg) {
        for (let i = 0; i < agg.cols; i++) {
          if (agg.n[i] === 0) continue;
          if (agg.l[i] < pMin) pMin = agg.l[i];
          if (agg.h[i] > pMax) pMax = agg.h[i];
        }
      } else if (i1 >= i0) {
        for (let i = i0; i <= i1; i++) {
          if (entry.nTick[i] === 0) continue;
          if (entry.l[i] < pMin) pMin = entry.l[i];
          if (entry.h[i] > pMax) pMax = entry.h[i];
        }
      }
      const havePrice = isFinite(pMin);
      if (havePrice) {
        const pad = (pMax - pMin) * 0.06 || 1;
        pMin -= pad;
        pMax += pad;
      } else {
        pMin = 0;
        pMax = 1;
      }
      if (!S.yAuto) {
        const mid = (pMin + pMax) / 2;
        const half = (pMax - pMin) / 2 * S.yScale;
        pMin = mid - half;
        pMax = mid + half;
      }
      pMin += S.yShift;
      pMax += S.yShift;
      S.yRange.min = pMin;
      S.yRange.max = pMax;
      const p2y = (p) => L.priceH - (p - pMin) / (pMax - pMin) * L.priceH;
      g2d.save();
      g2d.translate(L.tbW, 0);
      drawGrid(L, t2x, p2y, pMin, pMax, havePrice);
      if (i1 >= i0) drawCandles(L, entry, i0, i1, step, t2x, p2y, agg);
      let indN = 0;
      if (i1 >= i0 && S.indicators.length) {
        indN = buildSource(entry, i0, i1, step, t2x, agg, msPerPx);
        if (indN) {
          calcIndicators(indN);
          drawOverlayInds(L, p2y, indN);
        }
      }
      drawDrawings(L, t2x, p2y);
      if (i1 >= i0) drawLastPrice(L, entry, i0, i1, p2y);
      if (indN) drawIndPanels(L, indN);
      let ch = null;
      if (S.mouse) {
        ch = drawCrosshair(L, entry, step, t2x, p2y, pMin, pMax, havePrice);
      }
      drawLegend(L, entry, i0, i1, ch);
      drawIndLegend();
      drawSeparator(L);
      g2d.restore();
      if (!havePrice) drawCentered('нет данных в диапазоне');
    }

    function drawLastPrice(L, entry, i0, i1, p2y) {
      let i = i1;
      while (i >= i0 && entry.nTick[i] === 0) i--;
      if (i < i0) return;
      const c = entry.c[i];
      const up = entry.c[i] >= entry.o[i];
      const col = up ? S.colors.up : S.colors.down;
      const y = Math.round(p2y(c)) + 0.5;
      if (y < 0 || y > L.priceH) return;
      g2d.beginPath();
      g2d.setLineDash([4, 4]);
      g2d.moveTo(0, y);
      g2d.lineTo(L.plotW, y);
      g2d.strokeStyle = col;
      g2d.lineWidth = 1;
      g2d.stroke();
      g2d.setLineDash([]);
      const label = fmtPrice(c, 0.01);
      g2d.font = '11px sans-serif';
      const tw = g2d.measureText(label).width + 10;
      g2d.fillStyle = col;
      g2d.fillRect(L.plotW, y - 9, Math.max(tw, L.priceAxisW), 18);
      g2d.fillStyle = '#fff';
      g2d.textAlign = 'left';
      g2d.textBaseline = 'middle';
      g2d.fillText(label, L.plotW + 5, y);
    }

    function drawGrid(L, t2x, p2y, pMin, pMax, havePrice) {
      g2d.strokeStyle = S.colors.border;
      g2d.fillStyle = S.colors.fg2;
      g2d.lineWidth = 1;
      g2d.font = '11px sans-serif';
      g2d.textBaseline = 'middle';
      let timeStep = TIME_STEPS[TIME_STEPS.length - 1];
      for (const s of TIME_STEPS) {
        if (s / (S.viewTo - S.viewFrom) * L.plotW >= 90) { timeStep = s; break; }
      }
      const firstTick = Math.ceil(S.viewFrom / timeStep) * timeStep;
      g2d.textAlign = 'center';
      g2d.beginPath();
      for (let t = firstTick; t <= S.viewTo; t += timeStep) {
        const x = Math.round(t2x(t)) + 0.5;
        g2d.moveTo(x, 0);
        g2d.lineTo(x, L.plotH);
        g2d.fillText(fmtT(t), x, L.plotH + L.timeAxisH / 2);
      }
      g2d.stroke();
      if (!havePrice) return;
      const rawStep = (pMax - pMin) / Math.max(2, L.priceH / 48);
      const pStep = niceStep(rawStep);
      g2d.textAlign = 'left';
      g2d.beginPath();
      for (let p = Math.ceil(pMin / pStep) * pStep; p <= pMax; p += pStep) {
        const y = Math.round(p2y(p)) + 0.5;
        if (y < 6 || y > L.priceH - 6) continue;
        g2d.moveTo(0, y);
        g2d.lineTo(L.plotW, y);
        g2d.fillText(fmtPrice(p, pStep), L.plotW + 5, y);
      }
      g2d.stroke();
    }

    function aggregateColumns(entry, i0, i1, step, t2x, plotW) {
      const cols = plotW;
      const n = new Int32Array(cols);
      const o = new Float64Array(cols);
      const h = new Float64Array(cols);
      const l = new Float64Array(cols);
      const c = new Float64Array(cols);
      for (let i = i0; i <= i1; i++) {
        if (entry.nTick[i] === 0) continue;
        let col = Math.round(t2x(entry.t0 + i * step));
        if (col < 0) col = 0;
        else if (col >= cols) col = cols - 1;
        if (n[col] === 0) {
          o[col] = entry.o[i];
          h[col] = entry.h[i];
          l[col] = entry.l[i];
        } else {
          if (entry.h[i] > h[col]) h[col] = entry.h[i];
          if (entry.l[i] < l[col]) l[col] = entry.l[i];
        }
        c[col] = entry.c[i];
        n[col]++;
      }
      return { cols, n, o, h, l, c };
    }

    function drawCandles(L, entry, i0, i1, step, t2x, p2y, agg) {
      g2d.save();
      g2d.beginPath();
      g2d.rect(0, 0, L.plotW, L.priceH);
      g2d.clip();
      if (agg) {
        let prevUp = null;
        for (let x = 0; x < agg.cols; x++) {
          if (agg.n[x] === 0) continue;
          const up = agg.c[x] >= agg.o[x];
          if (up !== prevUp) {
            g2d.fillStyle = up ? S.colors.up : S.colors.down;
            g2d.strokeStyle = up ? S.colors.up : S.colors.down;
            prevUp = up;
          }
          const yH = p2y(agg.h[x]);
          const yL = p2y(agg.l[x]);
          g2d.fillRect(x, yH, 1, Math.max(1, yL - yH));
          const yO = p2y(agg.o[x]);
          const yC = p2y(agg.c[x]);
          const top = Math.min(yO, yC);
          const hgt = Math.max(1, Math.abs(yC - yO));
          g2d.fillRect(x - 1, top, 3, hgt);
        }
      } else {
        const pxPer = step / (S.viewTo - S.viewFrom) * L.plotW * (1 - RIGHT_OFF);
        const bw = Math.max(1, Math.floor(pxPer * 0.7));
        g2d.lineWidth = 1;
        for (let i = i0; i <= i1; i++) {
          if (entry.nTick[i] === 0) continue;
          const up = entry.c[i] >= entry.o[i];
          const color = up ? S.colors.up : S.colors.down;
          const cx = t2x(entry.t0 + i * step) + pxPer / 2;
          const x = Math.round(cx - bw / 2);
          g2d.strokeStyle = color;
          g2d.beginPath();
          g2d.moveTo(Math.round(cx) + 0.5, p2y(entry.h[i]));
          g2d.lineTo(Math.round(cx) + 0.5, p2y(entry.l[i]));
          g2d.stroke();
          const yO = p2y(entry.o[i]);
          const yC = p2y(entry.c[i]);
          const top = Math.min(yO, yC);
          const hgt = Math.max(1, Math.abs(yC - yO));
          g2d.fillStyle = color;
          g2d.fillRect(x, top, bw, hgt);
        }
      }
      g2d.restore();
    }

    function ensureBufs(n) {
      const b = S.buf;
      if (b.cap >= n) return;
      const cap = Math.max(1024, n * 2);
      for (const k of ['x', 't', 'o', 'h', 'l', 'c', 'sma', 'ema',
        'bbm', 'bbu', 'bbl', 'rsi', 'macd', 'sig', 'hist',
        'tmp1', 'tmp2']) {
        b[k] = new Float64Array(cap);
      }
      b.cap = cap;
    }

    function buildSource(entry, i0, i1, step, t2x, agg, msPerPx) {
      const b = S.buf;
      let j = 0;
      if (agg) {
        ensureBufs(agg.cols);
        for (let col = 0; col < agg.cols; col++) {
          if (agg.n[col] === 0) continue;
          b.x[j] = col + 0.5;
          b.t[j] = S.viewFrom + col * msPerPx;
          b.o[j] = agg.o[col]; b.h[j] = agg.h[col]; b.l[j] = agg.l[col];
          b.c[j] = agg.c[col];
          j++;
        }
      } else {
        ensureBufs(i1 - i0 + 1);
        const pxPer = step / msPerPx;
        for (let i = i0; i <= i1; i++) {
          if (entry.nTick[i] === 0) continue;
          b.x[j] = t2x(entry.t0 + i * step) + pxPer / 2;
          b.t[j] = entry.t0 + i * step;
          b.o[j] = entry.o[i]; b.h[j] = entry.h[i]; b.l[j] = entry.l[i];
          b.c[j] = entry.c[i];
          j++;
        }
      }
      return j;
    }

    function calcSMA(src, out, n, period) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += src[i];
        if (i >= period) sum -= src[i - period];
        out[i] = i >= period - 1 ? sum / period : NaN;
      }
    }

    function calcEMAFrom(src, out, n, from, period) {
      const k = 2 / (period + 1);
      let e = 0, sum = 0, cnt = 0;
      for (let i = 0; i < n; i++) {
        if (i < from) { out[i] = NaN; continue; }
        cnt++;
        if (cnt < period) { sum += src[i]; out[i] = NaN; continue; }
        if (cnt === period) {
          sum += src[i];
          e = sum / period;
          out[i] = e;
          continue;
        }
        e = src[i] * k + e * (1 - k);
        out[i] = e;
      }
    }

    function calcBB(src, mid, up, lo, n, period, mult) {
      let sum = 0, sq = 0;
      for (let i = 0; i < n; i++) {
        sum += src[i];
        sq += src[i] * src[i];
        if (i >= period) {
          sum -= src[i - period];
          sq -= src[i - period] * src[i - period];
        }
        if (i >= period - 1) {
          const m = sum / period;
          let va = sq / period - m * m;
          if (va < 0) va = 0;
          const sd = Math.sqrt(va);
          mid[i] = m;
          up[i] = m + mult * sd;
          lo[i] = m - mult * sd;
        } else {
          mid[i] = NaN; up[i] = NaN; lo[i] = NaN;
        }
      }
    }

    function calcRSI(src, out, n, period) {
      let avgG = 0, avgL = 0;
      for (let i = 0; i < n; i++) {
        if (i === 0) { out[i] = NaN; continue; }
        const ch = src[i] - src[i - 1];
        const g = ch > 0 ? ch : 0;
        const lo = ch < 0 ? -ch : 0;
        if (i <= period) {
          avgG += g / period;
          avgL += lo / period;
          out[i] = i === period
            ? (avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL)) : NaN;
        } else {
          avgG = (avgG * (period - 1) + g) / period;
          avgL = (avgL * (period - 1) + lo) / period;
          out[i] = avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL);
        }
      }
    }

    function calcMACD(b, n) {
      calcEMAFrom(b.c, b.tmp1, n, 0, 12);
      calcEMAFrom(b.c, b.tmp2, n, 0, 26);
      let m0 = n;
      for (let i = 0; i < n; i++) {
        if (Number.isNaN(b.tmp2[i])) { b.macd[i] = NaN; continue; }
        b.macd[i] = b.tmp1[i] - b.tmp2[i];
        if (m0 === n) m0 = i;
      }
      if (m0 < n) calcEMAFrom(b.macd, b.sig, n, m0, 9);
      else for (let i = 0; i < n; i++) b.sig[i] = NaN;
      for (let i = 0; i < n; i++) {
        b.hist[i] = Number.isNaN(b.sig[i]) ? NaN : b.macd[i] - b.sig[i];
      }
    }

    function calcIndicators(n) {
      const b = S.buf;
      const inds = S.indicators;
      if (inds.includes('sma')) calcSMA(b.c, b.sma, n, 20);
      if (inds.includes('ema')) calcEMAFrom(b.c, b.ema, n, 0, 50);
      if (inds.includes('bb')) calcBB(b.c, b.bbm, b.bbu, b.bbl, n, 20, 2);
      if (inds.includes('rsi')) calcRSI(b.c, b.rsi, n, 14);
      if (inds.includes('macd')) calcMACD(b, n);
    }

    function strokeB(buf, n, yOf, color, width) {
      const b = S.buf;
      g2d.strokeStyle = color;
      g2d.lineWidth = width;
      g2d.beginPath();
      let st = false;
      for (let i = 0; i < n; i++) {
        const v = buf[i];
        if (Number.isNaN(v)) { st = false; continue; }
        const y = yOf(v);
        if (!st) { g2d.moveTo(b.x[i], y); st = true; }
        else g2d.lineTo(b.x[i], y);
      }
      g2d.stroke();
    }

    function fillBBChannel(n, p2y) {
      const b = S.buf;
      g2d.fillStyle = 'rgba(79,142,247,0.07)';
      g2d.beginPath();
      let run = -1;
      for (let i = 0; i <= n; i++) {
        const ok = i < n && !Number.isNaN(b.bbu[i]) && !Number.isNaN(b.bbl[i]);
        if (ok && run < 0) run = i;
        if (!ok && run >= 0) {
          g2d.moveTo(b.x[run], p2y(b.bbu[run]));
          for (let j = run + 1; j < i; j++) g2d.lineTo(b.x[j], p2y(b.bbu[j]));
          for (let j = i - 1; j >= run; j--) g2d.lineTo(b.x[j], p2y(b.bbl[j]));
          g2d.closePath();
          run = -1;
        }
      }
      g2d.fill();
    }

    function drawOverlayInds(L, p2y, n) {
      const b = S.buf;
      const inds = S.indicators;
      g2d.save();
      g2d.beginPath();
      g2d.rect(0, 0, L.plotW, L.priceH);
      g2d.clip();
      if (inds.includes('bb')) {
        fillBBChannel(n, p2y);
        strokeB(b.bbu, n, p2y, 'rgba(125,134,153,0.7)', 1);
        strokeB(b.bbl, n, p2y, 'rgba(125,134,153,0.7)', 1);
        strokeB(b.bbm, n, p2y, 'rgba(125,134,153,0.9)', 1);
      }
      if (inds.includes('sma')) strokeB(b.sma, n, p2y, S.colors.accent, 1.5);
      if (inds.includes('ema')) strokeB(b.ema, n, p2y, '#f7a24f', 1.5);
      g2d.restore();
    }

    function drawIndPanels(L, n) {
      const b = S.buf;
      const panels = [];
      if (S.indicators.includes('rsi')) panels.push('rsi');
      if (S.indicators.includes('macd')) panels.push('macd');
      if (!panels.length || !L.indH) return;
      const y0 = L.priceH + L.volH;
      const ph = L.indH / panels.length;
      g2d.save();
      g2d.beginPath();
      g2d.rect(0, y0, L.plotW, L.indH);
      g2d.clip();
      g2d.font = '10px sans-serif';
      g2d.textAlign = 'left';
      g2d.textBaseline = 'top';
      for (let pi = 0; pi < panels.length; pi++) {
        const top = y0 + ph * pi;
        if (panels[pi] === 'rsi') {
          const mapY = (v) => top + ph - 3 - v / 100 * (ph - 6);
          g2d.strokeStyle = S.colors.border;
          g2d.setLineDash([3, 3]);
          g2d.beginPath();
          for (const lvl of [30, 70]) {
            const y = Math.round(mapY(lvl)) + 0.5;
            g2d.moveTo(0, y);
            g2d.lineTo(L.plotW, y);
          }
          g2d.stroke();
          g2d.setLineDash([]);
          strokeB(b.rsi, n, mapY, '#b46ff7', 1.5);
          g2d.fillStyle = '#b46ff7';
          g2d.fillText('RSI(14)', 8, top + 3);
        } else {
          let mn = Infinity, mx = -Infinity;
          for (let i = 0; i < n; i++) {
            let v = b.macd[i];
            if (!Number.isNaN(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
            v = b.sig[i];
            if (!Number.isNaN(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
            v = b.hist[i];
            if (!Number.isNaN(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
          }
          g2d.fillStyle = S.colors.fg2;
          g2d.fillText('MACD(12,26,9)', 8, top + 3);
          if (!isFinite(mn)) continue;
          if (mx === mn) { mx += 1; mn -= 1; }
          const pad = (mx - mn) * 0.08;
          mn -= pad;
          mx += pad;
          const yOf = (v) => top + ph - 3 - (v - mn) / (mx - mn) * (ph - 6);
          const yz = yOf(0);
          g2d.strokeStyle = S.colors.border;
          g2d.beginPath();
          g2d.moveTo(0, Math.round(yz) + 0.5);
          g2d.lineTo(L.plotW, Math.round(yz) + 0.5);
          g2d.stroke();
          const bw = n > 1 ? Math.max(1, (b.x[1] - b.x[0]) * 0.6) : 1;
          let prevUp = null;
          for (let i = 0; i < n; i++) {
            const v = b.hist[i];
            if (Number.isNaN(v)) continue;
            const up = v >= 0;
            if (up !== prevUp) {
              g2d.fillStyle = up ? S.colors.up : S.colors.down;
              prevUp = up;
            }
            const yv = yOf(v);
            g2d.fillRect(b.x[i] - bw / 2, Math.min(yz, yv), bw,
              Math.max(1, Math.abs(yv - yz)));
          }
          strokeB(b.macd, n, yOf, S.colors.accent, 1.2);
          strokeB(b.sig, n, yOf, '#f7a24f', 1.2);
        }
      }
      g2d.restore();
    }

    function drawDrawings(L, t2x, p2y) {
      if (!S.drawings.length && !S.temp && !S.placing) return;
      g2d.save();
      g2d.beginPath();
      g2d.rect(0, 0, S.w, L.plotH);
      g2d.clip();
      for (let di = 0; di < S.drawings.length; di++) {
        const dd = S.drawings[di];
        renderFigure(dd.kind, dd.kind === 'pen' ? dd.pts : dd.a,
          di === S.selected, false, L, t2x, p2y);
      }
      if (S.temp) {
        renderFigure(S.temp.kind, S.temp.a, false, false, L, t2x, p2y);
      }
      if (S.placing) {
        if (S.placing.kind === 'pen') {
          renderFigure('pen', S.placing.a, false, true, L, t2x, p2y);
        } else {
          const a = S.pairBuf;
          a[0] = S.placing.a[0];
          a[1] = S.placing.a.length > 1 ? S.placing.a[1] : S.placing.cur;
          renderFigure(S.placing.kind, a, true, true, L, t2x, p2y);
        }
      }
      g2d.restore();
      g2d.setLineDash([]);
    }

    function renderFigure(kind, a, sel, preview, L, t2x, p2y) {
      const col = S.colors.accent;
      if (preview) g2d.setLineDash([5, 4]);
      g2d.lineWidth = sel && !preview ? 2 : 1.5;
      if (kind === 'hline') {
        const y = p2y(a[0].p);
        g2d.strokeStyle = col;
        g2d.beginPath();
        g2d.moveTo(0, y);
        g2d.lineTo(L.plotW, y);
        g2d.stroke();
        g2d.setLineDash([]);
        const label = fmtPrice(a[0].p, niceStep(Math.abs(a[0].p) / 100 || 0.01));
        g2d.font = '11px sans-serif';
        const tw = g2d.measureText(label).width + 10;
        g2d.fillStyle = col;
        g2d.fillRect(L.plotW, y - 9, Math.max(tw, L.priceAxisW), 18);
        g2d.fillStyle = '#fff';
        g2d.textAlign = 'left';
        g2d.textBaseline = 'middle';
        g2d.fillText(label, L.plotW + 5, y);
      } else if (kind === 'trend' && a[0] && a[1]) {
        g2d.strokeStyle = col;
        g2d.beginPath();
        g2d.moveTo(t2x(a[0].t), p2y(a[0].p));
        g2d.lineTo(t2x(a[1].t), p2y(a[1].p));
        g2d.stroke();
      } else if (kind === 'pen' && a && a.length >= 2) {
        g2d.strokeStyle = col;
        g2d.lineWidth = 2;
        g2d.lineJoin = 'round';
        g2d.lineCap = 'round';
        g2d.beginPath();
        g2d.moveTo(t2x(a[0].t), p2y(a[0].p));
        for (let i = 1; i < a.length; i++) {
          g2d.lineTo(t2x(a[i].t), p2y(a[i].p));
        }
        g2d.stroke();
      } else if (kind === 'fibo' && a[0] && a[1]) {
        const x1 = t2x(a[0].t);
        const x2 = t2x(a[1].t);
        const dp = a[1].p - a[0].p;
        const pStep = niceStep(Math.abs(dp) / 20 || 0.01);
        g2d.font = '10px sans-serif';
        g2d.textAlign = 'left';
        g2d.textBaseline = 'middle';
        for (let fi = 0; fi < FIB_LEVELS.length; fi++) {
          const lvl = FIB_LEVELS[fi];
          const fcol = FIB_COLORS[fi];
          const p = a[0].p + dp * lvl;
          const y = p2y(p);
          g2d.strokeStyle = fcol;
          g2d.globalAlpha = 0.75;
          g2d.beginPath();
          g2d.moveTo(x1, y);
          g2d.lineTo(x2, y);
          g2d.stroke();
          g2d.globalAlpha = 1;
          const pct = (lvl * 100) % 1 === 0
            ? '' + lvl * 100 : (lvl * 100).toFixed(1);
          g2d.fillStyle = fcol;
          g2d.fillText(pct + '% ' + fmtPrice(p, pStep),
            Math.max(x1, x2) + 4, y);
        }
      } else if (kind === 'measure' && a[0] && a[1]) {
        const x1 = t2x(a[0].t), x2 = t2x(a[1].t);
        const y1 = p2y(a[0].p), y2 = p2y(a[1].p);
        const up = a[1].p >= a[0].p;
        const col2 = up ? S.colors.up : S.colors.down;
        const rx = Math.min(x1, x2), ry = Math.min(y1, y2);
        const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1);
        g2d.globalAlpha = 0.12;
        g2d.fillStyle = col2;
        g2d.fillRect(rx, ry, rw, rh);
        g2d.globalAlpha = 1;
        g2d.strokeStyle = col2;
        g2d.lineWidth = 1;
        g2d.strokeRect(rx, ry, rw, rh);
        const dp = a[1].p - a[0].p;
        const bars = Math.round(Math.abs(a[1].t - a[0].t) / S.stepMs);
        const line1 = fmtSigned(dp) + ' п.п.';
        const line2 = bars + ' бар · ' + fmtDurH(a[1].t - a[0].t);
        g2d.font = '11px sans-serif';
        const pw = Math.max(g2d.measureText(line1).width,
          g2d.measureText(line2).width) + 12;
        const px = Math.min(Math.max((x1 + x2) / 2 - pw / 2, 2), L.plotW - pw - 2);
        const py = Math.min(Math.max((y1 + y2) / 2 - 17, 2), L.plotH - 36);
        g2d.setLineDash([]);
        g2d.fillStyle = 'rgba(27,36,52,0.92)';
        g2d.fillRect(px, py, pw, 34);
        g2d.strokeStyle = S.colors.border;
        g2d.strokeRect(px + 0.5, py + 0.5, pw, 34);
        g2d.fillStyle = col2;
        g2d.textAlign = 'left';
        g2d.textBaseline = 'top';
        g2d.fillText(line1, px + 6, py + 4);
        g2d.fillStyle = S.colors.fg2;
        g2d.fillText(line2, px + 6, py + 19);
      }
      if (sel) {
        g2d.setLineDash([]);
        if (kind === 'pen') {
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (let i = 0; i < a.length; i++) {
            const qx = t2x(a[i].t), qy = p2y(a[i].p);
            if (qx < x0) x0 = qx;
            if (qx > x1) x1 = qx;
            if (qy < y0) y0 = qy;
            if (qy > y1) y1 = qy;
          }
          if (isFinite(x0)) {
            g2d.setLineDash([4, 3]);
            g2d.strokeStyle = col;
            g2d.lineWidth = 1;
            g2d.strokeRect(x0 - 4, y0 - 4, x1 - x0 + 8, y1 - y0 + 8);
          }
        } else {
          for (let i = 0; i < a.length; i++) {
            if (!a[i]) continue;
            g2d.beginPath();
            g2d.arc(t2x(a[i].t), p2y(a[i].p), 3.5, 0, Math.PI * 2);
            g2d.fillStyle = col;
            g2d.fill();
            g2d.strokeStyle = '#fff';
            g2d.lineWidth = 1;
            g2d.stroke();
          }
        }
      }
      g2d.setLineDash([]);
    }

    function drawIndLegend() {
      if (!S.indicators.length) return;
      g2d.font = '11px sans-serif';
      g2d.textAlign = 'left';
      g2d.textBaseline = 'top';
      let lx = 8;
      for (const key of S.indicators) {
        let name = null, col = null;
        if (key === 'sma') { name = 'SMA(20)'; col = S.colors.accent; }
        else if (key === 'ema') { name = 'EMA(50)'; col = '#f7a24f'; }
        else if (key === 'bb') { name = 'BB(20,2)'; col = S.colors.fg2; }
        else if (key === 'rsi') { name = 'RSI(14)'; col = '#b46ff7'; }
        else if (key === 'macd') { name = 'MACD(12,26,9)'; col = S.colors.fg2; }
        if (!name) continue;
        g2d.fillStyle = col;
        g2d.fillText(name, lx, 20);
        lx += g2d.measureText(name).width + 14;
      }
    }

    function drawCrosshair(L, entry, step, t2x, p2y, pMin, pMax, havePrice) {
      const rawX = S.mouse.x;
      const rawY = S.mouse.y;
      if (rawX < 0 || rawX > L.plotW || rawY < 0 || rawY > L.plotH) return null;
      const msPerPx = (S.viewTo - S.viewFrom) / (L.plotW * (1 - RIGHT_OFF));
      const tAt = S.viewFrom + rawX * msPerPx;
      const bt = Math.floor(tAt / step) * step;
      let candle = null;
      let hi = null;
      if (entry && entry.n) {
        const i = Math.round((bt - entry.t0) / step);
        if (i >= 0 && i < entry.n && entry.nTick[i] > 0) candle = i;
      }
      let mx = rawX;
      let my = rawY;
      if (S.magnet && candle !== null) {
        const pxPer = step / msPerPx;
        mx = t2x(entry.t0 + candle * step) + pxPer / 2;
        if (havePrice && rawY <= L.priceH) {
          const vals = [
            ['O', entry.o[candle]], ['H', entry.h[candle]],
            ['L', entry.l[candle]], ['C', entry.c[candle]],
          ];
          let best = MAGNET_PX + 1;
          for (const [k, v] of vals) {
            const dy = Math.abs(p2y(v) - rawY);
            if (dy < best) {
              best = dy;
              hi = k;
            }
          }
          if (hi !== null) my = p2y(vals.find((p) => p[0] === hi)[1]);
        }
      }
      g2d.strokeStyle = S.colors.fg2;
      g2d.lineWidth = 1;
      g2d.setLineDash([3, 3]);
      g2d.beginPath();
      g2d.moveTo(Math.round(mx) + 0.5, 0);
      g2d.lineTo(Math.round(mx) + 0.5, L.plotH);
      g2d.moveTo(0, Math.round(my) + 0.5);
      g2d.lineTo(L.plotW, Math.round(my) + 0.5);
      g2d.stroke();
      g2d.setLineDash([]);
      if (havePrice && my <= L.priceH) {
        const price = pMax - my / L.priceH * (pMax - pMin);
        const pStep = niceStep((pMax - pMin) / Math.max(2, L.priceH / 48));
        const label = fmtPrice(price, pStep);
        g2d.font = '11px sans-serif';
        const tw = g2d.measureText(label).width + 10;
        g2d.fillStyle = S.colors.bg3;
        g2d.fillRect(L.plotW, my - 9, tw, 18);
        g2d.strokeStyle = S.colors.border;
        g2d.strokeRect(L.plotW + 0.5, my - 8.5, tw, 17);
        g2d.fillStyle = hi !== null ? S.colors.accent : S.colors.fg;
        g2d.textAlign = 'left';
        g2d.textBaseline = 'middle';
        g2d.fillText(label, L.plotW + 5, my);
      }
      const tLabel = fmtT(bt);
      g2d.font = '11px sans-serif';
      const ttw = g2d.measureText(tLabel).width + 10;
      const tx = Math.min(L.plotW - ttw, Math.max(0, mx - ttw / 2));
      g2d.fillStyle = S.colors.bg3;
      g2d.fillRect(tx, L.plotH, ttw, L.timeAxisH);
      g2d.fillStyle = S.colors.fg;
      g2d.textAlign = 'center';
      g2d.fillText(tLabel, tx + ttw / 2, L.plotH + L.timeAxisH / 2);
      return { candle, hi };
    }

    function drawLegend(L, entry, i0, i1, ch) {
      if (!entry || i1 < i0) return;
      let i = ch && ch.candle !== null ? ch.candle : -1;
      if (i < 0) {
        i = i1;
        while (i >= i0 && entry.nTick[i] === 0) i--;
        if (i < i0) return;
      }
      const up = entry.c[i] >= entry.o[i];
      const dirCol = up ? S.colors.up : S.colors.down;
      const chg = entry.o[i] ? (entry.c[i] - entry.o[i]) / entry.o[i] * 100 : 0;
      const parts = [
        ['O', 'O ' + fmtPrice(entry.o[i], 0.01)],
        ['H', 'H ' + fmtPrice(entry.h[i], 0.01)],
        ['L', 'L ' + fmtPrice(entry.l[i], 0.01)],
        ['C', 'C ' + fmtPrice(entry.c[i], 0.01)],
        ['Δ', (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%'],
        ['V', 'N ' + fmtVol(entry.nTick[i])],
      ];
      g2d.font = '11px sans-serif';
      g2d.textAlign = 'left';
      g2d.textBaseline = 'top';
      let lx = 8;
      for (const [k, s] of parts) {
        const hot = ch && ch.hi === k;
        g2d.fillStyle = hot ? S.colors.accent :
          (k === 'V' ? S.colors.fg2 : dirCol);
        g2d.fillText(s, lx, 6);
        const w = g2d.measureText(s).width;
        if (hot) g2d.fillRect(lx, 19, w, 1.5);
        lx += w + 14;
      }
    }

    function drawSeparator(L) {
      g2d.strokeStyle = S.colors.border;
      g2d.lineWidth = 1;
      g2d.beginPath();
      g2d.moveTo(0, L.priceH + 0.5);
      g2d.lineTo(L.plotW, L.priceH + 0.5);
      if (L.indH) {
        const y = L.priceH + L.volH + 0.5;
        g2d.moveTo(0, y);
        g2d.lineTo(L.plotW, y);
      }
      g2d.moveTo(L.plotW + 0.5, 0);
      g2d.lineTo(L.plotW + 0.5, L.plotH);
      g2d.stroke();
    }

    function drawCentered(text) {
      g2d.font = '13px sans-serif';
      g2d.fillStyle = S.colors.fg2;
      g2d.textAlign = 'center';
      g2d.textBaseline = 'middle';
      g2d.fillText(text, S.w / 2, S.h / 2);
    }

    const chips = document.createElement('span');
    chips.className = 'candles-tf';
    for (const [label, tf] of TF_LADDER) {
      const b = document.createElement('button');
      b.textContent = label;
      if (tf === S.stepMs) b.classList.add('active');
      b.addEventListener('click', () => setTf(tf));
      chips.appendChild(b);
      S.tfButtons.set(tf, b);
    }
    hdr.appendChild(chips);
    const mag = document.createElement('button');
    mag.className = 'candles-mag';
    mag.title = 'магнит: прилипание кроссхейра к OHLC';
    mag.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none"' +
      ' stroke="currentColor" stroke-width="2" stroke-linecap="round">' +
      '<path d="M5 2v6a3 3 0 0 0 6 0V2"/><line x1="3" y1="2" x2="7" y2="2"/>' +
      '<line x1="9" y1="2" x2="13" y2="2"/></svg>';
    const syncMag = () => mag.classList.toggle('active', S.magnet);
    syncMag();
    mag.addEventListener('click', () => {
      S.magnet = !S.magnet;
      ctx.setState({ magnet: S.magnet });
      syncMag();
      requestDraw();
    });
    hdr.appendChild(mag);

    sel.addEventListener('change', () => {
      S.runIdx = sel.selectedIndex;
      ctx.setState({ run: S.runIdx });
      S.cacheKey = null;
      S.yAuto = true;
      S.yScale = 1;
      S.yShift = 0;
      S.drawings = (S.drawingsByRun[S.runIdx] ||= []);
      S.selected = -1;
      S.placing = null;
      S.fsm = S.tool ? 'toolActive' : 'idle';
      setViewportInitial();
      requestDraw();
    });

    onResize();

    onData(data => {
      S.runs = (data.runs || []).filter(r => r.series && r.series.length);
      sel.innerHTML = '';
      S.runs.forEach((r, i) => {
        const o = document.createElement('option');
        o.textContent = r.name;
        sel.appendChild(o);
      });
      if (S.runIdx >= S.runs.length) S.runIdx = 0;
      sel.selectedIndex = S.runIdx;
      S.cacheKey = null;
      if (!S.init && S.runs.length) {
        S.init = true;
        setViewportInitial();
      } else if (S.followTail && S.viewTo > S.viewFrom) {
        const te = tailEnd();
        if (isFinite(te) && te > S.viewTo) {
          const span = S.viewTo - S.viewFrom;
          S.viewTo = te;
          S.viewFrom = te - span;
        }
      }
      requestDraw();
    });
  },
};
