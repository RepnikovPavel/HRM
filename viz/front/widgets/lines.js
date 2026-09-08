import { onData } from '../shell/shell.js';

const MODES = [
  ['line', 'линия'],
  ['line+points', 'линия с точками'],
  ['points', 'маркеры измерений'],
  ['dashed', 'прерывистая'],
  ['dashed+points', 'прерывистая с точками'],
  ['step-before', 'ступени (до)'],
  ['step-middle', 'ступени (по центру)'],
  ['step-after', 'ступени (после)'],
];

export const def = {
  type: 'lines',
  title: 'Линии: exact accuracy(t)',
  hint: 'обе модели, линейный график',
  defaultSize: { w: 660, h: 430 },
  defaultState: () => ({ mode: 'step-after' }),

  mount(body, ctx) {
    const hdr = document.createElement('div');
    hdr.style.cssText = 'display:flex;align-items:center;gap:4px;padding:3px 6px;' +
      'background:var(--bg2);border-bottom:1px solid var(--border);font-size:12px';
    const sel = document.createElement('select');
    sel.title = 'режим отрисовки';
    sel.style.cssText = 'background:var(--bg3);color:var(--fg);border:1px solid var(--border);' +
      'border-radius:5px;padding:2px 4px;font-size:12px';
    for (const [v, label] of MODES) {
      const o = document.createElement('option');
      o.value = v; o.textContent = label;
      sel.appendChild(o);
    }
    hdr.appendChild(sel);
    body.appendChild(hdr);
    body.style.position = 'relative';
    const plot = document.createElement('div');
    plot.style.cssText = 'position:absolute;inset:25px 0 0 0';
    body.appendChild(plot);
    const cv = document.createElement('canvas');
    cv.style.cssText = 'display:block;width:100%;height:100%;cursor:crosshair';
    plot.appendChild(cv);
    const g = cv.getContext('2d');
    const ML = 46, MR = 14, MT = 26, MB = 24;

    const S = { w: 0, h: 0, runs: [], viewFrom: 0, viewTo: 1, mouse: null,
                dragging: false, dragX: 0, dragFrom: 0, init: false,
                mode: ctx.state.mode || 'step-after' };
    sel.value = S.mode;
    sel.addEventListener('change', () => {
      S.mode = sel.value;
      ctx.setState({ mode: S.mode });
      draw();
    });

    const X = t => ML + (S.w - ML - MR) * (t - S.viewFrom) / (S.viewTo - S.viewFrom);
    const inv = x => S.viewFrom + (x - ML) / (S.w - ML - MR) * (S.viewTo - S.viewFrom);

    function pathOf(series, Y) {
      const pts = series.filter(p => p[0] >= S.viewFrom && p[0] <= S.viewTo);
      if (!pts.length) return;
      g.beginPath();
      if (S.mode === 'step-before') {
        g.moveTo(X(pts[0][0]), Y(pts[0][1]));
        for (let i = 1; i < pts.length; i++) {
          g.lineTo(X(pts[i - 1][0]), Y(pts[i][1]));
          g.lineTo(X(pts[i][0]), Y(pts[i][1]));
        }
      } else if (S.mode === 'step-after') {
        g.moveTo(X(pts[0][0]), Y(pts[0][1]));
        for (let i = 1; i < pts.length; i++) {
          g.lineTo(X(pts[i][0]), Y(pts[i - 1][1]));
          g.lineTo(X(pts[i][0]), Y(pts[i][1]));
        }
      } else if (S.mode === 'step-middle') {
        g.moveTo(X(pts[0][0]), Y(pts[0][1]));
        for (let i = 1; i < pts.length; i++) {
          const xm = X((pts[i - 1][0] + pts[i][0]) / 2);
          g.lineTo(xm, Y(pts[i - 1][1]));
          g.lineTo(xm, Y(pts[i][1]));
          g.lineTo(X(pts[i][0]), Y(pts[i][1]));
        }
      } else {
        pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])));
      }
      g.stroke();
      if (S.mode.endsWith('+points') || S.mode === 'points') {
        for (const p of pts) {
          g.beginPath();
          g.arc(X(p[0]), Y(p[1]), 2.5, 0, 7);
          g.fill();
        }
      }
    }

    function draw() {
      g.clearRect(0, 0, S.w, S.h);
      if (!S.runs.length) return;
      let ymax = 1;
      for (const r of S.runs)
        for (const p of r.series)
          if (p[0] >= S.viewFrom && p[0] <= S.viewTo && p[1] > ymax) ymax = p[1];
      ymax *= 1.06;
      const Y = v => MT + (S.h - MT - MB) * (1 - v / ymax);

      g.strokeStyle = '#2b3650'; g.fillStyle = '#7d8699'; g.font = '11px monospace';
      for (let i = 0; i <= 5; i++) {
        const v = ymax * i / 5, y = Y(v);
        g.beginPath(); g.moveTo(ML, y); g.lineTo(S.w - MR, y); g.stroke();
        g.fillText(v.toFixed(0), 8, y + 4);
      }
      const span = S.viewTo - S.viewFrom;
      const rawStep = span / 8;
      const p10 = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const m = rawStep / p10;
      const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p10;
      for (let t = Math.ceil(S.viewFrom / step) * step; t <= S.viewTo; t += step) {
        const x = X(t);
        g.beginPath(); g.moveTo(x, MT); g.lineTo(x, S.h - MB); g.stroke();
        g.fillText(t.toFixed(t < 10 && step < 1 ? 1 : 0) + 'h', x - 8, S.h - 8);
      }

      S.runs.forEach((r, i) => {
        g.fillStyle = r.color;
        g.fillText(r.name.split(' (')[0], ML + 8 + i * 130, 16);
        g.strokeStyle = r.color;
        g.fillStyle = r.color;
        g.lineWidth = 1.6;
        if (S.mode.startsWith('dashed')) g.setLineDash([5, 4]);
        pathOf(r.series, Y);
        g.setLineDash([]);
      });
      g.lineWidth = 1;

      if (S.mouse && S.mouse.x > ML) {
        const t = inv(S.mouse.x);
        g.strokeStyle = '#4a5568';
        g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(S.mouse.x, MT); g.lineTo(S.mouse.x, S.h - MB); g.stroke();
        g.setLineDash([]);
        let ty = MT + 12;
        for (const r of S.runs) {
          let best = null, bd = Infinity;
          for (const p of r.series) {
            const d = Math.abs(p[0] - t);
            if (d < bd) { bd = d; best = p; }
          }
          if (best && bd < span * 0.03) {
            g.fillStyle = r.color;
            g.fillText(`${r.name.split(' (')[0]}: ${best[1].toFixed(2)}% @ ${best[0].toFixed(2)}h`,
              S.mouse.x + 8, ty);
            ty += 14;
          }
        }
      }
    }

    function onResize() {
      const r = plot.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      S.w = r.width; S.h = r.height;
      cv.width = Math.round(S.w * dpr);
      cv.height = Math.round(S.h * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }
    new ResizeObserver(onResize).observe(plot);

    cv.addEventListener('pointerdown', e => {
      S.dragging = true; S.dragX = e.offsetX; S.dragFrom = S.viewFrom;
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', e => {
      S.mouse = { x: e.offsetX, y: e.offsetY };
      if (S.dragging) {
        const span = S.viewTo - S.viewFrom;
        const dt = (S.dragX - e.offsetX) / (S.w - ML - MR) * span;
        S.viewFrom = S.dragFrom + dt;
        S.viewTo = S.viewFrom + span;
      }
      draw();
    });
    cv.addEventListener('pointerup', e => {
      S.dragging = false;
      if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
    });
    cv.addEventListener('pointerleave', () => { S.mouse = null; S.dragging = false; draw(); });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      const t = inv(e.offsetX);
      const k = Math.exp(e.deltaY * 0.0015);
      S.viewFrom = t - (t - S.viewFrom) * k;
      S.viewTo = t + (S.viewTo - t) * k;
      draw();
    }, { passive: false });

    onData(data => {
      S.runs = data.runs.filter(r => r.series.length);
      if (!S.init && S.runs.length) {
        S.init = true;
        let mx = 0;
        for (const r of S.runs)
          for (const p of r.series) if (p[0] > mx) mx = p[0];
        S.viewFrom = 0; S.viewTo = mx * 1.05;
      }
      draw();
    });
  },
};
