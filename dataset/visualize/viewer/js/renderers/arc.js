export const ARC_COLORS = [
  "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
  "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
];

export function drawGrid(canvas, grid, cell = 12) {
  const h = grid.length, w = grid[0].length;
  canvas.width = w * cell;
  canvas.height = h * cell;
  const ctx = canvas.getContext("2d");
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      ctx.fillStyle = ARC_COLORS[grid[y][x]] || "#fff";
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}

export function render(sample) {
  const box = document.createElement("div");
  box.className = "sample";
  const title = document.createElement("h4");
  title.textContent = `${sample.id} (${sample.split}${sample.group ? "/" + sample.group : ""})`;
  box.appendChild(title);
  const pairs = [...sample.train.map(p => ["train", p]), ...sample.test.map(p => ["test", p])];
  for (const [kind, pair] of pairs) {
    for (const [label, grid] of [["in", pair.input], ["out", pair.output]]) {
      const holder = document.createElement("div");
      holder.className = "pair";
      const canvas = document.createElement("canvas");
      drawGrid(canvas, grid);
      holder.appendChild(canvas);
      const cap = document.createElement("div");
      cap.textContent = `${kind} ${label}`;
      holder.appendChild(cap);
      box.appendChild(holder);
    }
  }
  return box;
}
