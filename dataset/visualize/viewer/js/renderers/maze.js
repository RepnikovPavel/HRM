const TILES = { "#": "#333", " ": "#fff", "S": "#2ECC40", "G": "#FF4136", "o": "#FFDC00" };

function drawMaze(canvas, flat, side, cell = 8) {
  canvas.width = canvas.height = side * cell;
  const ctx = canvas.getContext("2d");
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      ctx.fillStyle = TILES[flat[y * side + x]] || "#9cf";
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}

export function render(sample) {
  const box = document.createElement("div");
  box.className = "sample";
  const title = document.createElement("h4");
  title.textContent = `${sample.split} rating=${sample.rating}`;
  box.appendChild(title);
  for (const [label, flat] of [["task", sample.input], ["path", sample.output]]) {
    const holder = document.createElement("div");
    holder.className = "pair";
    const canvas = document.createElement("canvas");
    drawMaze(canvas, flat, sample.side);
    holder.appendChild(canvas);
    const cap = document.createElement("div");
    cap.textContent = label;
    holder.appendChild(cap);
    box.appendChild(holder);
  }
  return box;
}
