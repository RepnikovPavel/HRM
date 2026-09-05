function drawSudoku(canvas, flat, fixed, cell = 28) {
  const side = 9;
  canvas.width = canvas.height = side * cell;
  const ctx = canvas.getContext("2d");
  ctx.font = `${cell * 0.6}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      const v = flat[y * side + x];
      ctx.fillStyle = fixed && v !== "0" ? "#e8e8ff" : "#fff";
      ctx.fillRect(x * cell, y * cell, cell, cell);
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(x * cell, y * cell, cell, cell);
      if (v !== "0" && v !== ".") {
        ctx.fillStyle = "#111";
        ctx.fillText(v, x * cell + cell / 2, y * cell + cell / 2);
      }
    }
  }
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 2;
  for (let i = 0; i <= side; i += 3) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, side * cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(side * cell, i * cell); ctx.stroke();
  }
}

export function render(sample) {
  const box = document.createElement("div");
  box.className = "sample";
  const title = document.createElement("h4");
  title.textContent = `${sample.split} rating=${sample.rating}`;
  box.appendChild(title);
  for (const [label, flat, fixed] of [["puzzle", sample.input, true], ["solution", sample.output, false]]) {
    const holder = document.createElement("div");
    holder.className = "pair";
    const canvas = document.createElement("canvas");
    drawSudoku(canvas, flat, fixed);
    holder.appendChild(canvas);
    const cap = document.createElement("div");
    cap.textContent = label;
    holder.appendChild(cap);
    box.appendChild(holder);
  }
  return box;
}
