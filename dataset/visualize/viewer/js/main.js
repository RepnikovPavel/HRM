const RENDERERS = {
  arc: () => import("./renderers/arc.js"),
  sudoku: () => import("./renderers/sudoku.js"),
  maze: () => import("./renderers/maze.js"),
};

async function main() {
  const manifest = await (await fetch("/data/vis/index.json")).json();
  const nav = document.getElementById("datasets");
  for (const ds of manifest.datasets) {
    const btn = document.createElement("button");
    btn.textContent = ds.name;
    btn.onclick = () => show(ds, btn);
    nav.appendChild(btn);
  }
  if (manifest.datasets.length) {
    show(manifest.datasets[0], nav.firstChild);
  }
}

async function show(ds, btn) {
  document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const [samples, stats, task] = await Promise.all([
    fetch(`/data/${ds.vis}/samples.json`).then(r => r.json()),
    fetch(`/data/${ds.vis}/stats.json`).then(r => r.json()),
    fetch(`/data/${ds.vis}/TASK.md`).then(r => r.ok ? r.text() : ""),
  ]);
  const parts = Object.entries(stats.splits).map(([split, s]) => {
    const n = s.puzzles ?? s.examples;
    const extra = s.input_output_pairs ? `, ${s.input_output_pairs} pairs` : "";
    const grid = s.grid ? `, grid ${s.grid}` : "";
    return `${split}: ${n} puzzles${extra}${grid}`;
  });
  document.getElementById("stats").textContent =
    `${ds.name} — ${(stats.disk_bytes / 1e6).toFixed(1)} MB | ` + parts.join(" | ");
  document.getElementById("task-text").textContent = task;
  const mod = await RENDERERS[ds.type]();
  const host = document.getElementById("samples");
  host.textContent = "";
  for (const sample of samples.samples) {
    host.appendChild(mod.render(sample));
  }
}

main();
