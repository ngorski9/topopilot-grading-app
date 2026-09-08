let mode,
  user,
  password,
  trialData = [],
  selected,
  gradedIds = new Set();
const $ = (x) => document.querySelector(x);
const esc = (s) =>
  String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
async function login() {
  let r = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: $("#username").value,
      password: $("#password").value,
    }),
  });
  if (!r.ok) {
    $("#loginError").textContent = "Username or password not found.";
    return;
  }
  user = $("#username").value;
  password = $("#password").value;
  start("grader");
}
function spectate() {
  start("spectator");
}
async function start(m) {
  mode = m;
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#identity").textContent =
    m === "grader" ? "Grading as " + user : "Spectator mode";
  trialData = await (await fetch("/api/trials")).json();
  if (m === "grader") {
    gradedIds = new Set(
      await (
        await fetch(
          "/api/graded?grader=" +
            encodeURIComponent(user) +
            "&password=" +
            encodeURIComponent(password),
        )
      ).json(),
    );
    $("#nextUngraded").classList.remove("hidden");
  }
  let groups = [
    ...new Set(
      trialData.map(
        (t) =>
          `${t.task} · prompt ${t.prompt} · ${t.api ? "TopoPilot API" : "No API"}`,
      ),
    ),
  ].sort();
  $("#filter").innerHTML =
    '<option value="">All trials</option>' +
    groups.map((x) => `<option>${esc(x)}</option>`).join("");
  renderTrials();
}
function renderTrials() {
  let f = $("#filter").value;
  let data = trialData.filter(
    (t) =>
      !f ||
      f ===
        `${t.task} · prompt ${t.prompt} · ${t.api ? "TopoPilot API" : "No API"}`,
  );
  $("#trialList").innerHTML = data
    .map((t) => {
      let done = mode === "grader" && gradedIds.has(t.id);
      return `<div class="trial ${selected?.id === t.id ? "active" : ""}" data-trial-index="${trialData.indexOf(t)}"><span class="completion ${done ? "done" : "todo"}">${mode === "grader" ? (done ? "graded" : "ungraded") : ""}</span><b>${esc(t.task)}</b><br><span class="muted">${esc(t.agent)} · run ${t.run} · prompt ${t.prompt}</span><br><span class="pill">${t.api ? "TopoPilot API" : "No API"}</span></div>`;
    })
    .join("");
  document
    .querySelectorAll("[data-trial-index]")
    .forEach((card) =>
      card.addEventListener("click", () =>
        pick(trialData[Number(card.dataset.trialIndex)].id),
      ),
    );
}
function nextUngraded() {
  let start = selected
    ? trialData.findIndex((t) => t.id === selected.id) + 1
    : 0;
  let target = [...trialData.slice(start), ...trialData.slice(0, start)].find(
    (t) => !gradedIds.has(t.id),
  );
  if (target) pick(target.id);
  else alert("You have graded every trial.");
}
async function pick(id) {
  if (mode === "grader" && selected && selected.id !== id) {
    await saveGrade(false);
  }
  selected = trialData.find((t) => t.id === id);
  renderTrials();
  let files = await (
    await fetch("/api/files?trial=" + encodeURIComponent(id))
  ).json();
  let gradeQuery =
    "/api/grades?trial=" +
    encodeURIComponent(id) +
    (mode === "grader"
      ? "&grader=" +
        encodeURIComponent(user) +
        "&password=" +
        encodeURIComponent(password)
      : "");
  let grades = await (await fetch(gradeQuery)).json();
  $("#detail").innerHTML =
    `<h2>${esc(selected.task)}</h2><p><b>${esc(selected.agent)}</b> · run ${selected.run} · prompt ${selected.prompt} · ${selected.api ? "TopoPilot API" : "No API"}</p><div class="card"><h3>Files</h3><div id="fileList">${files.map((f, i) => `<button class="file" data-file-index="${i}">${esc(/^(claude|codex)_result\.txt$/.test(f) ? "agent response" : f)}</button>`).join("") || '<span class="muted">No files found.</span>'}</div></div><div id="viewer" class="card"><p class="muted">Choose a file above to view it here.</p></div>`;
  document
    .querySelectorAll("[data-file-index]")
    .forEach((b) =>
      b.addEventListener("click", () =>
        viewFile(files[Number(b.dataset.fileIndex)]),
      ),
    );
  $("#rubricPanel").innerHTML =
    mode === "grader" ? graderForm(grades) : spectator(grades);
  let first =
    files.find((f) => /\.(png|jpe?g|gif|webp|mp4)$/i.test(f)) ||
    files.find((f) => /\.(py|txt|log|jsonl|csv|pvsm)$/i.test(f));
  if (first) viewFile(first);
}
function highlightPython(text) {
  // Escape each token separately.  Chaining replacements after inserting
  // spans would re-highlight words in attributes such as `class="str"`.
  const tokens = /(#.*$)|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])*')|\b(def|class|import|from|for|while|if|elif|else|return|in|True|False|None|with|as|try|except|raise|lambda|yield)\b|\b\d+(?:\.\d+)?\b/gm;
  let output = "";
  let lastIndex = 0;
  for (const match of text.matchAll(tokens)) {
    output += esc(text.slice(lastIndex, match.index));
    let category = match[1] ? "com" : match[2] || match[3] ? "str" : match[4] ? "kw" : "num";
    output += `<span class="${category}">${esc(match[0])}</span>`;
    lastIndex = match.index + match[0].length;
  }
  return output + esc(text.slice(lastIndex));
}
function markdown(text) {
  let code = false,
    codeLines = [],
    list = false,
    out = [];
  const closeList = () => {
    if (list) {
      out.push("</ul>");
      list = false;
    }
  };
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(
        /\[([^\]]+)\]\((https?:[^ )]+)\)/g,
        '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
      );
  for (let line of text.split("\n")) {
    if (line.startsWith("```")) {
      if (code) {
        out.push("<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>");
        codeLines = [];
      }
      code = !code;
      continue;
    }
    if (code) {
      codeLines.push(line);
      continue;
    }
    let heading = line.match(/^(#{1,3})\s+(.+)/);
    let item = line.match(/^[-*]\s+(.+)/);
    if (heading) {
      closeList();
      out.push(
        "<h" +
          heading[1].length +
          ">" +
          inline(heading[2]) +
          "</h" +
          heading[1].length +
          ">",
      );
    } else if (item) {
      if (!list) {
        out.push("<ul>");
        list = true;
      }
      out.push("<li>" + inline(item[1]) + "</li>");
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      out.push("<p>" + inline(line) + "</p>");
    }
  }
  closeList();
  if (code)
    out.push("<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>");
  return out.join("");
}
async function viewFile(file) {
  let isResponse = /^(claude|codex)_result\.txt$/.test(file);
  document
    .querySelectorAll("[data-file-index]")
    .forEach((b) =>
      b.classList.toggle(
        "active",
        b.textContent === (isResponse ? "agent response" : file),
      ),
    );
  let url =
      "/file?trial=" +
      encodeURIComponent(selected.id) +
      "&path=" +
      encodeURIComponent(file),
    viewer = $("#viewer");
  if (/\.(png|jpe?g|gif|webp)$/i.test(file)) {
    viewer.innerHTML = `<h3>${esc(file)}</h3><div class="image-controls"><button type="button" onclick="changeImageZoom(-0.25)" aria-label="Zoom out">−</button><button type="button" onclick="resetImageZoom()">Reset</button><button type="button" onclick="changeImageZoom(0.25)" aria-label="Zoom in">+</button><span id="zoomLevel">100%</span></div><div id="imageStage" class="image-stage"><img id="zoomableImage" class="preview zoomable" src="${url}" alt="${esc(file)}" title="Click to zoom in"></div>`;
    resetImageZoom();
    $("#zoomableImage").addEventListener("click", () => changeImageZoom(0.25));
    $("#zoomableImage").addEventListener("load", (event) => {
      event.currentTarget.dataset.baseWidth = event.currentTarget.clientWidth;
      updateImageZoom();
    });
    return;
  }
  if (/\.mp4$/i.test(file)) {
    viewer.innerHTML = `<h3>${esc(file)}</h3><video class="preview" controls src="${url}"></video>`;
    return;
  }
  try {
    let text = await (await fetch(url)).text();
    if (isResponse) {
      viewer.innerHTML = `<h3>Agent response</h3><div class="markdown">${markdown(text)}</div>`;
    } else {
      viewer.innerHTML = `<h3>${esc(file)}</h3><pre class="code">${/\.py$/i.test(file) ? highlightPython(text) : esc(text)}</pre>`;
    }
  } catch {
    viewer.innerHTML = "<p>Unable to display this file.</p>";
  }
}
let imageZoom = 1;
function updateImageZoom() {
  let image = $("#zoomableImage");
  if (!image) return;
  if (image.dataset.baseWidth) {
    image.style.maxWidth = "none";
    image.style.maxHeight = "none";
    image.style.width = `${Number(image.dataset.baseWidth) * imageZoom}px`;
  }
  $("#zoomLevel").textContent = `${Math.round(imageZoom * 100)}%`;
}
function changeImageZoom(amount) {
  imageZoom = Math.max(0.25, Math.min(4, imageZoom + amount));
  updateImageZoom();
}
function resetImageZoom() {
  imageZoom = 1;
  updateImageZoom();
}
const dataRubric = [
  "Correctly computed",
  "Correctly computed, but edge cases are not handled correctly",
  "Computed with errors",
  "Not computed",
];
const visRubric = [
  "Correctly visualized",
  "Visualized, but design choices impair interpretability",
  "Visualization requirements are only partially fulfilled",
  "Not visualized",
];
function requirementCard(kind, requirement, index, grade, options) {
  let prefix = `${kind}_${index}`;
  return `<div class="requirement"><p><b>${esc(requirement)}</b></p><label>Rubric<select id="${prefix}_rating"><option value="">Select an option…</option>${options.map((option) => `<option value="${esc(option)}" ${grade.rating === option ? "selected" : ""}>${esc(option)}</option>`).join("")}</select></label><label>Comments (optional)<textarea id="${prefix}_comment" class="comment" placeholder="Comment (optional)">${esc(grade.comment || "")}</textarea></label></div>`;
}
function graderForm(grades) {
  let mine = grades.find((g) => g.grader === user) || {};
  let dataGrades = mine.data_grades || [];
  let visGrades = mine.vis_grades || [];
  return `<div class="card"><h3>Your grade</h3><div class="benchmark-row"><label>Benchmark score <input id="benchmark_score" value="${esc(mine.benchmark_score || "")}"></label><label><input id="benchmark_not_applicable" type="checkbox" ${mine.benchmark_not_applicable === "1" ? "checked" : ""}> Benchmark score not applicable</label></div><h4>Data requirements</h4>${selected.data_requirements.map((requirement, index) => requirementCard("data", requirement, index, dataGrades[index] || {}, dataRubric)).join("") || '<p class="muted">No data requirements are defined for this trial.</p>'}<h4>Visualization requirements</h4>${selected.vis_requirements.map((requirement, index) => requirementCard("vis", requirement, index, visGrades[index] || {}, visRubric)).join("") || '<p class="muted">No visualization requirements are defined for this trial.</p>'}<button onclick="saveGrade()">Save grade</button> <span id="status"></span></div>`;
}
function spectator(grades) {
  if (!grades.length)
    return '<div class="card"><h3>Grading</h3><p class="muted">No grader feedback yet.</p></div>';
  return `<div class="card"><h3>Anonymized grader feedback</h3>${grades
    .map(
      (g) =>
        `<div class="grader-grade"><b>${g.grader}</b><br>Benchmark score: ${esc(g.benchmark_score || "—")} ${g.benchmark_not_applicable === "1" ? "(not applicable)" : ""}<h4>Data requirements</h4><ul>${selected.data_requirements.map((requirement, index) => `<li><b>${esc(requirement)}</b>: ${esc((g.data_grades[index] || {}).rating || "Not graded")}${(g.data_grades[index] || {}).comment ? " — " + esc(g.data_grades[index].comment) : ""}</li>`).join("")}</ul><h4>Visualization requirements</h4><ul>${selected.vis_requirements.map((requirement, index) => `<li><b>${esc(requirement)}</b>: ${esc((g.vis_grades[index] || {}).rating || "Not graded")}${(g.vis_grades[index] || {}).comment ? " — " + esc(g.vis_grades[index].comment) : ""}</li>`).join("")}</ul></div>`,
    )
    .join("")}</div>`;
}
async function saveGrade(showStatus = true) {
  if (mode !== "grader" || !selected) return true;
  let d = {
    trial_id: selected.id,
    grader: user,
    password,
    data_grades: selected.data_requirements.map((_, index) => ({
      rating: $("#data_" + index + "_rating").value,
      comment: $("#data_" + index + "_comment").value,
    })),
    vis_grades: selected.vis_requirements.map((_, index) => ({
      rating: $("#vis_" + index + "_rating").value,
      comment: $("#vis_" + index + "_comment").value,
    })),
  };
  d.benchmark_score = $("#benchmark_score").value;
  d.benchmark_not_applicable = $("#benchmark_not_applicable").checked
    ? "1"
    : "0";
  let r = await fetch("/api/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(d),
  });
  if (r.ok) {
    gradedIds.add(selected.id);
    renderTrials();
  }
  if (showStatus && $("#status")) {
    $("#status").textContent = r.ok
      ? "Saved. Summary CSV refreshed."
      : "Could not save grade.";
  }
  return r.ok;
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") saveGrade(false);
});
