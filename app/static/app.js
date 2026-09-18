const SUBJECTS = ["All", "DSA", "Operating Systems", "DBMS", "Computer Networks"];
const STARTERS = [
  "What conditions must hold for a deadlock to occur?",
  "TCP vs UDP: what's the difference?",
  "Why do databases use B-trees instead of binary search trees?",
  "What is a functional dependency?",
];
const COLORS = [["#fbe7a2", "sand"], ["#cde3f6", "sky"], ["#cfded2", "sage"], ["#f5d9da", "rose"], ["#ddd8f2", "lilac"]];
const MODE_HINTS = {
  explain: "I teach you the topic, with the page behind every claim.",
  simple: "I explain it in plain words, still only from the books.",
  quiz: "I test you with three questions, then give the answer key.",
  socratic: "I guide you with questions instead of handing over the answer. Reply to keep going.",
};
const STRATEGIES = {
  analogy: ["Use an analogy", "Still confused. Try an analogy."],
  example: ["Walk through an example", "Still confused. Walk me through an example."],
  steps: ["Break it into steps", "Still confused. Break it into steps."],
};

const state = { mode: "explain", subject: "All", history: [], busy: false };
const turns = new Map(); // every answer keeps its own sources, so a citation always opens the right passage
let turnCount = 0;
let openPassage = null; // the source currently shown in the drawer

const $ = (id) => document.getElementById(id);
const thread = $("thread");

// localStorage can be missing or full (private windows), the app should work without it
const store = {
  read(key, fallback) {
    try { return JSON.parse(localStorage.getItem(`anchor.${key}`)) ?? fallback; } catch { return fallback; }
  },
  write(key, value) {
    try { localStorage.setItem(`anchor.${key}`, JSON.stringify(value)); } catch { /* keep going without saving */ }
  },
};

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const safeColor = (color) => (/^#[0-9a-f]{6}$/i.test(color) ? color : COLORS[0][0]);

// gpt-oss sometimes cites like 【1†L2-L4】 even though the prompt asks for [1],
// and wraps formulas in LaTeX delimiters that would otherwise show up as \(O(n)\)
const tidy = (text) => text
  .replace(/【(\d+)(?:†[^】]*)?】/g, "[$1]")
  .replace(/\\\((.+?)\\\)/g, "$1")
  .replace(/\\\[(.+?)\\\]/g, "$1");

// The answer arrives as Markdown. Escaping before parsing means nothing the model writes can inject HTML.
function renderMarkdown(source) {
  const code = [];
  const text = escapeHtml(tidy(source)).replace(/```[a-z]*\n?([\s\S]*?)```/g, (_, body) => `\u0000${code.push(body) - 1}\u0000`);

  // inline code is pulled out first so that something like a[1] in code isn't read as a citation
  const inline = (line) => {
    const spans = [];
    const out = line
      .replace(/`([^`]+)`/g, (_, body) => `\u0001${spans.push(body) - 1}\u0001`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\[(\d{1,2})\]/g, '<button class="cite" data-source="$1">$1</button>');
    return out.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${spans[i]}</code>`);
  };

  const html = [];
  let list = null;
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }

    const heading = line.match(/^#{1,4}\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);

    if (heading) {
      closeList();
      html.push(`<h3>${inline(heading[1])}</h3>`);
    } else if (bullet) {
      if (list !== "ul") { closeList(); html.push("<ul>"); list = "ul"; }
      html.push(`<li>${inline(bullet[1])}</li>`);
    } else if (numbered) {
      if (list !== "ol") { closeList(); html.push("<ol>"); list = "ol"; }
      html.push(`<li>${inline(numbered[1])}</li>`);
    } else {
      closeList();
      html.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return html.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => `<pre><code>${code[i]}</code></pre>`);
}

// The passage is cut at every place where an evidence word or one of the reader's highlights
// starts or ends, and each piece is wrapped in whatever covers it. Overlaps come out right.
function renderPassage(text, terms, marks) {
  const evidence = [];
  const words = (terms || []).filter((t) => t.length > 3).map(escapeRegExp);
  if (words.length) {
    for (const m of text.matchAll(new RegExp(`\\b(?:${words.join("|")})\\w{0,3}\\b`, "gi"))) {
      evidence.push([m.index, m.index + m[0].length]);
    }
  }
  const cuts = [...new Set([0, text.length, ...evidence.flat(), ...marks.flatMap((m) => [m.start, m.end])])]
    .filter((p) => p >= 0 && p <= text.length)
    .sort((a, b) => a - b);

  let html = "";
  for (let i = 0; i < cuts.length - 1; i++) {
    const [start, end] = [cuts[i], cuts[i + 1]];
    let piece = escapeHtml(text.slice(start, end));
    if (evidence.some(([a, b]) => a <= start && end <= b)) piece = `<span class="evidence">${piece}</span>`;
    const mine = marks.find((m) => m.start <= start && end <= m.end);
    if (mine) {
      piece = `<mark class="mine${mine.note ? " has-note" : ""}" data-mark="${escapeHtml(mine.id)}" ` +
        `style="background:${safeColor(mine.color)}" title="${escapeHtml(mine.note || "")}">${piece}</mark>`;
    }
    html += piece;
  }
  return html;
}

function passageKey(source) {
  let hash = 5381;
  for (let i = 0; i < source.text.length; i++) hash = ((hash * 33) ^ source.text.charCodeAt(i)) >>> 0;
  return `${source.docId}:${source.page}:${hash.toString(36)}`;
}

// "Chapter 11 Functional Dependencies" becomes "Functional Dependencies", the same as clean_title in rag/query.py
const shortSection = (section) => (section || "").split(">").pop().trim()
  .replace(/^(chapter|appendix)\s+\w+\s+/i, "")
  .replace(/^\d+(\.\d+)*\s+/, "")
  .replace(/[\s♥*⋆]+$/, "");

/* ---- turns ---- */

function newTurn(id, shown) {
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.dataset.turn = id;
  turn.innerHTML = `
    <div class="asked"></div>
    <div class="answer">
      <div class="status"><i class="dot"></i><span>searching the library…</span></div>
      <div class="answer-body"></div>
      <div class="sources"></div>
    </div>`;
  turn.querySelector(".asked").textContent = shown;
  thread.appendChild(turn);
  collapseOlder();
  turn.scrollIntoView({ behavior: "smooth", block: "end" });
  return turn;
}

// long threads get heavy, so all but the latest two answers fold down to their first lines
function collapseOlder() {
  [...thread.querySelectorAll(".turn")].slice(0, -2).forEach((turn) => {
    const answer = turn.querySelector(".answer");
    if (!answer || answer.dataset.expanded || answer.classList.contains("collapsed")) return;
    answer.classList.add("collapsed");
    const more = document.createElement("button");
    more.className = "expand";
    more.textContent = "Show the full answer";
    more.onclick = () => {
      answer.classList.remove("collapsed");
      answer.dataset.expanded = "1";
      more.remove();
    };
    answer.appendChild(more);
  });
}

const pillMarkup = (s) => `
  <button class="source-pill" data-source="${s.number}">
    <b>${s.number}</b><span>p.${escapeHtml(s.page)} · ${escapeHtml(s.book)}</span>
  </button>`;

function traceTable(sources) {
  const rows = sources.map((s) => `
    <tr>
      <td class="num">${s.number}</td>
      <td>${escapeHtml(s.section)}</td>
      <td class="num">${s.scores.bm25 ?? "—"}</td>
      <td class="num">${s.scores.dense ?? "—"}</td>
      <td class="num">${s.scores.rerank === null ? "—" : s.scores.rerank.toFixed(3)}</td>
    </tr>`).join("");
  return `
    <details class="trace">
      <summary>How I found this</summary>
      <table>
        <thead><tr><th class="num">#</th><th>Section</th><th class="num">keyword rank</th><th class="num">meaning rank</th><th class="num">reranker</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </details>`;
}

function noteLines(data) {
  const lines = [];
  for (const [typed, meant] of data.corrections || []) {
    lines.push(`Showing results for <b>${escapeHtml(meant)}</b> (you typed ${escapeHtml(typed)})`);
  }
  for (const [acronym, meaning] of data.expansions || []) {
    lines.push(`Reading <b>${escapeHtml(acronym)}</b> as ${escapeHtml(meaning)}`);
  }
  return lines.map((l) => `<p class="note-line">${l}</p>`).join("");
}

function showRefusal(turn, record, data) {
  const card = turn.querySelector(".answer, .refusal");
  card.className = "refusal";
  card.innerHTML = `
    ${noteLines(data)}
    <h3>Not in the library</h3>
    <p>I couldn't find a passage in these nine books that answers this, so answering would mean making something up.</p>
    <div class="suggest" data-slot="suggest"><span class="suggest-label">Looking for the closest topics…</span></div>
    <details class="trace">
      <summary>Nearest passages the search found</summary>
      <div class="sources">${record.sources.map(pillMarkup).join("")}</div>
    </details>`;
  if (record.suggest) showSuggestions(turn, record.suggest);
}

function showSuggestions(turn, data) {
  const slot = turn.querySelector('[data-slot="suggest"]');
  if (!slot) return;
  if (data.kind === "unavailable") {
    slot.innerHTML = `<span class="suggest-label">Try rephrasing it, or browse the Library to see what the books cover.</span>`;
    return;
  }
  if (!data.topics.length) {
    slot.innerHTML = `<span class="suggest-label">Nothing in these books is close to this. They cover data structures and algorithms, operating systems, databases and computer networks.</span>`;
    return;
  }
  const label = data.kind === "typo" ? "Did you mean:" : "Closest topics in the library:";
  slot.innerHTML = `
    <span class="suggest-label">${label}</span>
    <div class="suggest-row">${data.topics.map((t) => `<button class="suggestion" data-ask="${escapeHtml(`Explain ${t}`)}">${escapeHtml(t)}</button>`).join("")}</div>`;
}

function showClarify(turn, data) {
  const card = turn.querySelector(".answer");
  card.className = "clarify";
  card.innerHTML = `
    <h3>${escapeHtml(data.term)} means two different things in these books. Which one do you mean?</h3>
    <div class="suggest-row">${data.options.map((o) => `<button class="suggestion" data-ask="${escapeHtml(o.question)}">${escapeHtml(o.meaning)}</button>`).join("")}</div>`;
}

function answerFooter(turn, data) {
  const foot = document.createElement("div");
  foot.className = "answer-foot";
  foot.innerHTML = `
    <button class="chip" data-action="confused">I'm still confused</button>
    <button class="chip" data-action="practice">Quick practice</button>
    ${data.fellBack ? `<span class="muted" title="The main model's free-tier allowance ran out, so a fallback model answered">answered by ${escapeHtml(data.model)}</span>` : ""}
    <span class="timing">${(data.totalMs / 1000).toFixed(1)}s</span>`;
  turn.querySelector(".answer").appendChild(foot);
}

function showStrategies(turn) {
  if (turn.querySelector(".strategies")) return;
  const row = document.createElement("div");
  row.className = "strategies";
  row.innerHTML = `<span>Try another way:</span>` +
    Object.entries(STRATEGIES).map(([key, [label]]) => `<button class="chip" data-strategy="${key}">${label}</button>`).join("");
  turn.querySelector(".answer-foot").after(row);
}

// Ask the reranker whether each claim is backed by the passage it cites. The answer is already on
// screen by now, so this arrives a moment later and never holds up the reply.
async function checkGrounding(turn, record) {
  try {
    const response = await fetch("/api/grounding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: record.answer, sources: record.sources.map((s) => ({ number: s.number, text: s.text })) }),
    });
    const data = await response.json();
    if (!data.total) return; // nothing that makes a claim, e.g. a reply that is only questions
    const weak = data.sentences.filter((s) => !s.ok);
    const body = turn.querySelector(".answer-body");
    const chip = document.createElement("button");
    chip.className = "chip grounding";
    chip.textContent = `${data.supported}/${data.total} claims supported by the sources`;
    chip.title = weak.length
      ? "Show the claims no retrieved passage supports. Claims without a citation are checked against every passage, not just the cited one."
      : "Every claim is backed by one of the retrieved passages";
    chip.disabled = !weak.length;

    let flagged = false;
    chip.onclick = () => {
      flagged = !flagged;
      let text = record.answer;
      if (flagged) weak.forEach((s) => { text = text.replace(s.text, `⟦${s.text}⟧`); });
      body.innerHTML = renderMarkdown(text).replace(/⟦/g, '<span class="unchecked">').replace(/⟧/g, "</span>");
      chip.classList.toggle("is-on", flagged);
    };
    turn.querySelector(".answer-foot")?.insertBefore(chip, turn.querySelector(".timing"));
  } catch {
    // a failed check shouldn't disturb an answer that is already readable
  }
}

function logActivity(record, refused) {
  if (STRATEGIES[record.mode]) return; // trying another way isn't a new question
  const top = record.sources[0];
  const activity = store.read("activity", []);
  activity.push({
    question: record.question,
    mode: record.mode,
    refused,
    topic: refused || !top ? "" : shortSection(top.section),
    book: refused || !top ? "" : top.book,
    time: Date.now(),
  });
  store.write("activity", activity.slice(-300));
  drawRecent();
}

async function ask(question, { mode = state.mode, previous = null, shown = question } = {}) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("send").disabled = true;
  document.querySelector(".welcome")?.remove();

  const id = ++turnCount;
  const record = { id, question, mode, answer: "", sources: [], terms: [], suggest: null };
  turns.set(id, record);
  const turn = newTurn(id, shown);
  const status = turn.querySelector(".status");
  const body = turn.querySelector(".answer-body");

  // a socratic reply builds on the question it asked last time, if the last turn was socratic too
  if (mode === "socratic" && !previous && state.history.at(-1)?.mode === "socratic") {
    previous = state.history.at(-1).content;
  }

  let finished = false;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: state.history.slice(-6), mode, subject: state.subject, previous }),
    });
    if (!response.ok) throw new Error(`server returned ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const name = part.match(/^event: (.+)$/m)?.[1];
        const data = JSON.parse(part.match(/^data: (.+)$/m)[1]);

        if (name === "clarify") {
          showClarify(turn, data);
        } else if (name === "sources") {
          record.sources = data.sources;
          record.terms = data.terms;
          record.searchQuery = data.searchQuery;
          record.info = { corrections: data.corrections, expansions: data.expansions };
          if (!data.grounded) {
            showRefusal(turn, record, data);
          } else {
            status.querySelector("span").textContent = `reading ${data.sources.length} passages…`;
            turn.querySelector(".sources").innerHTML = data.sources.map(pillMarkup).join("");
            const notes = noteLines(data);
            const rewritten = data.searchQuery !== question && !(data.expansions || []).length && !(data.corrections || []).length
              ? `<p class="note-line">Searched for: ${escapeHtml(data.searchQuery)}</p>` : "";
            if (notes || rewritten) body.insertAdjacentHTML("beforebegin", notes + rewritten);
          }
        } else if (name === "suggest") {
          record.suggest = data;
          showSuggestions(turn, data);
        } else if (name === "token") {
          status.hidden = true;
          record.answer += data.text;
          body.innerHTML = renderMarkdown(record.answer);
        } else if (name === "error") {
          throw new Error(data.message);
        } else if (name === "done") {
          finished = true;
          status.hidden = true;
          if (data.clarify) continue;
          if (data.refused) {
            if (!turn.querySelector(".refusal")) showRefusal(turn, record, record.info || {});
            logActivity(record, true);
            continue;
          }
          record.answer = data.answer;
          body.innerHTML = renderMarkdown(record.answer);
          turn.querySelector(".sources").insertAdjacentHTML("afterend", traceTable(record.sources));
          answerFooter(turn, data);
          checkGrounding(turn, record);
          state.history.push({ role: "user", content: question }, { role: "assistant", content: record.answer, mode });
          logActivity(record, false);
        }
      }
    }
    if (!finished) throw new Error("the answer stopped part way");
  } catch (error) {
    status.hidden = true;
    // whatever had already streamed stays on screen, the error goes underneath it
    body.insertAdjacentHTML("beforeend", `<p class="muted">Something went wrong: ${escapeHtml(error.message)}. Try asking again.</p>`);
  }

  state.busy = false;
  $("send").disabled = false;
}

/* ---- practice ---- */

// label is the topic name progress is kept under; by default the section the questions came from
async function practice(topic, { label = "", shown = topic } = {}) {
  if (state.busy || !topic) return;
  state.busy = true;
  $("send").disabled = true;
  document.querySelector(".welcome")?.remove();
  const id = ++turnCount;
  const record = { id, question: topic, mode: "practice", answer: "", sources: [], terms: [] };
  turns.set(id, record);
  const turn = newTurn(id, `Practice questions: ${shown}`);
  const status = turn.querySelector(".status");
  status.querySelector("span").textContent = "writing questions from the same passages…";

  try {
    const response = await fetch("/api/practice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, subject: state.subject }),
    });
    if (!response.ok) throw new Error(`the server returned ${response.status}`);
    const data = await response.json();
    status.hidden = true;
    record.sources = data.sources;
    label = label || shortSection(data.sources[0]?.section) || topic;
    const body = turn.querySelector(".answer-body");
    if (!data.questions.length) {
      body.innerHTML = `<p class="muted">I couldn't write practice questions for this topic from the books.</p>`;
    } else {
      const batch = Date.now().toString(36); // turn numbers restart on reload, so ids come from the clock
      body.innerHTML = data.questions.map((q, i) => `
        <div class="practice" data-id="${batch}-${i}" data-topic="${escapeHtml(label)}" data-question="${escapeHtml(q.question)}">
          <p class="practice-q">${i + 1}. ${escapeHtml(q.question)}</p>
          <textarea placeholder="Try answering first (optional)"></textarea>
          <div class="actions"><button class="chip" data-action="reveal">Reveal the answer</button></div>
          <div class="practice-a" hidden>
            <p>${escapeHtml(q.answer || "")}${Number(q.source) ? ` <button class="cite" data-source="${Number(q.source)}">${Number(q.source)}</button>` : ""}</p>
            <div class="grade">How did you do?
              <button class="chip" data-grade="got">I got it</button>
              <button class="chip" data-grade="missed">Not quite</button>
            </div>
          </div>
        </div>`).join("");
      turn.querySelector(".sources").innerHTML = data.sources.map(pillMarkup).join("");
    }
  } catch (error) {
    status.hidden = true;
    turn.querySelector(".answer-body").innerHTML = `<p class="muted">Couldn't write practice questions: ${escapeHtml(error.message)}. Try again in a moment.</p>`;
  }
  state.busy = false;
  $("send").disabled = false;
}

function gradePractice(button) {
  const card = button.closest(".practice");
  card.querySelectorAll("[data-grade]").forEach((b) => b.classList.toggle("is-picked", b === button));
  // changing your mind replaces the earlier grade for the same question
  const results = store.read("practice", []).filter((r) => r.id !== card.dataset.id);
  results.push({ id: card.dataset.id, question: card.dataset.question, topic: card.dataset.topic, result: button.dataset.grade, time: Date.now() });
  store.write("practice", results.slice(-500));
}

/* ---- source drawer and personal highlights ---- */

function openSource(turnId, number) {
  const record = turns.get(Number(turnId));
  const source = record?.sources.find((s) => String(s.number) === String(number));
  if (!source) return;
  openPassage = { source, terms: record.terms, key: passageKey(source) };
  hideTools();
  drawPassage();
  $("drawer").hidden = false;
}

function drawPassage() {
  const { source, terms, key } = openPassage;
  const marks = store.read("marks", []).filter((m) => m.key === key);
  $("drawer-title").textContent = `Source ${source.number}`;
  $("drawer-body").innerHTML = `
    <div class="meta">
      <span class="book">${escapeHtml(source.book)}</span>
      <span class="where">${escapeHtml(source.section)}</span>
      <span class="page">page ${escapeHtml(source.page)} · reranker ${source.scores.rerank === null ? "—" : source.scores.rerank.toFixed(3)}</span>
    </div>
    <div class="passage" id="passage">${renderPassage(source.text, terms, marks)}</div>
    <p class="drawer-legend"><span class="evidence">Blue</span> marks the words your question matched. Select any text to highlight it in your colour or add a note.</p>
    ${marks.length ? `<div class="drawer-notes"><h4>Your highlights on this page</h4><div class="list">${marks.map(markItem).join("")}</div></div>` : ""}`;
}

function markItem(mark) {
  return `
    <div class="item">
      <span class="bar-colour" style="background:${safeColor(mark.color)}"></span>
      <div class="body">
        <p class="quote-text">“${escapeHtml(mark.text)}”</p>
        ${mark.note ? `<p class="note-text">${escapeHtml(mark.note)}</p>` : ""}
        <span class="meta-line">${escapeHtml(mark.book)} · p.${escapeHtml(mark.page)}</span>
      </div>
      <button class="icon" data-remove-mark="${escapeHtml(mark.id)}" aria-label="Remove highlight">✕</button>
    </div>`;
}

function saveMark(start, end, color, note) {
  const { source, key } = openPassage;
  const marks = store.read("marks", []);
  marks.push({
    id: `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    key, start, end, color: safeColor(color), note,
    text: source.text.slice(start, end),
    book: source.book, section: source.section, page: source.page, time: Date.now(),
  });
  store.write("marks", marks);
  window.getSelection()?.removeAllRanges();
  hideTools();
  drawPassage();
}

function updateMark(id, changes) {
  store.write("marks", store.read("marks", []).map((m) => (m.id === id ? { ...m, ...changes } : m)));
  hideTools();
  if (openPassage) drawPassage();
}

function removeMark(id) {
  store.write("marks", store.read("marks", []).filter((m) => m.id !== id));
  hideTools();
  if (openPassage && !$("drawer").hidden) drawPassage();
  if (!$("view-progress").hidden) loadProgress();
}

const swatchRow = () => COLORS.map(([c, n]) => `<button class="swatch" data-color="${c}" title="${n}" style="background:${c}"></button>`).join("");
const myColor = () => safeColor(store.read("color", COLORS[0][0]));

function placeTools(rect) {
  const tools = $("mark-tools");
  tools.hidden = false;
  const top = Math.min(rect.bottom + 8, window.innerHeight - tools.offsetHeight - 8);
  const left = Math.min(Math.max(8, rect.left), window.innerWidth - tools.offsetWidth - 8);
  tools.style.top = `${Math.max(8, top)}px`;
  tools.style.left = `${left}px`;
}

function hideTools() {
  $("mark-tools").hidden = true;
}

function newMarkTools(rect, start, end) {
  const tools = $("mark-tools");
  tools.innerHTML = `<div class="row">${swatchRow()}</div><div class="row"><button class="text" data-act="note">Highlight with a note</button></div>`;
  tools.onclick = (event) => {
    const color = event.target.closest(".swatch")?.dataset.color;
    if (color) return saveMark(start, end, color, "");
    if (event.target.closest('[data-act="note"]')) noteEditor(rect, { start, end });
  };
  placeTools(rect);
}

function noteEditor(rect, target) {
  const tools = $("mark-tools");
  const text = target.mark ? target.mark.text : openPassage.source.text.slice(target.start, target.end);
  tools.innerHTML = `
    <p class="quote">“${escapeHtml(text.length > 90 ? `${text.slice(0, 90)}…` : text)}”</p>
    <textarea placeholder="Your note">${escapeHtml(target.mark?.note || "")}</textarea>
    <div class="row"><button class="text" data-act="save">Save</button><button class="text" data-act="cancel">Cancel</button></div>`;
  tools.onclick = (event) => {
    if (event.target.closest('[data-act="cancel"]')) return hideTools();
    if (!event.target.closest('[data-act="save"]')) return;
    const note = tools.querySelector("textarea").value.trim();
    if (target.mark) updateMark(target.mark.id, { note });
    else saveMark(target.start, target.end, myColor(), note);
  };
  placeTools(rect);
  tools.querySelector("textarea").focus();
}

function existingMarkTools(element) {
  const mark = store.read("marks", []).find((m) => m.id === element.dataset.mark);
  if (!mark) return;
  const rect = element.getBoundingClientRect();
  const tools = $("mark-tools");
  tools.innerHTML = `
    ${mark.note ? `<p class="quote">${escapeHtml(mark.note)}</p>` : ""}
    <div class="row">${swatchRow()}</div>
    <div class="row"><button class="text" data-act="note">${mark.note ? "Edit note" : "Add a note"}</button><button class="text" data-act="remove">Remove</button></div>`;
  tools.onclick = (event) => {
    const color = event.target.closest(".swatch")?.dataset.color;
    if (color) return updateMark(mark.id, { color });
    if (event.target.closest('[data-act="remove"]')) return removeMark(mark.id);
    if (event.target.closest('[data-act="note"]')) noteEditor(rect, { mark });
  };
  placeTools(rect);
}

function handleSelection(event) {
  if (event.target.closest("#mark-tools")) return;
  const passage = $("passage");
  const selection = window.getSelection();
  const selecting = passage && selection && !selection.isCollapsed &&
    passage.contains(selection.anchorNode) && passage.contains(selection.focusNode);

  if (!selecting) {
    const mark = event.target.closest("mark.mine");
    return mark ? existingMarkTools(mark) : hideTools();
  }
  // offsets are counted in the passage's plain text, which is exactly the stored source text
  const range = selection.getRangeAt(0);
  const before = document.createRange();
  before.selectNodeContents(passage);
  before.setEnd(range.startContainer, range.startOffset);
  const start = before.toString().length;
  const end = start + range.toString().length;
  if (end - start < 2) return hideTools();
  newMarkTools(range.getBoundingClientRect(), start, end);
}

/* ---- side panels and pages ---- */

function drawRecent() {
  const asked = [...new Set(store.read("activity", []).map((a) => a.question).reverse())].slice(0, 8);
  $("recent").innerHTML = asked.map((q) => `<button title="${escapeHtml(q)}" data-ask="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join("");
}

async function loadLibrary() {
  const data = await (await fetch("/api/library")).json();
  $("library-count").textContent = `${data.books.length} books · ${data.chunks.toLocaleString()} passages indexed`;
  $("library").innerHTML = data.books.map((book) => `
    <article class="card">
      <h3>${escapeHtml(book.title)}</h3>
      <p class="by">${escapeHtml(book.authors)}</p>
      <div class="tags">${book.subjects.map((s) => `<span class="tag">${escapeHtml(s)}</span>`).join("")}</div>
      <div class="foot"><span>${escapeHtml(book.license)}</span><span>${book.chunks} passages</span></div>
    </article>`).join("");
}

function loadProgress() {
  const activity = store.read("activity", []);
  const results = store.read("practice", []);
  const marks = store.read("marks", []);
  const answered = activity.filter((a) => !a.refused && a.topic);

  const topics = new Map();
  for (const a of answered) {
    const key = `${a.book}|${a.topic}`;
    const t = topics.get(key) || { book: a.book, topic: a.topic, count: 0, last: 0 };
    t.count += 1;
    t.last = Math.max(t.last, a.time);
    topics.set(key, t);
  }
  // a topic is judged on its latest five answers, so getting it right on a retry clears it
  const recent = new Map();
  for (const r of results) recent.set(r.topic, [...(recent.get(r.topic) || []), r.result].slice(-5));
  const weak = [...recent.entries()]
    .filter(([, list]) => list.filter((r) => r === "missed").length * 2 >= list.length && list.includes("missed"))
    .map(([topic]) => topic);
  const got = results.filter((r) => r.result === "got").length;

  const byBook = new Map();
  for (const t of [...topics.values()].sort((a, b) => b.last - a.last)) {
    if (!byBook.has(t.book)) byBook.set(t.book, []);
    byBook.get(t.book).push(t);
  }

  $("progress").innerHTML = `
    <div class="metrics">
      <div class="metric"><b>${activity.length}</b><span>questions asked</span></div>
      <div class="metric"><b>${topics.size}</b><span>topics explored</span></div>
      <div class="metric"><b>${results.length ? `${Math.round((got / results.length) * 100)}%` : "—"}</b><span>practice answers you got right${results.length ? ` (${got} of ${results.length})` : ""}</span></div>
      <div class="metric"><b>${marks.length}</b><span>highlights and notes</span></div>
    </div>

    <section class="panel">
      <h3>Worth another look</h3>
      <p>Topics where you missed at least half of your last five practice answers.</p>
      ${weak.length ? `<div class="topic-row">${weak.map((t) => `<button class="topic weak" data-practice="${escapeHtml(t)}">${escapeHtml(t)} · practise again</button>`).join("")}</div>`
        : `<p class="empty">Nothing yet. Use Quick practice under any answer and grade yourself, and weak spots show up here.</p>`}
    </section>

    <section class="panel">
      <h3>What you've studied</h3>
      <p>Grouped by book. Select a topic to go over it again.</p>
      ${byBook.size ? `<div class="topics">${[...byBook.entries()].map(([book, list]) => `
        <div><h4>${escapeHtml(book)}</h4><div class="topic-row">${list.map((t) => `<button class="topic" data-ask="${escapeHtml(`Explain ${t.topic}`)}">${escapeHtml(t.topic)}${t.count > 1 ? ` · ${t.count}` : ""}</button>`).join("")}</div></div>`).join("")}</div>`
        : `<p class="empty">Ask a question and the topics you cover will collect here.</p>`}
    </section>

    <section class="panel">
      <h3>Your highlights and notes</h3>
      <p>Everything you've marked in the sources, newest first.</p>
      ${marks.length ? `<div class="list">${[...marks].reverse().map(markItem).join("")}</div>`
        : `<p class="empty">Open any source and select text to highlight it or add a note.</p>`}
    </section>`;
}

async function loadEvaluation() {
  const data = await (await fetch("/api/evaluation")).json();
  const rows = Object.entries(data.retrieval).filter(([key]) => key.length === 1);
  const best = rows.reduce((a, b) => (b[1]["mrr@10"] > a[1]["mrr@10"] ? b : a));
  const pct = (x) => `${Math.round(x * 100)}%`;

  $("eval-models").textContent = `${data.models.answers} answering · ${data.models.judge} judging`;
  $("evaluation").innerHTML = `
    <div class="metrics">
      <div class="metric"><b>${pct(best[1]["hit@5"])}</b><span>right passage in the top 5</span></div>
      <div class="metric"><b>${pct(data.answers.answered)}</b><span>answerable questions answered</span></div>
      <div class="metric"><b>${pct(data.answers.fullyCorrect)}</b><span>fully correct</span></div>
      <div class="metric"><b>${pct(data.answers.faithful)}</b><span>every claim backed by a source</span></div>
      <div class="metric"><b>${pct(data.answers.refusedCorrectly)}</b><span>off-syllabus questions refused</span></div>
    </div>

    <section class="panel">
      <h3>What each retrieval step is worth</h3>
      <p>Measured on ${data.answers.answerable} hand-checked questions, each with quotes from the books. The highlighted row is what this app runs.</p>
      <table>
        <thead><tr><th>Setup</th><th class="num">top 1</th><th class="num">top 3</th><th class="num">top 5</th><th class="num">MRR</th><th class="num">ms</th></tr></thead>
        <tbody>${rows.map(([key, row]) => `
          <tr class="${key === best[0] ? "best" : ""}">
            <td><span class="row-key">${escapeHtml(key)}</span>${escapeHtml(row.name)}</td>
            <td class="num">${row["hit@1"].toFixed(2)}</td>
            <td class="num">${row["hit@3"].toFixed(2)}</td>
            <td class="num">${row["hit@5"].toFixed(2)}</td>
            <td class="num">${row["mrr@10"].toFixed(3)}</td>
            <td class="num">${Math.round(row.ms_per_query)}</td>
          </tr>`).join("")}</tbody>
      </table>
    </section>

    <section class="panel">
      <h3>Refusing honestly</h3>
      <p>A question whose best passage scores below ${data.refusalThreshold} is refused before the LLM is called; ${data.answers.stoppedBeforeLLM} of ${data.answers.uncovered} off-syllabus questions were stopped that way, and the model refused the rest after reading the passages.</p>
    </section>

    <section class="panel">
      <h3>What is still wrong</h3>
      <p>Every answer the judge marked down, kept here rather than hidden.</p>
      <ul class="notes">${data.answers.notes.map((n) => `<li><strong>${escapeHtml(n.question)}</strong><br>${escapeHtml(n.reason)}</li>`).join("")}</ul>
    </section>`;
}

/* ---- chrome ---- */

function showView(name) {
  ["ask", "library", "progress", "evaluation"].forEach((view) => { $(`view-${view}`).hidden = view !== name; });
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.view === name));
  hideTools();
  if (name !== "ask") $("drawer").hidden = true; // the drawer belongs to the answers on the Ask page
  if (name === "library") loadLibrary();
  if (name === "progress") loadProgress();
  if (name === "evaluation") loadEvaluation();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("is-active", b.dataset.mode === mode));
  $("mode-hint").textContent = MODE_HINTS[mode];
}

function setColor(color) {
  store.write("color", safeColor(color));
  document.querySelectorAll("#swatches .swatch").forEach((s) => s.classList.toggle("is-active", s.dataset.color === safeColor(color)));
}

function setup() {
  $("subjects").innerHTML = SUBJECTS.map((s) => `<button class="subject ${s === "All" ? "is-active" : ""}" data-subject="${s}">${s}</button>`).join("");
  $("starters").innerHTML = STARTERS.map((s) => `<button class="starter" data-ask="${escapeHtml(s)}">${escapeHtml(s)}</button>`).join("");
  $("swatches").innerHTML = swatchRow();
  setColor(myColor());
  setMode("explain");
  drawRecent();

  $("composer").onsubmit = (event) => {
    event.preventDefault();
    if (state.busy) return; // keep what they typed until the current answer is done
    const question = $("question").value;
    $("question").value = "";
    ask(question);
  };

  $("nav").onclick = (event) => {
    const button = event.target.closest(".nav-item");
    if (button) showView(button.dataset.view);
  };

  $("modes").onclick = (event) => {
    const button = event.target.closest(".mode");
    if (button) setMode(button.dataset.mode);
  };

  $("subjects").onclick = (event) => {
    const button = event.target.closest(".subject");
    if (!button) return;
    state.subject = button.dataset.subject;
    document.querySelectorAll(".subject").forEach((b) => b.classList.toggle("is-active", b === button));
  };

  $("swatches").onclick = (event) => {
    const swatch = event.target.closest(".swatch");
    if (swatch) setColor(swatch.dataset.color);
  };

  $("new-thread").onclick = () => location.reload();
  $("drawer-close").onclick = () => { $("drawer").hidden = true; hideTools(); };
  $("clear-progress").onclick = () => {
    if (!confirm("Clear your questions, practice results, highlights and notes from this browser?")) return;
    ["activity", "practice", "marks"].forEach((key) => store.write(key, []));
    drawRecent();
    loadProgress();
  };

  document.addEventListener("mouseup", handleSelection);
  // a phone selects with a long press and drag handles, which never send a mouseup
  let settling;
  document.addEventListener("selectionchange", () => {
    clearTimeout(settling);
    settling = setTimeout(() => {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed && matchMedia("(pointer: coarse)").matches) handleSelection({ target: document.body });
    }, 400);
  });

  document.addEventListener("click", (event) => {
    const target = event.target;

    const asking = target.closest("[data-ask]");
    if (asking) {
      showView("ask");
      return ask(asking.dataset.ask);
    }

    const practising = target.closest("[data-practice]");
    if (practising) {
      showView("ask");
      const topic = practising.dataset.practice;
      return practice(`Explain ${topic}`, { label: topic, shown: topic });
    }

    const removing = target.closest("[data-remove-mark]");
    if (removing) return removeMark(removing.dataset.removeMark);

    const turnElement = target.closest(".turn");
    const record = turnElement && turns.get(Number(turnElement.dataset.turn));
    if (!record) return;

    const cite = target.closest(".cite, .source-pill");
    if (cite) return openSource(record.id, cite.dataset.source);

    const action = target.closest("[data-action]")?.dataset.action;
    if (action === "confused") return showStrategies(turnElement);
    // the search query stands on its own, a socratic reply like "is it the lock?" doesn't
    if (action === "practice") return practice(record.searchQuery || record.question);
    if (action === "reveal") {
      const card = target.closest(".practice");
      card.querySelector(".practice-a").hidden = false;
      target.closest(".actions").remove();
      return;
    }

    const grade = target.closest("[data-grade]");
    if (grade) return gradePractice(grade);

    const strategy = target.closest("[data-strategy]")?.dataset.strategy;
    if (strategy) ask(record.question, { mode: strategy, previous: record.answer, shown: STRATEGIES[strategy][1] });
  });
}

setup();
