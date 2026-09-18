const SUBJECTS = ["All", "DSA", "Operating Systems", "DBMS", "Computer Networks"];
// example questions from the evaluation's hand-checked set, so every one is known to be answerable
const STARTERS = {
  All: [
    "What conditions must hold for a deadlock to occur?",
    "TCP vs UDP: what's the difference?",
    "Why do databases use B-trees instead of binary search trees?",
    "What is a functional dependency?",
  ],
  DSA: [
    "How does quicksort work?",
    "Which data structure does BFS use, and how long does it take on a graph?",
    "Can I use Dijkstra's algorithm on a graph with negative edge weights?",
  ],
  "Operating Systems": [
    "What conditions must hold for a deadlock to occur?",
    "What is thrashing?",
    "What is a page fault and who handles it?",
  ],
  DBMS: [
    "What is a functional dependency?",
    "What is normalization in DBMS and why do we do it?",
    "What is the difference between a primary key and a foreign key?",
  ],
  "Computer Networks": [
    "TCP vs UDP: what's the difference?",
    "What is slow start in TCP?",
    "What does DNS do?",
  ],
};
const COLORS = [["#fbe7a2", "sand"], ["#cde3f6", "sky"], ["#cfded2", "sage"], ["#f5d9da", "rose"], ["#ddd8f2", "lilac"]];
const MODE_HINTS = {
  explain: "I teach you the topic, with the page behind every claim.",
  simple: "I explain it in plain words, still only from the books.",
  quiz: "I set you a marked quiz: multiple choice, fill in the blank and short answers.",
  socratic: "I guide you with questions instead of handing over the answer. Reply to keep going.",
};
const STRATEGIES = {
  analogy: ["Use an analogy", "Still confused. Try an analogy."],
  example: ["Walk through an example", "Still confused. Walk me through an example."],
  steps: ["Break it into steps", "Still confused. Break it into steps."],
};
// the placement staples of each subject, looked up in the books' own titles for the landing page
const CORE_TOPICS = {
  DSA: ["sorting", "binary search", "hash tables", "linked lists", "array-based lists", "heaps", "binary trees",
    "graph traversal", "shortest paths", "spanning trees", "dynamic programming", "greedy"],
  "Operating Systems": ["threads", "scheduling", "races", "mutexes", "semaphores", "deadlock", "virtual memory",
    "policies for virtual memory", "processes and protection", "disk space allocation", "atomic transactions", "remote procedure call"],
  DBMS: ["relational data model", "entity relationship data model", "er modelling", "integrity rules", "functional dependencies",
    "normalization", "structured query language", "data manipulation language", "atomic transactions", "write-ahead logging", "b-trees"],
  "Computer Networks": ["architecture", "framing", "error detection", "reliable transmission", "switching basics", "internet (ip)",
    "routing", "(udp)", "(tcp)", "tcp congestion control", "queuing disciplines", "cryptographic building blocks"],
};
// section titles that say nothing on their own, so the chapter's name goes with them
const VAGUE = new Set(["introduction", "summary", "overview", "perspective", "exercises", "notes", "discussion", "applications", "analysis"]);

const state = { mode: "explain", subject: "All", history: [], busy: false };
const turns = new Map(); // every answer keeps its own sources, so a citation always opens the right passage
let turnCount = 0;
let contents = null; // the books' tables of contents, fetched once
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
// The first `skip` characters repeat the passage before it, so a reader showing both hides them.
function renderPassage(text, terms, marks, skip = 0) {
  const evidence = [];
  const words = (terms || []).filter((t) => t.length > 3).map(escapeRegExp);
  if (words.length) {
    for (const m of text.matchAll(new RegExp(`\\b(?:${words.join("|")})\\w{0,3}\\b`, "gi"))) {
      evidence.push([m.index, m.index + m[0].length]);
    }
  }
  const cuts = [...new Set([0, skip, text.length, ...evidence.flat(), ...marks.flatMap((m) => [m.start, m.end])])]
    .filter((p) => p >= 0 && p <= text.length)
    .sort((a, b) => a - b);

  let html = "";
  for (let i = 0; i < cuts.length - 1; i++) {
    const [start, end] = [cuts[i], cuts[i + 1]];
    let piece = escapeHtml(text.slice(start, end));
    if (end <= skip) {
      html += `<span class="repeat">${piece}</span>`;
      continue;
    }
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

const citeButton = (number) => (Number(number) ? `<button class="cite" data-source="${Number(number)}">${Number(number)}</button>` : "");

/* ---- passages on screen ---- */

// Every passage drawn anywhere (the source drawer, a topic, a search result) is registered here,
// so a highlight made in one place is redrawn in every other place showing the same passage.
const views = new Map();
let viewCount = 0;
const marksFor = (key) => store.read("marks", []).filter((m) => m.key === key);

function passageMarkup(source, terms = [], skip = 0) {
  const id = ++viewCount;
  const view = { source, terms, skip, key: passageKey(source) };
  views.set(id, view);
  return `<div class="passage" data-view="${id}">${renderPassage(source.text, terms, marksFor(view.key), skip)}</div>`;
}

function redrawPassages() {
  document.querySelectorAll(".passage[data-view]").forEach((element) => {
    const view = views.get(Number(element.dataset.view));
    if (view) element.innerHTML = renderPassage(view.source.text, view.terms, marksFor(view.key), view.skip);
  });
  if (openPassage && !$("drawer").hidden) drawDrawerNotes();
  if (!$("view-progress").hidden) loadProgress();
}

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

// a turn starts the same way whatever it is: the question, a status line, and the busy flag
function startTurn(record, shown, status) {
  state.busy = true;
  $("send").disabled = true;
  document.querySelector(".welcome")?.remove();
  record.id = ++turnCount;
  record.subject = state.subject;
  turns.set(record.id, record);
  const turn = newTurn(record.id, shown);
  if (status) turn.querySelector(".status span").textContent = status;
  return turn;
}

function endTurn() {
  state.busy = false;
  $("send").disabled = false;
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
    // when the subject picked the meaning, the other one is a click away
    const others = (data.alternatives || []).filter((a) => a.term === acronym);
    const offers = others.map((a) => ` <button class="link" data-ask="${escapeHtml(a.question)}" data-subject="${escapeHtml(a.subject)}">Meant ${escapeHtml(a.meaning)}?</button>`);
    lines.push(`Reading <b>${escapeHtml(acronym)}</b> as ${escapeHtml(meaning)}${others.length ? ", going by the subject you picked." : ""}${offers.join("")}`);
  }
  return lines.map((l) => `<p class="note-line">${l}</p>`).join("");
}

function showRefusal(turn, record, data) {
  const card = turn.querySelector(".answer, .refusal");
  card.className = "refusal";
  const narrowed = record.subject && record.subject !== "All";
  card.innerHTML = `
    ${noteLines(data)}
    <h3>Not in the library</h3>
    <p>I couldn't find a passage in these ${narrowed ? `${escapeHtml(record.subject)} books` : "nine books"} that answers this, so answering would mean making something up.</p>
    ${narrowed ? `<p class="subject-hint">Only the ${escapeHtml(record.subject)} books were searched. <button class="link" data-ask="${escapeHtml(record.question)}" data-subject="All">Search every subject</button></p>` : ""}
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
    slot.innerHTML = `<span class="suggest-label">Try rephrasing it, or browse the Topics to see what the books cover.</span>`;
    return;
  }
  if (!data.topics.length) {
    slot.innerHTML = `<span class="suggest-label">Nothing in these books is close to this. They cover data structures and algorithms, operating systems, databases and computer networks.</span>`;
    return;
  }
  const label = { typo: "Did you mean:", lookalike: "Nothing here covers that word. If you meant a similar-looking term:" }[data.kind]
    || "Closest topics in the library:";
  // the topics come from the whole library, so asking about one searches all of it
  const record = turns.get(Number(turn.dataset.turn));
  const widen = record?.subject && record.subject !== "All" ? ' data-subject="All"' : "";
  slot.innerHTML = `
    <span class="suggest-label">${label}</span>
    <div class="suggest-row">${data.topics.map((t) => `<button class="suggestion" data-ask="${escapeHtml(`Explain ${t}`)}"${widen}>${escapeHtml(t)}</button>`).join("")}</div>`;
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
  if (mode === "quiz") return quiz(question, { shown });
  if (state.busy || !question.trim()) return;

  const record = { question, mode, answer: "", sources: [], terms: [], suggest: null };
  const turn = startTurn(record, shown);
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
      body: JSON.stringify({ question, history: state.history.slice(-6), mode, subject: record.subject, previous }),
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
          record.info = { corrections: data.corrections, expansions: data.expansions, alternatives: data.alternatives };
          if (!data.grounded) {
            showRefusal(turn, record, data);
          } else {
            status.querySelector("span").textContent = `reading ${data.sources.length} passages…`;
            turn.querySelector(".sources").innerHTML = data.sources.map(pillMarkup).join("");
            const notes = noteLines(data);
            // a follow-up rewritten to stand on its own is shown, a change of capital letters isn't
            const rewritten = normalise(data.searchQuery) !== normalise(question) && !(data.expansions || []).length && !(data.corrections || []).length
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
  endTurn();
}

async function postJson(url, payload) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `the server returned ${response.status}`);
  }
  return response.json();
}

/* ---- quick practice ---- */

// label is the topic name progress is kept under; by default the section the questions came from
async function practice(topic, { label = "", shown = topic } = {}) {
  if (state.busy || !topic) return;
  const record = { question: topic, mode: "practice", answer: "", sources: [], terms: [] };
  const turn = startTurn(record, `Practice questions: ${shown}`, "writing questions from the same passages…");
  const status = turn.querySelector(".status");
  const body = turn.querySelector(".answer-body");

  try {
    const data = await postJson("/api/practice", { topic, subject: record.subject });
    status.hidden = true;
    record.sources = data.sources;
    label = label || shortSection(data.sources[0]?.section) || topic;
    if (!data.questions.length) {
      body.innerHTML = `<p class="muted">I couldn't write practice questions for this topic from the books.</p>`;
    } else {
      const batch = Date.now().toString(36); // turn numbers restart on reload, so ids come from the clock
      body.innerHTML = data.questions.map((q, i) => practiceMarkup(q, i, `${batch}-${i}`, label)).join("");
      turn.querySelector(".sources").innerHTML = data.sources.map(pillMarkup).join("");
    }
  } catch (error) {
    status.hidden = true;
    body.innerHTML = `<p class="muted">Couldn't write practice questions: ${escapeHtml(error.message)}.</p>`;
  }
  endTurn();
}

function practiceMarkup(q, i, id, label) {
  return `
    <div class="practice" data-id="${id}" data-topic="${escapeHtml(label)}" data-question="${escapeHtml(q.question)}"
         data-answer="${escapeHtml(q.answer)}" data-explanation="${escapeHtml(q.explanation || "")}">
      <p class="practice-q">${i + 1}. ${escapeHtml(q.question)}</p>
      <textarea placeholder="Try answering first (optional)"></textarea>
      <div class="actions"><button class="chip" data-action="reveal">Reveal the answer</button></div>
      <div class="practice-a" hidden>
        <p>${escapeHtml(q.answer)} ${citeButton(q.source)}</p>
        <div class="grade">How did you do?
          <button class="chip" data-grade="got">I got it</button>
          <button class="chip" data-grade="missed">Not quite</button>
        </div>
        <div class="why" hidden>
          ${q.explanation ? `<p><b>Why:</b> ${escapeHtml(q.explanation)} ${citeButton(q.source)}</p>` : ""}
          <div class="why-actions">
            <button class="chip" data-action="detail">Explain in detail</button>
            <button class="icon" data-action="dismiss" title="I understand it now" aria-label="Close, I understand it now">✕</button>
          </div>
        </div>
      </div>
    </div>`;
}

function gradePractice(button) {
  const card = button.closest(".practice");
  const missed = button.dataset.grade === "missed";
  card.querySelectorAll("[data-grade]").forEach((b) => b.classList.toggle("is-picked", b === button));
  card.querySelector(".why").hidden = !missed; // a wrong answer gets the reasoning, a right one doesn't need it
  // changing your mind replaces the earlier grade for the same question
  const results = store.read("practice", []).filter((r) => r.id !== card.dataset.id);
  results.push({
    id: card.dataset.id, question: card.dataset.question, answer: card.dataset.answer, explanation: card.dataset.explanation,
    topic: card.dataset.topic, result: button.dataset.grade, time: Date.now(),
  });
  store.write("practice", results.slice(-500));
}

/* ---- quiz ---- */

async function quiz(topic, { shown = topic } = {}) {
  if (state.busy || !topic.trim()) return;
  const record = { question: topic, mode: "quiz", answer: "", sources: [], terms: [] };
  const turn = startTurn(record, `Quiz: ${shown}`, "writing a quiz from the passages…");
  const status = turn.querySelector(".status");
  const body = turn.querySelector(".answer-body");

  try {
    const data = await postJson("/api/quiz", { topic, subject: record.subject });
    status.hidden = true;
    if (data.clarify) {
      showClarify(turn, data.clarify);
    } else if (!data.questions.length) {
      body.innerHTML = `<p class="muted">I couldn't write a quiz on this from the books. Try one of the topics they cover, listed under Topics.</p>`;
    } else {
      record.sources = data.sources;
      record.quiz = {
        id: Date.now().toString(36),
        topic: shortSection(data.sources[0]?.section) || topic,
        questions: data.questions,
        results: data.questions.map(() => null),
      };
      body.innerHTML = quizMarkup(record.quiz);
      turn.querySelector(".sources").innerHTML = data.sources.map(pillMarkup).join("");
    }
  } catch (error) {
    status.hidden = true;
    body.innerHTML = `<p class="muted">Couldn't write a quiz: ${escapeHtml(error.message)}.</p>`;
  }
  endTurn();
}

function quizMarkup(quiz) {
  const total = quiz.questions.length;
  return `
    <div class="quiz">
      <div class="quiz-head"><b>${total} marks</b><span class="quiz-score">0 of ${total} answered</span></div>
      ${quiz.questions.map(questionMarkup).join("")}
      <div class="quiz-result" hidden></div>
    </div>`;
}

function questionMarkup(q, i) {
  const feedback = `<div class="q-feedback" hidden></div>`;
  if (q.type === "mcq") {
    return `
      <div class="q" data-index="${i}">
        <p class="q-text">${i + 1}. ${escapeHtml(q.question)}</p>
        <div class="options">${q.options.map((o, j) => `<button class="option" data-option="${j}"><b>${"ABCDE"[j]}</b><span>${escapeHtml(o)}</span></button>`).join("")}</div>
        ${feedback}
      </div>`;
  }
  if (q.type === "blank") {
    const [before, after] = q.question.split("____");
    return `
      <div class="q" data-index="${i}">
        <p class="q-text">${i + 1}. ${escapeHtml(before)}<input class="blank" autocomplete="off" aria-label="Fill in the blank">${escapeHtml(after)}</p>
        <div class="actions"><button class="chip" data-action="check">Check</button></div>
        ${feedback}
      </div>`;
  }
  return `
    <div class="q" data-index="${i}">
      <p class="q-text">${i + 1}. ${escapeHtml(q.question)}</p>
      <textarea placeholder="Your answer"></textarea>
      <div class="actions"><button class="chip" data-action="show-answer">Show the answer</button></div>
      ${feedback}
    </div>`;
}

// typed answers are compared loosely: case, punctuation and an article don't matter, a typo or two doesn't either
const normalise = (text) => text.toLowerCase().replace(/[^a-z0-9]+/g, " ").replace(/\b(a|an|the)\b/g, " ").replace(/\s+/g, " ").trim();

function similarity(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const above = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] === b[j - 1] ? 0 : 1));
      diagonal = above;
    }
  }
  return 1 - row[b.length] / Math.max(a.length, b.length, 1);
}

function blankMatches(typed, q) {
  const got = normalise(typed);
  if (!got) return false;
  return [q.answer, ...(q.accept || [])].some((expected) => {
    const want = normalise(expected);
    if (!want) return false;
    const extra = got.split(" ").length - want.split(" ").length;
    return got === want || similarity(got, want) >= 0.8 || (extra <= 2 && ` ${got} `.includes(` ${want} `));
  });
}

function quizFeedback(questionElement, q, right, lead = "") {
  const box = questionElement.querySelector(".q-feedback");
  box.hidden = false;
  box.className = `q-feedback ${right ? "right" : "wrong"}`;
  box.innerHTML = `
    <p><b>${right ? "Correct." : "Not quite."}</b> ${lead}${escapeHtml(q.explanation)} ${citeButton(q.source)}</p>
    ${right ? "" : `<button class="chip" data-action="detail">Explain in detail</button>`}`;
}

function answerQuiz(turn, record, questionElement, action) {
  const quiz = record.quiz;
  const index = Number(questionElement.dataset.index);
  const q = quiz.questions[index];
  if (quiz.results[index] !== null && action.kind !== "grade") return; // answered already

  if (action.kind === "option") {
    const right = action.option === q.answer;
    questionElement.querySelectorAll(".option").forEach((button, j) => {
      button.disabled = true;
      button.classList.toggle("right", j === q.answer);
      button.classList.toggle("wrong", j === action.option && !right);
    });
    quiz.results[index] = right;
    quizFeedback(questionElement, q, right);
  } else if (action.kind === "check") {
    const input = questionElement.querySelector(".blank");
    if (!input.value.trim()) return input.focus();
    const right = blankMatches(input.value, q);
    input.disabled = true;
    input.classList.add(right ? "right" : "wrong");
    questionElement.querySelector(".actions").remove();
    quiz.results[index] = right;
    quizFeedback(questionElement, q, right, right ? "" : `The answer is <b>${escapeHtml(q.answer)}</b>. `);
  } else if (action.kind === "show") {
    // a written answer can't be marked by string matching, so the student marks it against the model answer
    questionElement.querySelector(".actions").remove();
    questionElement.querySelector("textarea").disabled = true;
    const box = questionElement.querySelector(".q-feedback");
    box.hidden = false;
    box.innerHTML = `
      <p><b>Model answer:</b> ${escapeHtml(q.answer)} ${citeButton(q.source)}</p>
      <div class="grade">Did yours say the same?
        <button class="chip" data-quiz-grade="got">Yes</button>
        <button class="chip" data-quiz-grade="missed">Not quite</button>
      </div>`;
    return;
  } else if (action.kind === "grade") {
    const right = action.grade === "got";
    quiz.results[index] = right;
    quizFeedback(questionElement, q, right, `The model answer: ${escapeHtml(q.answer)} `);
  }
  updateQuizScore(turn, record);
}

function updateQuizScore(turn, record) {
  const { questions, results } = record.quiz;
  const answered = results.filter((r) => r !== null).length;
  const score = results.filter((r) => r === true).length;
  turn.querySelector(".quiz-score").textContent = `${answered} of ${questions.length} answered · ${score} right`;
  if (answered < questions.length) return;

  // the sections behind the missed questions are what to read next
  const missed = questions.filter((_, i) => results[i] === false);
  const review = new Map();
  for (const q of missed) {
    const source = record.sources[q.source - 1];
    if (source) review.set(source.section, source);
  }
  const box = turn.querySelector(".quiz-result");
  box.hidden = false;
  box.innerHTML = `
    <p class="score">${score}/${questions.length}</p>
    ${review.size
      ? `<p>Review:</p><div class="topic-row">${[...review.values()].map((s) => `<button class="topic" data-topic-book="${escapeHtml(s.docId)}" data-topic-path="${escapeHtml(s.section)}">${escapeHtml(shortSection(s.section))}</button>`).join("")}</div>`
      : `<p>Full marks. Try a quiz on a harder topic, or switch to Socratic mode to go deeper.</p>`}`;

  const quizzes = store.read("quizzes", []).filter((q) => q.id !== record.quiz.id);
  quizzes.push({
    id: record.quiz.id,
    topic: record.quiz.topic,
    question: record.question,
    score,
    total: questions.length,
    missed: missed.map((q) => ({ question: q.question, answer: q.type === "mcq" ? q.options[q.answer] : q.answer, explanation: q.explanation })),
    time: Date.now(),
  });
  store.write("quizzes", quizzes.slice(-100));
}

/* ---- source drawer ---- */

function openSource(turnId, number) {
  const record = turns.get(Number(turnId));
  const source = record?.sources.find((s) => String(s.number) === String(number));
  if (!source) return;
  openPassage = { source, terms: record.terms };
  hideTools();
  drawDrawer();
  $("drawer").hidden = false;
}

function drawDrawer() {
  const { source, terms } = openPassage;
  $("drawer-title").textContent = `Source ${source.number}`;
  $("drawer-body").innerHTML = `
    <div class="meta">
      <span class="book">${escapeHtml(source.book)}</span>
      <span class="where">${escapeHtml(source.section)}</span>
      <span class="page">page ${escapeHtml(source.page)} · reranker ${source.scores.rerank === null ? "—" : source.scores.rerank.toFixed(3)}</span>
    </div>
    ${passageMarkup(source, terms)}
    <p class="drawer-legend"><span class="evidence">Blue</span> marks the words your question matched. Select any text to highlight it in your colour or add a note.</p>
    <div class="drawer-notes" id="drawer-notes"></div>`;
  drawDrawerNotes();
}

function drawDrawerNotes() {
  const marks = marksFor(passageKey(openPassage.source));
  $("drawer-notes").innerHTML = marks.length ? `<h4>Your highlights on this page</h4><div class="list">${marks.map(markItem).join("")}</div>` : "";
}

/* ---- personal highlights ---- */

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

function saveMark(view, start, end, color, note) {
  const { source, key } = view;
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
  redrawPassages();
}

function updateMark(id, changes) {
  store.write("marks", store.read("marks", []).map((m) => (m.id === id ? { ...m, ...changes } : m)));
  hideTools();
  redrawPassages();
}

function removeMark(id) {
  store.write("marks", store.read("marks", []).filter((m) => m.id !== id));
  hideTools();
  redrawPassages();
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

function newMarkTools(rect, view, start, end) {
  const tools = $("mark-tools");
  tools.innerHTML = `<div class="row">${swatchRow()}</div><div class="row"><button class="text" data-act="note">Highlight with a note</button></div>`;
  tools.onclick = (event) => {
    const color = event.target.closest(".swatch")?.dataset.color;
    if (color) return saveMark(view, start, end, color, "");
    if (event.target.closest('[data-act="note"]')) noteEditor(rect, { view, start, end });
  };
  placeTools(rect);
}

function noteEditor(rect, target) {
  const tools = $("mark-tools");
  const text = target.mark ? target.mark.text : target.view.source.text.slice(target.start, target.end);
  tools.innerHTML = `
    <p class="quote">“${escapeHtml(text.length > 90 ? `${text.slice(0, 90)}…` : text)}”</p>
    <textarea placeholder="Your note">${escapeHtml(target.mark?.note || "")}</textarea>
    <div class="row"><button class="text" data-act="save">Save</button><button class="text" data-act="cancel">Cancel</button></div>`;
  tools.onclick = (event) => {
    if (event.target.closest('[data-act="cancel"]')) return hideTools();
    if (!event.target.closest('[data-act="save"]')) return;
    const note = tools.querySelector("textarea").value.trim();
    if (target.mark) updateMark(target.mark.id, { note });
    else saveMark(target.view, target.start, target.end, myColor(), note);
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

const passageOf = (node) => (node?.nodeType === Node.TEXT_NODE ? node.parentElement : node)?.closest?.(".passage[data-view]");

function handleSelection(event) {
  if (event.target.closest?.("#mark-tools")) return;
  const selection = window.getSelection();
  const passage = selection && !selection.isCollapsed ? passageOf(selection.anchorNode) : null;
  if (!passage || passageOf(selection.focusNode) !== passage) {
    const mark = event.target.closest?.("mark.mine");
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
  newMarkTools(range.getBoundingClientRect(), views.get(Number(passage.dataset.view)), start, end);
}

/* ---- topics ---- */

async function loadContents() {
  if (!contents) contents = (await (await fetch("/api/topics")).json()).books;
  return contents;
}

// a book's chapters for one subject: the OS book counts for DBMS only through its chapter on transactions
const chaptersFor = (book, subject) => book.chapters.filter((c) => subject === "All" || c.subjects.includes(subject));
const booksFor = (subject) => contents.filter((b) => chaptersFor(b, subject).length);

// front matter like "Why This Book?" isn't a topic anyone studies
const isFrontMatter = (book, chapter) => chapter.title === book.title || /edition|why this book|lecture notes/i.test(chapter.title);

// "Introduction" alone says nothing, so it is asked about with its chapter
const topicName = (title, chapter) => (chapter && VAGUE.has(title.toLowerCase()) ? `${chapter} (${title})` : title);

async function loadTopics() {
  $("topics").innerHTML = `<p class="muted">Loading the tables of contents…</p>`;
  await loadContents();
  const query = $("topic-query").value.trim();
  $("topics").innerHTML = query ? topicMatches(query) : topicIndex();
}

function topicIndex() {
  const books = booksFor(state.subject);
  return `
    <p class="lead">${state.subject === "All" ? "Every chapter of the nine books" : `The chapters of the ${escapeHtml(state.subject)} books`}, as the books themselves list them. Open a section to read it, then ask about it, practise it or take a quiz.</p>
    ${books.map((book) => `
      <section class="book-block">
        <h3>${escapeHtml(book.title)}</h3>
        ${chaptersFor(book, state.subject).filter((c) => !isFrontMatter(book, c)).map((c) => chapterMarkup(book, c)).join("")}
      </section>`).join("")}`;
}

function chapterMarkup(book, chapter) {
  if (!chapter.sections.length) {
    return `<button class="chapter-row" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(chapter.path)}"><span>${escapeHtml(chapter.title)}</span><small>p.${escapeHtml(chapter.pages)}</small></button>`;
  }
  return `
    <details class="chapter">
      <summary><span>${escapeHtml(chapter.title)}</span><small>p.${escapeHtml(chapter.pages)} · ${chapter.sections.length} sections</small></summary>
      <div class="topic-row">
        <button class="topic whole" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(chapter.path)}">The whole chapter</button>
        ${chapter.sections.map((s) => `<button class="topic" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(s.path)}">${escapeHtml(s.title)}</button>`).join("")}
      </div>
    </details>`;
}

function topicMatches(query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const hits = [];
  for (const book of booksFor(state.subject)) {
    for (const chapter of chaptersFor(book, state.subject)) {
      for (const item of [chapter, ...chapter.sections]) {
        if (words.every((w) => item.title.toLowerCase().includes(w))) hits.push({ book, item });
      }
    }
  }
  return `
    <p class="lead">${hits.length ? `Sections whose titles match “${escapeHtml(query)}”.` : `No section title matches “${escapeHtml(query)}”.`}
      Search the books to find every passage that talks about it.</p>
    ${hits.length ? `<div class="topic-row">${hits.slice(0, 40).map(({ book, item }) => `
      <button class="topic" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(item.path)}">${escapeHtml(item.title)} <small>${escapeHtml(book.title)}</small></button>`).join("")}</div>` : ""}
    <div id="search-results"></div>`;
}

async function searchBooks(query) {
  if (!query) return;
  if (!$("search-results")) $("topics").innerHTML = topicMatches(query);
  const box = $("search-results");
  box.innerHTML = `<p class="muted">Searching the books…</p>`;
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&subject=${encodeURIComponent(state.subject)}`);
    if (!response.ok) throw new Error(`the server returned ${response.status}`);
    const data = await response.json();
    const fixed = data.corrections.map(([typed, meant]) => `Showing results for <b>${escapeHtml(meant)}</b> (you typed ${escapeHtml(typed)}). `).join("");
    box.innerHTML = data.passages.length
      ? `<h3 class="results-title">Passages about “${escapeHtml(query)}”</h3>${fixed ? `<p class="note-line">${fixed}</p>` : ""}
         ${data.passages.map((p) => passageCard(p, data.terms)).join("")}`
      : `<p class="lead">Nothing in ${state.subject === "All" ? "the books" : `the ${escapeHtml(state.subject)} books`} matches “${escapeHtml(query)}”. Try another word, or ask Anchor.</p>`;
  } catch (error) {
    box.innerHTML = `<p class="muted">Couldn't search: ${escapeHtml(error.message)}.</p>`;
  }
}

function passageCard(source, terms) {
  const title = shortSection(source.section);
  return `
    <article class="passage-card">
      <div class="meta"><span class="book">${escapeHtml(source.book)}</span><span class="where">${escapeHtml(source.section)}</span><span class="page">page ${escapeHtml(source.page)}</span></div>
      ${passageMarkup(source, terms)}
      <div class="topic-actions">
        <button class="chip" data-topic-book="${escapeHtml(source.docId)}" data-topic-path="${escapeHtml(source.section)}">Read the whole section</button>
        <button class="chip" data-ask="${escapeHtml(`Explain ${title}`)}" data-mode="explain">Ask Anchor about this</button>
      </div>
    </article>`;
}

async function openTopic(bookId, path) {
  showView("topics", { load: false });
  $("topics").innerHTML = `<p class="muted">Opening…</p>`;
  await loadContents();
  const book = contents.find((b) => b.id === bookId);
  const chapter = book?.chapters.find((c) => c.path === path);
  try {
    const response = await fetch(`/api/topic?book=${encodeURIComponent(bookId)}&path=${encodeURIComponent(path)}`);
    if (!response.ok) throw new Error(`the server returned ${response.status}`);
    drawTopic(await response.json(), book, chapter);
  } catch (error) {
    $("topics").innerHTML = `<p class="muted">Couldn't open this topic: ${escapeHtml(error.message)}.</p>`;
  }
  $("topics").scrollTop = 0;
}

function drawTopic(data, book, chapter) {
  const parts = data.path.split(" > ");
  const title = shortSection(data.path);
  const parent = parts.length > 1 ? shortSection(parts.slice(0, -1).join(" > ")) : "";
  const name = topicName(title, parent);
  $("topics").innerHTML = `
    <div class="reader-head">
      <button class="link back" data-back-to-topics>← All topics</button>
      <h2>${escapeHtml(title)}</h2>
      <p class="muted">${escapeHtml(data.book)}${parent ? ` · ${escapeHtml(parent)}` : ""} · pages ${escapeHtml(data.pages)} · ${data.total} passages${data.total > data.passages.length ? `, the first ${data.passages.length} shown` : ""}</p>
      <div class="topic-actions">
        <button class="chip" data-ask="${escapeHtml(`Explain ${name}`)}" data-mode="explain">Ask Anchor about this</button>
        <button class="chip" data-practice="${escapeHtml(name)}">Practice</button>
        <button class="chip" data-quiz="${escapeHtml(name)}">Quiz me</button>
      </div>
      ${chapter?.sections.length ? `<div class="topic-row">${chapter.sections.map((s) => `<button class="topic" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(s.path)}">${escapeHtml(s.title)}</button>`).join("")}</div>` : ""}
      <p class="drawer-legend">Select any text to highlight it in your colour or add a note.</p>
    </div>
    ${data.passages.map((p) => `
      <div class="reader-passage">
        <span class="page-label">p.${escapeHtml(p.page)}</span>
        ${passageMarkup(p, [], p.skip)}
      </div>`).join("")}`;
}

/* ---- landing page ---- */

function drawWelcome() {
  if (!$("starters")) return; // the conversation has started
  $("starters").innerHTML = (STARTERS[state.subject] || STARTERS.All)
    .map((s) => `<button class="starter" data-ask="${escapeHtml(s)}">${escapeHtml(s)}</button>`).join("");
  drawWelcomeTopics();
}

async function drawWelcomeTopics() {
  const box = $("welcome-topics");
  if (!box) return;
  if (state.subject === "All") {
    box.innerHTML = `
      <p class="welcome-label">Or pick a subject to see what its books cover</p>
      <div class="topic-row">${SUBJECTS.slice(1).map((s) => `<button class="topic" data-subject-pick="${escapeHtml(s)}">${escapeHtml(s)}</button>`).join("")}</div>`;
    return;
  }
  const subject = state.subject;
  await loadContents();
  if (subject !== state.subject || !$("welcome-topics")) return; // the subject changed while loading
  const everything = booksFor(subject).flatMap((book) => chaptersFor(book, subject)
    .flatMap((chapter) => [chapter, ...chapter.sections].map((item) => ({ book, item }))));
  // for each staple, the most specific title that has it: "Deadlock" rather than "Synchronization and Deadlocks"
  const picked = [];
  for (const word of CORE_TOPICS[subject] || []) {
    const match = everything
      .filter(({ item }) => item.title.toLowerCase().includes(word) && !picked.some((p) => p.item.title === item.title))
      .sort((a, b) => a.item.title.length - b.item.title.length)[0];
    if (match) picked.push(match);
  }
  box.innerHTML = `
    <p class="welcome-label">Topics in the ${escapeHtml(subject)} books</p>
    <div class="topic-row">
      ${picked.map(({ book, item }) => `<button class="topic" data-topic-book="${escapeHtml(book.id)}" data-topic-path="${escapeHtml(item.path)}">${escapeHtml(item.title)}</button>`).join("")}
      <button class="topic more" data-view-link="topics">All topics →</button>
    </div>`;
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

const shortDate = (time) => new Date(time).toLocaleDateString(undefined, { day: "numeric", month: "short" });

function loadProgress() {
  const activity = store.read("activity", []);
  const results = store.read("practice", []);
  const quizzes = store.read("quizzes", []);
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
  const weakPractice = [...recent.entries()]
    .filter(([, list]) => list.filter((r) => r === "missed").length * 2 >= list.length && list.includes("missed"))
    .map(([topic]) => topic);
  // and a quiz on its latest attempt
  const latestQuiz = new Map();
  for (const q of quizzes) latestQuiz.set(q.topic, q);
  const weakQuizzes = [...latestQuiz.values()].filter((q) => q.score / q.total < 0.6);
  const got = results.filter((r) => r.result === "got").length;
  const quizAverage = quizzes.length ? Math.round((quizzes.reduce((sum, q) => sum + q.score / q.total, 0) / quizzes.length) * 100) : null;

  const notQuite = [
    ...results.filter((r) => r.result === "missed").map((r) => ({ ...r, from: "practice" })),
    ...quizzes.flatMap((q) => (q.missed || []).map((m) => ({ ...m, topic: q.topic, time: q.time, from: "quiz" }))),
  ].sort((a, b) => b.time - a.time).slice(0, 12);

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
      <div class="metric"><b>${quizAverage === null ? "—" : `${quizAverage}%`}</b><span>average quiz score${quizzes.length ? ` over ${quizzes.length} quiz${quizzes.length > 1 ? "zes" : ""}` : ""}</span></div>
      <div class="metric"><b>${marks.length}</b><span>highlights and notes</span></div>
    </div>

    <section class="panel">
      <h3>Worth another look</h3>
      <p>Topics where you missed at least half of your last five practice answers, or scored under 60% in the latest quiz.</p>
      ${weakPractice.length || weakQuizzes.length ? `<div class="topic-row">
        ${weakPractice.map((t) => `<button class="topic weak" data-practice="${escapeHtml(t)}">${escapeHtml(t)} · practise again</button>`).join("")}
        ${weakQuizzes.map((q) => `<button class="topic weak" data-quiz="${escapeHtml(q.question || q.topic)}">${escapeHtml(q.topic)} · retake quiz</button>`).join("")}
      </div>` : `<p class="empty">Nothing yet. Practise or take a quiz, and weak spots show up here.</p>`}
    </section>

    <section class="panel">
      <h3>Quiz scores</h3>
      <p>Your latest quizzes. Retake one to see if it has stuck.</p>
      ${quizzes.length ? `<div class="list">${[...quizzes].reverse().slice(0, 8).map((q) => `
        <div class="item score-item">
          <b class="score-badge ${q.score / q.total < 0.6 ? "low" : ""}">${q.score}/${q.total}</b>
          <div class="body"><p class="quote-text">${escapeHtml(q.topic)}</p><span class="meta-line">${shortDate(q.time)}</span></div>
          <button class="chip" data-quiz="${escapeHtml(q.question || q.topic)}">Retake</button>
        </div>`).join("")}</div>` : `<p class="empty">Switch to Quiz me, or pick Quiz me on any topic page.</p>`}
    </section>

    <section class="panel">
      <h3>Answers you marked not quite</h3>
      <p>The questions you missed in practice and quizzes, with the answer, newest first.</p>
      ${notQuite.length ? `<div class="list">${notQuite.map((m) => `
        <div class="item">
          <div class="body">
            <p class="quote-text">${escapeHtml(m.question)}</p>
            ${m.answer ? `<p class="note-text"><b>Answer:</b> ${escapeHtml(m.answer)}</p>` : ""}
            ${m.explanation ? `<p class="note-text">${escapeHtml(m.explanation)}</p>` : ""}
            <span class="meta-line">${escapeHtml(m.topic || "")} · ${m.from} · ${shortDate(m.time)}</span>
          </div>
          <button class="chip" data-ask="${escapeHtml(m.question)}" data-mode="explain">Explain in detail</button>
        </div>`).join("")}</div>` : `<p class="empty">Nothing missed yet.</p>`}
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
        : `<p class="empty">Open any source or topic and select text to highlight it or add a note.</p>`}
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

function showView(name, { load = true } = {}) {
  ["ask", "topics", "library", "progress", "evaluation"].forEach((view) => { $(`view-${view}`).hidden = view !== name; });
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.view === name));
  hideTools();
  if (name !== "ask") $("drawer").hidden = true; // the drawer belongs to the answers on the Ask page
  if (!load) return;
  if (name === "topics") loadTopics();
  if (name === "library") loadLibrary();
  if (name === "progress") loadProgress();
  if (name === "evaluation") loadEvaluation();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("is-active", b.dataset.mode === mode));
  $("mode-hint").textContent = MODE_HINTS[mode];
}

function setSubject(subject) {
  if (!SUBJECTS.includes(subject)) return;
  state.subject = subject;
  document.querySelectorAll(".subject").forEach((b) => b.classList.toggle("is-active", b.dataset.subject === subject));
  drawWelcome();
  if (!$("view-topics").hidden && !document.querySelector(".reader-head")) loadTopics();
}

function setColor(color) {
  store.write("color", safeColor(color));
  document.querySelectorAll("#swatches .swatch").forEach((s) => s.classList.toggle("is-active", s.dataset.color === safeColor(color)));
}

function setup() {
  $("subjects").innerHTML = SUBJECTS.map((s) => `<button class="subject ${s === "All" ? "is-active" : ""}" data-subject="${s}">${s}</button>`).join("");
  $("swatches").innerHTML = swatchRow();
  setColor(myColor());
  setMode("explain");
  drawWelcome();
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
    if (button) setSubject(button.dataset.subject);
  };

  $("swatches").onclick = (event) => {
    const swatch = event.target.closest(".swatch");
    if (swatch) setColor(swatch.dataset.color);
  };

  // titles filter as you type; searching the books themselves waits for Enter
  let typing;
  $("topic-query").oninput = () => {
    clearTimeout(typing);
    typing = setTimeout(loadTopics, 150);
  };
  $("topic-search").onsubmit = async (event) => {
    event.preventDefault();
    await loadContents();
    searchBooks($("topic-query").value.trim());
  };

  $("new-thread").onclick = () => location.reload();
  $("drawer-close").onclick = () => { $("drawer").hidden = true; hideTools(); };
  $("clear-progress").onclick = () => {
    if (!confirm("Clear your questions, practice and quiz results, highlights and notes from this browser?")) return;
    ["activity", "practice", "quizzes", "marks"].forEach((key) => store.write(key, []));
    drawRecent();
    redrawPassages();
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

  // Enter in a fill-in-the-blank checks it
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches?.("input.blank")) {
      event.preventDefault();
      event.target.closest(".q").querySelector('[data-action="check"]')?.click();
    }
  });

  document.addEventListener("click", (event) => {
    const target = event.target;

    const asking = target.closest("[data-ask]");
    if (asking) {
      if (asking.dataset.subject) setSubject(asking.dataset.subject);
      showView("ask");
      return ask(asking.dataset.ask, { mode: asking.dataset.mode || state.mode });
    }

    const practising = target.closest("[data-practice]");
    if (practising) {
      showView("ask");
      const topic = practising.dataset.practice;
      return practice(`Explain ${topic}`, { label: topic, shown: topic });
    }

    const quizzing = target.closest("[data-quiz]");
    if (quizzing) {
      showView("ask");
      return quiz(quizzing.dataset.quiz);
    }

    const topic = target.closest("[data-topic-book]");
    if (topic) return openTopic(topic.dataset.topicBook, topic.dataset.topicPath);

    if (target.closest("[data-back-to-topics]")) return loadTopics();

    const picked = target.closest("[data-subject-pick]");
    if (picked) return setSubject(picked.dataset.subjectPick);

    const link = target.closest("[data-view-link]");
    if (link) return showView(link.dataset.viewLink);

    const removing = target.closest("[data-remove-mark]");
    if (removing) return removeMark(removing.dataset.removeMark);

    const turnElement = target.closest(".turn");
    const record = turnElement && turns.get(Number(turnElement.dataset.turn));
    if (!record) return;

    const cite = target.closest(".cite, .source-pill");
    if (cite) return openSource(record.id, cite.dataset.source);

    const question = target.closest(".q");
    if (question && record.quiz) {
      const option = target.closest(".option");
      const grade = target.closest("[data-quiz-grade]");
      const action = target.closest("[data-action]")?.dataset.action;
      if (option) return answerQuiz(turnElement, record, question, { kind: "option", option: Number(option.dataset.option) });
      if (action === "check") return answerQuiz(turnElement, record, question, { kind: "check" });
      if (action === "show-answer") return answerQuiz(turnElement, record, question, { kind: "show" });
      if (grade) return answerQuiz(turnElement, record, question, { kind: "grade", grade: grade.dataset.quizGrade });
      if (action === "detail") return ask(record.quiz.questions[Number(question.dataset.index)].question, { mode: "explain" });
      return;
    }

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
    if (action === "detail") return ask(target.closest(".practice").dataset.question, { mode: "explain" });
    if (action === "dismiss") {
      target.closest(".why").hidden = true;
      return;
    }

    const grade = target.closest("[data-grade]");
    if (grade) return gradePractice(grade);

    const strategy = target.closest("[data-strategy]")?.dataset.strategy;
    if (strategy) ask(record.question, { mode: strategy, previous: record.answer, shown: STRATEGIES[strategy][1] });
  });
}

setup();
