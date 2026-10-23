"use strict";

const TIERS = ["S", "A", "B", "C", "D", "E", "F"];
// Tiers split at natural gaps in the ratings, so the names describe standing
// relative to the league rather than fixed thresholds.
const TIER_NAMES = {
  S: "Lombardi or Bust",
  A: "Legit Threats",
  B: "Nobody Wants This Matchup",
  C: "Coin-Flip Club",
  D: "Hanging Around",
  E: "Scouting College QBs",
  F: "On the Clock",
};
const $ = (sel, root = document) => root.querySelector(sel);

const state = { data: null, sort: "rank", asc: true, notes: true };

const fmt = {
  pts: (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1),
  pct: (v) => (v >= 0.995 ? ">99%" : v < 0.005 ? "<1%" : Math.round(v * 100) + "%"),
  record: (t) => `${t.wins}-${t.losses}${t.ties ? "-" + t.ties : ""}`,
};

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
}

function logo(team, cls = "logo") {
  const img = el("img", { class: cls, src: `logos/${team.team}.png`, alt: "", loading: "lazy" });
  img.addEventListener("error", () => {
    // Bundled copy first, then the original URL, then a lettered circle.
    if (!img.dataset.retried) {
      img.dataset.retried = "1";
      img.src = team.logo;
      return;
    }
    const fallback = el("span", { class: "logo-fallback", text: team.abbr });
    fallback.style.background = team.color;
    img.replaceWith(fallback);
  });
  return img;
}

function change(t) {
  const n = t.rank_change;
  if (n > 0) return el("span", { class: "up", text: "▲" + n, title: `Up ${n} from #${t.previous_rank}` });
  if (n < 0) return el("span", { class: "down", text: "▼" + -n, title: `Down ${-n} from #${t.previous_rank}` });
  return el("span", { class: "flat", text: "–" });
}

function rangeBar(t, lo, hi) {
  const span = hi - lo;
  const pos = (v) => ((v - lo) / span) * 100;
  const bar = el("div", { class: "range", title: `80% range ${fmt.pts(t.range[0])} to ${fmt.pts(t.range[1])}` });
  const fill = el("i");
  fill.style.left = pos(t.range[0]) + "%";
  fill.style.width = pos(t.range[1]) - pos(t.range[0]) + "%";
  const dot = el("b");
  dot.style.left = pos(t.rating) + "%";
  bar.append(fill, dot);
  return bar;
}

const sorters = {
  rank: (t) => t.rank,
  name: (t) => t.name,
  rating: (t) => -t.rating,
  record: (t) => -(t.wins + t.ties / 2) / Math.max(1, t.wins + t.losses + t.ties),
  sor: (t) => t.strength_of_record_rank,
  playoffs: (t) => -t.odds.make_playoffs,
  sb: (t) => -t.odds.win_super_bowl,
  vegas: (t) => -(t.vegas_difference ?? 0),
};

function renderRankings() {
  const { teams } = state.data;
  const lo = Math.min(...teams.map((t) => t.range[0]));
  const hi = Math.max(...teams.map((t) => t.range[1]));
  const key = sorters[state.sort];
  const rows = [...teams].sort((a, b) => {
    const x = key(a), y = key(b);
    const c = x < y ? -1 : x > y ? 1 : a.rank - b.rank;
    return state.asc ? c : -c;
  });

  document.querySelectorAll("#rank-table th[data-sort]").forEach((th) => {
    const active = th.dataset.sort === state.sort;
    th.classList.toggle("sorted", active);
    th.classList.toggle("asc", active && !state.asc);
    th.setAttribute("aria-sort", active ? (state.asc ? "ascending" : "descending") : "none");
  });

  const body = $("#rank-table tbody");
  body.replaceChildren(
    ...rows.flatMap((t) => {
      const vd = t.vegas_difference;
      const tr = el("tr", { class: "row" + (state.notes && t.note ? " has-note" : ""), tabindex: "0" });
      tr.append(
        el("td", { class: "num", text: t.rank }),
        el("td", { class: "chg" }, change(t)),
        el("td", {}, el("div", { class: "team" }, logo(t), el("div", {}, t.name, el("small", {}, t.starter ?? "")))),
        el("td", { class: "num" }, el("span", { class: `tier-chip tier-${t.tier}`, text: t.tier, title: TIER_NAMES[t.tier] }), el("span", { class: "rating", text: fmt.pts(t.rating) })),
        el("td", { class: "range-col" }, rangeBar(t, lo, hi)),
        el("td", { class: "num", text: fmt.record(t) }),
        el("td", { class: "num hide-sm", text: t.strength_of_record_rank }),
        el("td", { class: "num hide-sm", text: fmt.pct(t.odds.make_playoffs) }),
        el("td", { class: "num hide-sm", text: fmt.pct(t.odds.win_super_bowl) }),
        el("td", { class: "num hide-md " + (vd > 1 ? "up" : vd < -1 ? "down" : "flat"), text: vd == null ? "–" : fmt.pts(vd) }),
      );
      const open = () => showDetail(t);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open()));
      if (!state.notes || !t.note) return [tr];
      const noteRow = el("tr", { class: "note-row" }, el("td", { colspan: "10", text: t.note }));
      noteRow.addEventListener("click", open);
      return [tr, noteRow];
    }),
  );
}

function renderTiers() {
  const { teams } = state.data;
  const list = $("#tier-list");
  list.replaceChildren(
    ...TIERS.map((tier) => {
      const members = teams.filter((t) => t.tier === tier).sort((a, b) => a.rank - b.rank);
      const label = el("div", { class: `tier-label tier-${tier}`, text: TIER_NAMES[tier] });
      const body = el("div", { class: "tier-body" });
      if (!members.length) body.append(el("span", { class: "empty", text: "No teams in this tier right now" }));
      for (const t of members) {
        const btn = el("button", { class: "logo-btn", title: `${t.rank}. ${t.name} (${fmt.pts(t.rating)})`, "aria-label": `${t.rank}. ${t.name}, rating ${fmt.pts(t.rating)}` });
        btn.append(logo(t, ""), el("span", { class: "abbr", text: t.abbr }));
        btn.addEventListener("click", () => showDetail(t));
        body.append(btn);
      }
      return el("div", { class: "tier-row" }, label, body);
    }),
  );
}

function renderGames() {
  const { games, next_week } = state.data;
  const byTeam = Object.fromEntries(state.data.teams.map((t) => [t.team, t]));
  const body = $("#games-table tbody");
  if (!games.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: "5", text: "No games scheduled." })));
    return;
  }
  const rows = [...games].sort((a, b) => Math.abs(b.model_line - (b.market_line ?? b.model_line)) - Math.abs(a.model_line - (a.market_line ?? a.model_line)));
  body.replaceChildren(
    ...rows.map((g) => {
      const gap = g.market_line == null ? null : g.model_line - g.market_line;
      const away = byTeam[g.away], home = byTeam[g.home];
      return el(
        "tr",
        {},
        el("td", {}, el("div", { class: "team" }, logo(away), `${away.abbr} @ ${home.abbr}`, logo(home))),
        el("td", { class: "num", text: fmt.pts(g.model_line) }),
        el("td", { class: "num", text: g.market_line == null ? "–" : fmt.pts(g.market_line) }),
        el("td", { class: "num " + (gap == null ? "flat" : Math.abs(gap) >= 3 ? "up" : "flat"), text: gap == null ? "–" : fmt.pts(gap) }),
        el("td", { class: "num hide-sm", text: fmt.pct(g.home_win_probability) }),
      );
    }),
  );
  $("#view-games .note").prepend(`Week ${next_week}, biggest model–Vegas disagreements first. `);
}

const BAR_LABELS = [
  ["offense", "Offense"], ["defense", "Defense"], ["special_teams", "Special teams"],
  ["scoring", "Scoring margin"], ["quarterback", "This week's QB"], ["injuries", "Injuries"],
];

function showDetail(t) {
  const d = $("#detail");
  d.style.setProperty("--team", t.color);
  const max = Math.max(4, ...BAR_LABELS.map(([k]) => Math.abs(t.breakdown[k])));
  const bars = BAR_LABELS.map(([k, name]) => {
    const v = t.breakdown[k];
    const fill = el("div", { class: "fill " + (v >= 0 ? "pos" : "neg") });
    fill.style.width = (Math.abs(v) / max) * 50 + "%";
    return el("div", { class: "bar" }, el("span", { text: name }), el("div", { class: "track" }, fill), el("span", { class: "val", text: fmt.pts(v) }));
  });

  const game = (g) =>
    g ? `${g.opponent_name} ${g.score}, week ${g.week}${g.opponent_starter ? ` (vs ${g.opponent_starter})` : ""}` : "–";
  const off = t.offseason;
  const why = [off.qb_changed && (off.veteran_qb ? "new veteran QB" : "new QB"), off.coach_changed && "new head coach"].filter(Boolean).join(", ");
  const kv = (rows) => el("div", { class: "kv" }, ...rows.map(([a, b]) => el("div", {}, el("span", { text: a }), el("span", { text: b }))));

  d.replaceChildren(
    el("div", { class: "d-head" },
      logo(t),
      el("div", {}, el("h2", { text: `#${t.rank} ${t.name}` }), el("p", { text: `${t.division} · ${fmt.record(t)} · ${t.starter ? "QB " + t.starter : ""}` })),
      el("button", { class: "d-close", "aria-label": "Close", text: "×", onclick: () => d.close() }),
    ),
    el("div", { class: "d-body" },
      t.note ? el("p", { class: "d-note", text: t.note }) : null,
      el("div", { class: "stats" },
        stat(fmt.pts(t.rating), "rating this week"),
        stat(fmt.pts(t.healthy_rating), "healthy, main QB"),
        stat(`${fmt.pts(t.range[0])} to ${fmt.pts(t.range[1])}`, "80% range"),
        stat(fmt.pct(t.chance_best), "chance truly #1"),
      ),
      el("div", {}, el("h3", { text: "Where the rating comes from" }), el("div", { class: "bars" }, ...bars)),
      el("div", {}, el("h3", { text: "Résumé (not part of the rank)" }),
        kv([
          ["Strength of record", `#${t.strength_of_record_rank} (${fmt.pct(t.strength_of_record)} of a playoff team would match it)`],
          ["Point differential", (t.point_diff > 0 ? "+" : "") + t.point_diff],
          ["Best win", game(t.best_win)],
          ["Worst loss", game(t.worst_loss)],
          ["Schedule left", t.remaining_schedule == null ? "–" : `${fmt.pts(t.remaining_schedule)} avg opponent (#${t.remaining_schedule_rank} hardest)`],
        ])),
      el("div", {}, el("h3", { text: "Rest of season" }),
        kv([
          ["Projected wins", t.odds.projected_wins.toFixed(1)],
          ["Make playoffs", fmt.pct(t.odds.make_playoffs)],
          ["Win division", fmt.pct(t.odds.win_division)],
          ["1 seed (bye)", fmt.pct(t.odds.top_seed)],
          ["Win conference", fmt.pct(t.odds.win_conference)],
          ["Win Super Bowl", fmt.pct(t.odds.win_super_bowl)],
        ])),
      el("div", {}, el("h3", { text: "Context" }),
        kv([
          ["Preseason rating", `${fmt.pts(t.preseason_rating)}${why ? " (" + why + ")" : ""}`],
          ["Change since last week", `${fmt.pts(t.rating_change)} points`],
          ["Vegas rating", t.vegas_rating == null ? "–" : `${fmt.pts(t.vegas_rating)} (model ${t.vegas_difference > 0 ? "higher" : "lower"} by ${Math.abs(t.vegas_difference).toFixed(1)})`],
        ])),
    ),
  );
  d.showModal();
}

function stat(value, label) {
  return el("div", { class: "stat" }, el("b", { text: value }), el("span", { text: label }));
}

function setView(view) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  document.querySelectorAll(".view").forEach((s) => (s.hidden = s.id !== "view-" + view));
  try { localStorage.setItem("view", view); } catch (e) { /* storage may be blocked */ }
  if (location.hash !== "#" + view) history.replaceState(null, "", "#" + view);
}

function init(data) {
  state.data = data;
  const when = new Date(data.generated).toLocaleDateString(undefined, { month: "long", day: "numeric" });
  $("#subtitle").textContent = `${data.season} season through week ${data.through_week} · updated ${when} · ratings are points vs. an average team, ±${data.rating_sd.toFixed(1)}`;
  renderRankings();
  renderTiers();
  renderGames();

  document.querySelectorAll("#rank-table th[data-sort]").forEach((th) => {
    const sort = () => {
      state.asc = state.sort === th.dataset.sort ? !state.asc : true;
      state.sort = th.dataset.sort;
      renderRankings();
    };
    th.tabIndex = 0;
    th.addEventListener("click", sort);
    th.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), sort()));
  });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  try { state.notes = localStorage.getItem("notes") !== "off"; } catch (e) { /* ignore */ }
  const toggle = $("#show-notes");
  toggle.checked = state.notes;
  toggle.addEventListener("change", () => {
    state.notes = toggle.checked;
    try { localStorage.setItem("notes", state.notes ? "on" : "off"); } catch (e) { /* ignore */ }
    renderRankings();
  });
  renderRankings();
  $("#detail").addEventListener("click", (e) => e.target === e.currentTarget && e.currentTarget.close());

  let saved = null;
  try { saved = localStorage.getItem("view"); } catch (e) { /* ignore */ }
  const wanted = location.hash.slice(1) || saved;
  setView(["rankings", "tiers", "games"].includes(wanted) ? wanted : "rankings");
}

fetch("rankings.json", { cache: "no-cache" })
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
  .then(init)
  .catch((err) => {
    $("#subtitle").textContent = "";
    const box = $("#error");
    box.hidden = false;
    box.textContent = `Couldn't load rankings.json (${err.message}). If you opened this file directly, serve the folder instead: python3 -m http.server -d site`;
  });
