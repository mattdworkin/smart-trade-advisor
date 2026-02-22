const roleSelect = document.getElementById("roleSelect");
const refreshBtn = document.getElementById("refreshBtn");
const queryForm = document.getElementById("queryForm");
const questionInput = document.getElementById("questionInput");
const agentAnswer = document.getElementById("agentAnswer");
const actionList = document.getElementById("actionList");

const dashboardTargets = {
  winners: document.getElementById("winnersList"),
  losers: document.getElementById("losersList"),
  longTerm: document.getElementById("longTermList"),
  watchlist: document.getElementById("watchlistList"),
  nyt: document.getElementById("nytList"),
};

const kpiTargets = {
  lastRefresh: document.getElementById("lastRefresh"),
  topWinner: document.getElementById("topWinner"),
  topLoser: document.getElementById("topLoser"),
  topLongTerm: document.getElementById("topLongTerm"),
};

const refreshSeconds = Number(document.body.dataset.refreshSeconds || 900);
const userId = getUserId();

function getUserId() {
  const existing = localStorage.getItem("trade-agent-user-id");
  if (existing) {
    return existing;
  }
  const generated = `web-${Math.random().toString(36).slice(2, 10)}`;
  localStorage.setItem("trade-agent-user-id", generated);
  return generated;
}

function formatPct(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function renderCards(container, cards, key) {
  container.innerHTML = "";
  if (!cards || cards.length === 0) {
    container.textContent = "No signals available.";
    return;
  }

  cards.forEach((card) => {
    const move = key === "long" ? card.expected_return_30d : card.expected_return_1d;
    const tone = move >= 0 ? "up" : "down";

    const el = document.createElement("article");
    el.className = "signal-card";
    el.innerHTML = `
      <div class="signal-head">
        <strong>${card.symbol}</strong>
        <span class="${tone}">${formatPct(move)}</span>
      </div>
      <p class="muted">Confidence ${(card.confidence * 100).toFixed(1)}%</p>
      <p class="muted">${(card.rationale || []).slice(0, 1).join(" ")}</p>
    `;
    container.appendChild(el);
  });
}

function renderNews(items) {
  dashboardTargets.nyt.innerHTML = "";
  if (!items || items.length === 0) {
    dashboardTargets.nyt.textContent = "No NYT briefing available.";
    return;
  }
  items.forEach((item) => {
    const el = document.createElement("article");
    el.className = "headline-item";
    const when = new Date(item.published_at).toLocaleString();
    el.innerHTML = `
      <a href="${item.url}" target="_blank" rel="noreferrer">${item.title}</a>
      <p>${item.summary}</p>
      <p class="muted">${when}</p>
    `;
    dashboardTargets.nyt.appendChild(el);
  });
}

function renderKpis(data) {
  const winner = data.winners?.[0];
  const loser = data.losers?.[0];
  const longTerm = data.long_term?.[0];
  kpiTargets.lastRefresh.textContent = new Date(data.generated_at).toLocaleString();
  kpiTargets.topWinner.textContent = winner ? `${winner.symbol} ${formatPct(winner.expected_return_1d)}` : "-";
  kpiTargets.topLoser.textContent = loser ? `${loser.symbol} ${formatPct(loser.expected_return_1d)}` : "-";
  kpiTargets.topLongTerm.textContent = longTerm
    ? `${longTerm.symbol} ${formatPct(longTerm.expected_return_30d)}`
    : "-";
}

async function loadDashboard() {
  try {
    const role = roleSelect.value;
    const response = await fetch(`/api/dashboard?role=${encodeURIComponent(role)}`, {
      headers: {
        "x-user-id": userId,
        "x-user-role": role,
      },
    });
    if (!response.ok) {
      throw new Error(`Dashboard error ${response.status}`);
    }
    const data = await response.json();
    renderKpis(data);
    renderCards(dashboardTargets.winners, data.winners, "short");
    renderCards(dashboardTargets.losers, data.losers, "short");
    renderCards(dashboardTargets.longTerm, data.long_term, "long");
    renderCards(dashboardTargets.watchlist, data.watchlist, "short");
    renderNews(data.nyt_briefing);
  } catch (err) {
    agentAnswer.textContent = `Dashboard load failed: ${err.message}`;
  }
}

async function refreshManually() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Refreshing...";
  try {
    await fetch("/api/refresh", { method: "POST" });
    await loadDashboard();
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh Now";
  }
}

async function runAgentQuery(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }
  agentAnswer.textContent = "Running query...";
  actionList.innerHTML = "";

  const role = roleSelect.value;
  const response = await fetch("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-user-id": userId,
      "x-user-role": role,
    },
    body: JSON.stringify({ question, role }),
  });
  if (!response.ok) {
    agentAnswer.textContent = `Query failed: ${response.status}`;
    return;
  }
  const payload = await response.json();
  agentAnswer.textContent = payload.answer;

  (payload.actions || []).forEach((action) => {
    const li = document.createElement("li");
    li.textContent = `${action.action_type}: ${action.reason}`;
    actionList.appendChild(li);
  });
}

refreshBtn.addEventListener("click", refreshManually);
roleSelect.addEventListener("change", loadDashboard);
queryForm.addEventListener("submit", runAgentQuery);

loadDashboard();
setInterval(loadDashboard, Math.max(60, refreshSeconds) * 1000);
