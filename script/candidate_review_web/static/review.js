const state = {
  candidate: null,
  navigation: null,
  view: "preview",
  busy: false,
  imageReady: false,
  filtersLoaded: false,
  renderToken: 0,
};

const labelNames = {
  upright_cone: "直立锥桶",
  fallen_cone: "倒伏锥桶",
  not_cone: "非锥桶",
  ignore: "忽略（不进入训练）",
};

const elements = {
  image: document.getElementById("candidateImage"),
  empty: document.getElementById("emptyState"),
  reviewed: document.getElementById("reviewedCount"),
  total: document.getElementById("totalCount"),
  progress: document.getElementById("progressBar"),
  reviewFilter: document.getElementById("reviewFilter"),
  poolFilter: document.getElementById("poolFilter"),
  splitFilter: document.getElementById("splitFilter"),
  undo: document.getElementById("undoButton"),
  previous: document.getElementById("previousButton"),
  next: document.getElementById("nextButton"),
  position: document.getElementById("candidatePosition"),
  poolName: document.getElementById("poolName"),
  sourceSplit: document.getElementById("sourceSplit"),
  confidence: document.getElementById("confidence"),
  angle: document.getElementById("angle"),
  sourceImage: document.getElementById("sourceImage"),
  reviewStatus: document.getElementById("reviewStatus"),
  uprightTotal: document.getElementById("uprightTotal"),
  fallenTotal: document.getElementById("fallenTotal"),
  notConeTotal: document.getElementById("notConeTotal"),
  ignoreTotal: document.getElementById("ignoreTotal"),
  toast: document.getElementById("toast"),
  decisions: Array.from(document.querySelectorAll(".decision")),
  labelTotals: Array.from(document.querySelectorAll(".label-total")),
  views: Array.from(document.querySelectorAll(".view-option")),
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("visible"), 1800);
}

function setBusy(value) {
  state.busy = value;
  const hasCandidate = Boolean(state.candidate);
  elements.decisions.forEach((button) => {
    button.disabled = value || !hasCandidate || !state.imageReady;
  });
  elements.undo.disabled = value;
  elements.previous.disabled = value || !state.navigation?.previous_candidate_id;
  elements.next.disabled = value || !state.navigation?.next_candidate_id;
  elements.views.forEach((button) => { button.disabled = value || !hasCandidate; });
  elements.labelTotals.forEach((button) => { button.disabled = value; });
  [elements.reviewFilter, elements.poolFilter, elements.splitFilter].forEach((filter) => {
    filter.disabled = value;
  });
}

function updateStats(stats) {
  elements.reviewed.textContent = stats.reviewed;
  elements.total.textContent = stats.total;
  const progress = stats.total > 0 ? (stats.reviewed / stats.total) * 100 : 100;
  elements.progress.style.width = `${progress}%`;
  elements.uprightTotal.textContent = stats.label_counts.upright_cone;
  elements.fallenTotal.textContent = stats.label_counts.fallen_cone;
  elements.notConeTotal.textContent = stats.label_counts.not_cone;
  elements.ignoreTotal.textContent = stats.label_counts.ignore;
}

function updateNavigation(navigation) {
  state.navigation = navigation;
  const position = navigation?.position;
  elements.position.textContent = position ? `${position} / ${navigation.total}` : `- / ${navigation?.total ?? 0}`;
}

function updateReviewFilterState() {
  elements.labelTotals.forEach((button) => {
    button.classList.toggle("active", button.dataset.review === elements.reviewFilter.value);
  });
}

function updateReviewStatus(label) {
  elements.reviewStatus.className = "review-status";
  if (label) {
    elements.reviewStatus.textContent = `已标注为：${labelNames[label]}`;
    elements.reviewStatus.classList.add(label);
  } else {
    elements.reviewStatus.textContent = "未标注";
  }
  elements.decisions.forEach((button) => {
    const selected = button.dataset.label === label;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function preloadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = resolve;
    image.onerror = () => reject(new Error("候选图片加载失败"));
    image.src = url;
  });
}

async function renderCandidate(candidate) {
  const token = ++state.renderToken;
  state.candidate = candidate;
  state.imageReady = false;
  const hasCandidate = Boolean(candidate);
  elements.image.hidden = !hasCandidate;
  elements.empty.hidden = hasCandidate;
  if (!hasCandidate) {
    elements.poolName.textContent = "-";
    elements.sourceSplit.textContent = "-";
    elements.confidence.textContent = "-";
    elements.angle.textContent = "-";
    elements.sourceImage.textContent = "-";
    updateReviewStatus(null);
    elements.image.removeAttribute("src");
    setBusy(state.busy);
    return;
  }
  elements.poolName.textContent = candidate.review_pool_name;
  elements.sourceSplit.textContent = candidate.source_split;
  elements.confidence.textContent = candidate.yolo_confidence
    ? Number(candidate.yolo_confidence).toFixed(3)
    : "GT";
  elements.angle.textContent = candidate.long_axis_angle
    ? `${Number(candidate.long_axis_angle).toFixed(1)}°`
    : "-";
  elements.sourceImage.textContent = candidate.source_image;
  updateReviewStatus(candidate.review_label);
  setBusy(state.busy);
  const imageUrl = state.view === "preview" ? candidate.preview_url : candidate.crop_url;
  await preloadImage(imageUrl);
  if (token !== state.renderToken) return;
  elements.image.src = imageUrl;
  if (typeof elements.image.decode === "function") {
    await elements.image.decode();
  }
  if (token !== state.renderToken) return;
  state.imageReady = true;
  setBusy(state.busy);
}

function populateFilters(payload) {
  if (state.filtersLoaded) return;
  payload.pools.forEach((pool) => {
    const option = document.createElement("option");
    option.value = pool.value;
    option.textContent = pool.name;
    elements.poolFilter.appendChild(option);
  });
  payload.splits.forEach((split) => {
    const option = document.createElement("option");
    option.value = split;
    option.textContent = split;
    elements.splitFilter.appendChild(option);
  });
  state.filtersLoaded = true;
}

function filterParams() {
  return new URLSearchParams({
    review: elements.reviewFilter.value,
    pool: elements.poolFilter.value,
    split: elements.splitFilter.value,
  });
}

async function fetchPayload(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function applyPayload(payload) {
  populateFilters(payload);
  updateStats(payload.stats);
  updateNavigation(payload.navigation);
  await renderCandidate(payload.candidate);
}

async function loadCurrent() {
  setBusy(true);
  try {
    await applyPayload(await fetchPayload(`/api/current?${filterParams()}`));
  } catch (error) {
    showToast(`加载失败: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function loadCandidate(candidateId) {
  if (state.busy || !candidateId) return;
  setBusy(true);
  try {
    await applyPayload(await fetchPayload(`/api/candidate/${candidateId}?${filterParams()}`));
  } catch (error) {
    showToast(`加载失败: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function submitLabel(label) {
  if (state.busy || !state.candidate || !state.imageReady) return;
  const candidateId = state.candidate.candidate_id;
  const nextCandidateId = state.navigation?.next_candidate_id;
  setBusy(true);
  try {
    const response = await fetch("/api/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: candidateId, label }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    if (nextCandidateId) {
      await applyPayload(
        await fetchPayload(`/api/candidate/${nextCandidateId}?${filterParams()}`),
      );
    } else {
      await applyPayload(await fetchPayload(`/api/current?${filterParams()}`));
    }
  } catch (error) {
    showToast(`保存失败: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function undoLast() {
  if (state.busy) return;
  setBusy(true);
  try {
    const response = await fetch("/api/undo", { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!payload.result) {
      showToast("没有可撤销的标注");
    }
    await applyPayload(await fetchPayload(`/api/current?${filterParams()}`));
  } catch (error) {
    showToast(`撤销失败: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

elements.decisions.forEach((button) => {
  button.addEventListener("click", () => submitLabel(button.dataset.label));
});

elements.views.forEach((button) => {
  button.addEventListener("click", async () => {
    if (state.busy || !state.candidate || button.dataset.view === state.view) return;
    setBusy(true);
    state.view = button.dataset.view;
    elements.views.forEach((item) => item.classList.toggle("active", item === button));
    try {
      await renderCandidate(state.candidate);
    } catch (error) {
      showToast(`加载失败: ${error.message}`);
    } finally {
      setBusy(false);
    }
  });
});

[elements.reviewFilter, elements.poolFilter, elements.splitFilter].forEach((filter) => {
  filter.addEventListener("change", () => {
    updateReviewFilterState();
    loadCurrent();
  });
});

elements.labelTotals.forEach((button) => {
  button.addEventListener("click", () => {
    if (state.busy) return;
    elements.reviewFilter.value = button.dataset.review;
    updateReviewFilterState();
    loadCurrent();
  });
});

elements.undo.addEventListener("click", undoLast);
elements.previous.addEventListener("click", () => loadCandidate(state.navigation?.previous_candidate_id));
elements.next.addEventListener("click", () => loadCandidate(state.navigation?.next_candidate_id));

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLSelectElement) return;
  const labels = {
    "1": "upright_cone",
    "2": "fallen_cone",
    "3": "not_cone",
    "4": "ignore",
  };
  if (labels[event.key]) submitLabel(labels[event.key]);
  if (event.key === "ArrowLeft") loadCandidate(state.navigation?.previous_candidate_id);
  if (event.key === "ArrowRight") loadCandidate(state.navigation?.next_candidate_id);
  if (event.key.toLowerCase() === "z") undoLast();
});

updateReviewFilterState();
loadCurrent();
