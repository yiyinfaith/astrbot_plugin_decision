/**
 * 人际网络管理页（由原独立人际网络插件逐行移植而来）。
 *
 * 路由前缀约定：
 *   AstrBot 的插件页桥（window.AstrBotPluginPage）会自动把当前插件名拼到请求路径前面，
 *   即 bridge.apiGet("network/personas") 实际请求
 *   `/astrbot_plugin_decision/network/personas`。
 *   本插件的后端把人际网络的全部 WebAPI 注册在 `/astrbot_plugin_decision/network/<route>`
 *   之下，而桥已经补上了 `astrbot_plugin_decision`，因此这里**每一条** route 都必须
 *   以 `network/` 开头（含 apiGet / apiPost / upload / download 四类调用），且只能写一次，
 *   不得重复加前缀。
 */
const bridge = window.AstrBotPluginPage;

const uiText = {
  nothingSelected: "未选择项目", statusActive: "进行中", statusUncertain: "不确定", statusEnded: "已结束",
  edit: "编辑", delete: "删除", strength: "关系亲密度", description: "关系描述", lifeEvents: "共同经历",
  noDetails: "暂无详细资料", root: "人格", bio: "身份简介", personality: "性格", preferences: "偏好",
  facts: "重要事实", identities: "平台身份", nicknames: "常用昵称", relationships: "相关关系", notes: "管理员备注",
  uploadAvatar: "上传头像", noPeople: "当前人格还没有人物", missing: "人格已缺失", relationCount: "条关系",
  personCount: "个人物", saved: "已保存", deleted: "已删除", merged: "人物已合并", imported: "导入完成",
  exported: "导出已开始", confirmDeletePerson: "删除该人物及其全部关系和经历？", confirmDeleteRelation: "删除该关系？",
  confirmDeleteEvent: "删除该条人生经历？", interactionActivity: "互动状态", recentInteraction: "最近互动",
  confirmMerge: "重复人物会被删除，关系和身份将迁移到保留人物。继续？", invalidMerge: "请选择两个不同的人物",
  previewImport: "导入预检", confirmImport: "确认合并导入？不会删除本地数据。", loadFailed: "加载失败",
  actionFailed: "操作失败",
};

const state = {
  personas: [],
  persona: null,
  network: { network: {}, characters: [], identities: [], relationships: [], life_events: [] },
  selected: null,
  cy: null,
  editingAliases: [],
  connection: { sourceId: null, pointerId: null, targetId: null },
  generation: { draft: null, expected: "", expectedCount: 6, allowFillExisting: false, busy: false, jobId: null },
};

let generationPollToken = 0;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const t = (key) => uiText[key] || key;
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const splitList = (value) => value.split(/[，,\n]/).map((item) => item.trim()).filter(Boolean);
const byId = (id) => state.network.characters.find((item) => item.id === id);
const aliasNames = (character) => (character.alias_usages || []).map((item) => item.alias);
const eventParticipants = (item) => (item.participant_ids || []).map((id) => byId(id)?.name || "?").join("、");
const eventTimeInput = (value) => {
  const date = value ? new Date(value) : new Date();
  const offset = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
};
const eventRows = (events) => events.length
  ? events.map((item) => `<div class="event-row"><button type="button" data-event-id="${esc(item.id)}"><strong>${esc(item.event_type)}</strong><div>${esc(item.summary)}</div><div class="relation-meta">${esc(new Date(item.occurred_at).toLocaleString("zh-CN"))} · ${esc(eventParticipants(item))}</div></button></div>`).join("")
  : `<p>${esc(t("noDetails"))}</p>`;
const interactionSummary = (stats) => stats?.last_interaction_at
  ? `<p>${esc(stats.activity)} · ${esc(new Date(stats.last_interaction_at).toLocaleDateString("zh-CN"))}</p><div class="relation-meta">近 7 / 30 / 90 天：${stats.count_7d} / ${stats.count_30d} / ${stats.count_90d} 次</div>`
  : `<p>暂无互动记录</p>`;

function toast(message, error = false) {
  const item = document.createElement("div");
  item.className = `toast${error ? " error" : ""}`;
  item.textContent = message;
  $("#toast-region").append(item);
  window.setTimeout(() => item.remove(), 3500);
}

function confirmAction(message, confirmLabel = "确认") {
  const dialog = $("#confirm-dialog");
  const confirmButton = $("#confirm-submit");
  $("#confirm-message").textContent = message;
  confirmButton.textContent = confirmLabel;
  return new Promise((resolve) => {
    let settled = false;
    const finish = (confirmed) => {
      if (settled) return;
      settled = true;
      confirmButton.onclick = null;
      dialog.removeEventListener("close", cancel);
      if (dialog.open) dialog.close();
      resolve(confirmed);
    };
    const cancel = () => finish(false);
    confirmButton.onclick = () => finish(true);
    dialog.addEventListener("close", cancel, { once: true });
    dialog.showModal();
  });
}

function avatarMarkup(character, size = "medium") {
  const initial = esc((character.name || "?").trim().slice(0, 1).toUpperCase());
  const image = character.avatar_data ? `<img src="${esc(character.avatar_data)}" alt="" />` : initial;
  return `<div class="avatar ${size}">${image}</div>`;
}

function relationshipLabel(item) {
  return `${byId(item.source_id)?.name || "?"} → ${byId(item.target_id)?.name || "?"}`;
}

async function loadPersonas(preserve = true) {
  const previous = preserve ? state.persona?.persona_id : null;
  const result = await bridge.apiGet("network/personas");
  state.personas = result.personas || [];
  state.persona = state.personas.find((item) => item.persona_id === previous)
    || state.personas.find((item) => !item.persona_missing)
    || state.personas[0]
    || null;
  const select = $("#persona-select");
  select.innerHTML = state.personas.map((item) => `<option value="${esc(item.persona_id)}">${esc(item.name)}${item.persona_missing ? ` · ${t("missing")}` : ""}</option>`).join("");
  if (state.persona) select.value = state.persona.persona_id;
}

async function loadNetwork() {
  if (!state.persona) return;
  state.network = await bridge.apiGet("network/network", { persona_id: state.persona.persona_id });
  state.selected = null;
  $("#network-enabled").checked = Boolean(state.network.network.enabled);
  renderSummary();
  renderGraph();
  renderPeople();
  renderInspector();
}

function renderSummary() {
  const people = state.network.characters.filter((item) => !item.is_persona).length;
  const relations = state.network.relationships.length;
  const events = state.network.life_events.length;
  $("#network-summary").textContent = `${people} ${t("personCount")} · ${relations} ${t("relationCount")} · ${events} 条经历`;
}

function graphElements() {
  const nodes = state.network.characters.map((item) => ({
    data: {
      id: item.id,
      label: item.name,
      avatar: item.avatar_data || "",
      root: item.is_persona ? "yes" : "no",
    },
    classes: [item.is_persona ? "persona-root" : "", item.avatar_data ? "has-avatar" : ""].filter(Boolean).join(" "),
  }));
  const edges = state.network.relationships.map((item) => ({
    data: { id: item.id, source: item.source_id, target: item.target_id, label: item.relation_type, strength: item.strength },
    classes: `status-${item.status}`,
  }));
  return [...nodes, ...edges];
}

function resetConnectionDrag() {
  state.connection.pointerId = null;
  state.connection.targetId = null;
  state.cy?.nodes().removeClass("connection-target");
  $("#connection-overlay").classList.remove("active");
  updateConnectionHandle();
}

function updateConnectionHandle() {
  const handle = $("#connection-handle");
  if (!state.cy) {
    handle.classList.add("hidden");
    return;
  }
  if (state.connection.pointerId !== null) {
    handle.classList.remove("hidden");
    handle.classList.add("dragging");
    return;
  }
  handle.classList.remove("dragging");
  const sourceId = state.selected?.type === "character" ? state.selected.id : null;
  const node = sourceId ? state.cy.$id(sourceId) : null;
  if (!node || !node.length || node.hasClass("filtered")) {
    state.connection.sourceId = null;
    handle.classList.add("hidden");
    return;
  }
  state.connection.sourceId = sourceId;
  const position = node.renderedPosition();
  handle.style.left = `${position.x + node.renderedWidth() / 2}px`;
  handle.style.top = `${position.y}px`;
  handle.classList.remove("hidden");
}

function connectionTargetAt(clientX, clientY) {
  const rect = $("#graph-canvas").getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let matched = null;
  state.cy.nodes().forEach((node) => {
    if (matched || node.id() === state.connection.sourceId || node.hasClass("filtered")) return;
    const box = node.renderedBoundingBox({ includeLabels: false, includeOverlays: false });
    if (x >= box.x1 - 8 && x <= box.x2 + 8 && y >= box.y1 - 8 && y <= box.y2 + 8) matched = node;
  });
  return { node: matched, x, y };
}

function startConnectionDrag(event) {
  if (!state.connection.sourceId || !state.cy) return;
  event.preventDefault();
  event.stopPropagation();
  state.connection.pointerId = event.pointerId;
  const source = state.cy.$id(state.connection.sourceId);
  const position = source.renderedPosition();
  const line = $("#connection-line");
  line.setAttribute("x1", position.x);
  line.setAttribute("y1", position.y);
  line.setAttribute("x2", position.x);
  line.setAttribute("y2", position.y);
  $("#connection-overlay").classList.add("active");
  $("#connection-handle").classList.add("dragging");
  $("#connection-handle").setPointerCapture?.(event.pointerId);
}

function moveConnectionDrag(event) {
  if (state.connection.pointerId !== event.pointerId) return;
  event.preventDefault();
  const target = connectionTargetAt(event.clientX, event.clientY);
  $("#connection-line").setAttribute("x2", target.x);
  $("#connection-line").setAttribute("y2", target.y);
  state.cy.nodes().removeClass("connection-target");
  state.connection.targetId = target.node?.id() || null;
  target.node?.addClass("connection-target");
}

function finishConnectionDrag(event) {
  if (state.connection.pointerId !== event.pointerId) return;
  const sourceId = state.connection.sourceId;
  const targetId = state.connection.targetId;
  resetConnectionDrag();
  if (sourceId && targetId && sourceId !== targetId) {
    openRelationshipDialog(null, { sourceId, targetId });
  }
}

function bindConnectionHandle() {
  const handle = $("#connection-handle");
  handle.onpointerdown = startConnectionDrag;
  handle.onpointermove = moveConnectionDrag;
  handle.onpointerup = finishConnectionDrag;
  handle.onpointercancel = resetConnectionDrag;
}

function renderGraph() {
  const styles = getComputedStyle(document.documentElement);
  if (state.cy) state.cy.destroy();
  state.cy = window.cytoscape({
    container: $("#network-graph"),
    elements: graphElements(),
    minZoom: 0.25,
    maxZoom: 2.5,
    style: [
      { selector: "node", style: {
        width: 62, height: 62, label: "data(label)", "font-size": 11, color: styles.getPropertyValue("--text").trim(),
        "text-valign": "bottom", "text-margin-y": 9, "text-wrap": "ellipsis", "text-max-width": 92,
        "background-color": styles.getPropertyValue("--accent-soft").trim(), "border-width": 2,
        "border-color": styles.getPropertyValue("--accent").trim(),
      } },
      { selector: "node.has-avatar", style: { "background-image": "data(avatar)", "background-fit": "cover" } },
      { selector: "node.persona-root", style: { width: 76, height: 76, "border-width": 4, "background-color": styles.getPropertyValue("--amber").trim(), "border-color": styles.getPropertyValue("--amber").trim() } },
      { selector: "node:selected", style: { "border-width": 5, "border-color": styles.getPropertyValue("--coral").trim() } },
      { selector: "node.connection-target", style: { "border-width": 7, "border-color": styles.getPropertyValue("--coral").trim(), "overlay-opacity": 0.12, "overlay-color": styles.getPropertyValue("--coral").trim(), "overlay-padding": 9 } },
      { selector: "edge", style: {
        width: "mapData(strength, 0, 100, 1, 5)", "curve-style": "bezier", "target-arrow-shape": "triangle",
        "line-color": styles.getPropertyValue("--accent").trim(), "target-arrow-color": styles.getPropertyValue("--accent").trim(),
        label: "data(label)", "font-size": 9, color: styles.getPropertyValue("--muted").trim(), "text-background-color": styles.getPropertyValue("--surface").trim(),
        "text-background-opacity": 0.9, "text-background-padding": 3, "text-rotation": "autorotate",
      } },
      { selector: "edge.status-ended", style: { "line-style": "dashed", opacity: 0.45 } },
      { selector: "edge.status-uncertain", style: { "line-style": "dotted", "line-color": styles.getPropertyValue("--amber").trim(), "target-arrow-color": styles.getPropertyValue("--amber").trim() } },
      { selector: "edge:selected", style: { width: 5, "line-color": styles.getPropertyValue("--coral").trim(), "target-arrow-color": styles.getPropertyValue("--coral").trim() } },
      { selector: ".filtered", style: { opacity: 0.08, "text-opacity": 0 } },
    ],
    layout: { name: "cose", animate: false, padding: 54, nodeRepulsion: 8500, idealEdgeLength: 130, gravity: 0.35 },
  });
  state.cy.on("tap", "node, edge", (event) => {
    state.selected = { type: event.target.isNode() ? "character" : "relationship", id: event.target.id() };
    renderInspector();
    updateConnectionHandle();
  });
  state.cy.on("tap", (event) => {
    if (event.target === state.cy) { state.selected = null; renderInspector(); updateConnectionHandle(); }
  });
  state.cy.on("pan zoom resize render position", updateConnectionHandle);
  applyGraphFilters();
  updateConnectionHandle();
}

function applyGraphFilters() {
  if (!state.cy) return;
  const query = $("#graph-search").value.trim().toLowerCase();
  const status = $("#status-filter").value;
  state.cy.elements().removeClass("filtered");
  if (status) {
    state.network.relationships.filter((item) => item.status !== status).forEach((item) => state.cy.$id(item.id).addClass("filtered"));
  }
  if (query) {
    state.network.characters.forEach((item) => {
      const haystack = [item.name, ...aliasNames(item), item.bio].join(" ").toLowerCase();
      if (!haystack.includes(query)) state.cy.$id(item.id).addClass("filtered");
    });
    state.network.relationships.forEach((item) => {
      const haystack = `${item.relation_type} ${item.description} ${relationshipLabel(item)}`.toLowerCase();
      if (!haystack.includes(query)) state.cy.$id(item.id).addClass("filtered");
    });
  }
  updateConnectionHandle();
}

function renderInspector() {
  const panel = $("#inspector");
  if (!state.selected) {
    panel.className = "inspector empty";
    panel.innerHTML = `<div class="empty-state"><div class="empty-glyph">⌁</div><h2>${esc(t("nothingSelected"))}</h2></div>`;
    return;
  }
  panel.className = "inspector";
  if (state.selected.type === "relationship") {
    const relation = state.network.relationships.find((item) => item.id === state.selected.id);
    if (!relation) { state.selected = null; renderInspector(); return; }
    const events = state.network.life_events.filter((item) => [relation.source_id, relation.target_id].every((id) => item.participant_ids.includes(id))).slice(0, 10);
    panel.innerHTML = `
      <div class="profile-heading"><span class="status-badge ${esc(relation.status)}">${esc(t(`status${relation.status[0].toUpperCase()}${relation.status.slice(1)}`))}</span><h2>${esc(relation.relation_type)}</h2><p class="alias-line">${esc(relationshipLabel(relation))}</p></div>
      <div class="inspector-actions"><button class="primary-button" id="inspect-edit-relation">${esc(t("edit"))}</button><button class="danger-button" id="inspect-delete-relation">${esc(t("delete"))}</button></div>
      <div class="detail-section"><h3>${esc(t("strength"))}</h3><p>${relation.strength}</p></div>
      <div class="detail-section"><h3>${esc(t("description"))}</h3><p>${esc(relation.description || t("noDetails"))}</p></div>
      <div class="detail-section"><h3>${esc(t("interactionActivity"))}</h3>${interactionSummary(relation.interaction_stats)}</div>
      <div class="detail-section"><h3>${esc(t("lifeEvents"))}</h3>${eventRows(events)}</div>`;
    $("#inspect-edit-relation").onclick = () => openRelationshipDialog(relation);
    $("#inspect-delete-relation").onclick = () => deleteRelationship(relation.id);
  } else {
    const character = byId(state.selected.id);
    if (!character) { state.selected = null; renderInspector(); return; }
    const relations = state.network.relationships.filter((item) => item.source_id === character.id || item.target_id === character.id);
    const identities = state.network.identities.filter((item) => item.character_id === character.id);
    const events = state.network.life_events.filter((item) => item.participant_ids.includes(character.id)).slice(0, 10);
    panel.innerHTML = `
      <div class="profile-heading">${avatarMarkup(character, "large")}<h2>${esc(character.name)}</h2><p class="alias-line">${character.alias_usages?.length ? character.alias_usages.map((item) => `${esc(item.alias)} ×${item.use_count}`).join(" · ") : (character.is_persona ? esc(t("root")) : "")}</p></div>
      <div class="inspector-actions"><button class="primary-button" id="inspect-edit-person">${esc(t("edit"))}</button><button class="secondary-button" id="inspect-avatar">${esc(t("uploadAvatar"))}</button>${character.is_persona ? "" : `<button class="danger-button" id="inspect-delete-person">${esc(t("delete"))}</button>`}</div>
      <div class="detail-section"><h3>${esc(t("bio"))}</h3><p>${esc(character.bio || t("noDetails"))}</p></div>
      <div class="detail-section"><h3>${esc(t("personality"))}</h3><p>${esc(character.personality || t("noDetails"))}</p></div>
      <div class="detail-section"><h3>${esc(t("preferences"))}</h3><div class="tag-list">${character.preferences.map((item) => `<span class="tag">${esc(item)}</span>`).join("") || esc(t("noDetails"))}</div></div>
      <div class="detail-section"><h3>${esc(t("facts"))}</h3><div class="tag-list">${character.facts.map((item) => `<span class="tag">${esc(item)}</span>`).join("") || esc(t("noDetails"))}</div></div>
      ${identities.length ? `<div class="detail-section"><h3>${esc(t("identities"))}</h3>${identities.map((item) => `<div class="identity-row"><strong>${esc(item.nicknames?.[0]?.nickname || item.user_id)}</strong><div class="relation-meta">${esc(item.platform)} · ${esc(item.user_id)}${item.session_id ? ` · ${esc(item.session_id)}` : ""}</div>${item.nicknames?.length ? `<div class="tag-list nickname-list">${item.nicknames.map((nickname) => `<span class="tag" title="${esc(t("nicknames"))}">${esc(nickname.nickname)} ×${nickname.use_count}</span>`).join("")}</div>` : ""}</div>`).join("")}</div>` : ""}
      <div class="detail-section"><h3>${esc(t("relationships"))}</h3>${relations.length ? relations.map((item) => `<div class="relation-row"><button data-relation-id="${esc(item.id)}"><strong>${esc(item.relation_type)}</strong><div class="relation-meta">${esc(relationshipLabel(item))} · 亲密度 ${item.strength} · ${esc(item.interaction_stats.activity)}</div></button></div>`).join("") : `<p>${esc(t("noDetails"))}</p>`}</div>
      <div class="detail-section"><h3>${esc(t("lifeEvents"))}</h3>${eventRows(events)}</div>
      ${character.notes ? `<div class="detail-section"><h3>${esc(t("notes"))}</h3><p>${esc(character.notes)}</p></div>` : ""}`;
    $("#inspect-edit-person").onclick = () => openCharacterDialog(character);
    $("#inspect-avatar").onclick = () => uploadAvatar(character.id);
    const deleteButton = $("#inspect-delete-person");
    if (deleteButton) deleteButton.onclick = () => deleteCharacter(character.id);
  }
  panel.querySelectorAll("[data-relation-id]").forEach((button) => {
    button.onclick = () => { state.selected = { type: "relationship", id: button.dataset.relationId }; renderInspector(); };
  });
  panel.querySelectorAll("[data-event-id]").forEach((button) => {
    button.onclick = () => openLifeEventDialog(state.network.life_events.find((item) => item.id === button.dataset.eventId));
  });
}

function renderPeople() {
  const query = $("#people-search").value.trim().toLowerCase();
  const people = state.network.characters.filter((item) => [item.name, ...aliasNames(item), item.bio].join(" ").toLowerCase().includes(query));
  $("#people-grid").innerHTML = people.length ? people.map((item) => {
    const count = state.network.relationships.filter((relation) => relation.source_id === item.id || relation.target_id === item.id).length;
    return `<article class="person-card" data-character-id="${esc(item.id)}"><div class="person-card-head">${avatarMarkup(item)}<div><h2>${esc(item.name)}</h2><div class="alias-line">${item.is_persona ? `<span class="root-badge">${esc(t("root"))}</span>` : esc(aliasNames(item).slice(0, 2).join(" · "))}</div></div></div><p class="bio">${esc(item.bio || item.personality || t("noDetails"))}</p><div class="person-card-footer"><span>${count} ${esc(t("relationCount"))}</span><button type="button">${esc(t("edit"))}</button></div></article>`;
  }).join("") : `<div class="empty-state"><div class="empty-glyph">○</div><h2>${esc(t("noPeople"))}</h2></div>`;
  $$(".person-card").forEach((card) => {
    card.onclick = () => openCharacterDialog(byId(card.dataset.characterId));
  });
}

function openCharacterDialog(character = null) {
  $("#character-form").reset();
  $("#character-id").value = character?.id || "";
  $("#character-name").value = character?.name || "";
  state.editingAliases = (character?.alias_usages || []).map((item) => ({ ...item }));
  renderAliasEditor();
  $("#character-bio").value = character?.bio || "";
  $("#character-personality").value = character?.personality || "";
  $("#character-preferences").value = character?.preferences?.join("\n") || "";
  $("#character-facts").value = character?.facts?.join("\n") || "";
  $("#character-notes").value = character?.notes || "";
  $("#delete-character-button").classList.toggle("hidden", !character || character.is_persona);
  $("#character-dialog").showModal();
}

function renderAliasEditor() {
  const editor = $("#character-aliases");
  editor.innerHTML = state.editingAliases.length
    ? state.editingAliases.map((item, index) => `<div class="alias-row" data-alias-index="${index}"><input class="alias-name" value="${esc(item.alias)}" maxlength="100" aria-label="人物别名" /><div class="alias-count"><input class="alias-use-count" type="number" min="1" max="999999" value="${Math.max(1, Number(item.use_count) || 1)}" aria-label="使用次数" /></div><button class="icon-button alias-delete" type="button" title="删除别名" aria-label="删除别名">×</button></div>`).join("")
    : '<div class="alias-empty">暂无别名</div>';
  $$(".alias-row").forEach((row) => {
    const index = Number(row.dataset.aliasIndex);
    row.querySelector(".alias-name").oninput = (event) => { state.editingAliases[index].alias = event.target.value; };
    row.querySelector(".alias-use-count").oninput = (event) => { state.editingAliases[index].use_count = Math.max(1, Number(event.target.value) || 1); };
    row.querySelector(".alias-delete").onclick = () => { state.editingAliases.splice(index, 1); renderAliasEditor(); };
  });
}

function fillCharacterSelects() {
  const options = state.network.characters.map((item) => `<option value="${esc(item.id)}">${esc(item.name)}</option>`).join("");
  ["#relationship-source", "#relationship-target", "#merge-target", "#merge-duplicate"].forEach((selector) => { $(selector).innerHTML = options; });
  $("#life-event-participants").innerHTML = options;
}

function openLifeEventDialog(item = null) {
  fillCharacterSelects();
  $("#life-event-form").reset();
  $("#life-event-id").value = item?.id || "";
  $("#life-event-time").value = eventTimeInput(item?.occurred_at);
  $("#life-event-type").value = item?.event_type || "";
  $("#life-event-summary").value = item?.summary || "";
  $("#life-event-importance").value = item?.importance ?? 50;
  $("#life-event-importance-output").value = item?.importance ?? 50;
  $("#life-event-tone").value = item?.emotional_tone || "";
  const selected = new Set(item?.participant_ids || [state.network.characters.find((character) => character.is_persona)?.id].filter(Boolean));
  [...$("#life-event-participants").options].forEach((option) => { option.selected = selected.has(option.value); });
  $("#delete-life-event-button").classList.toggle("hidden", !item);
  $("#life-event-dialog").showModal();
}

function openRelationshipDialog(relation = null, endpoints = {}) {
  fillCharacterSelects();
  $("#relationship-form").reset();
  $("#relationship-id").value = relation?.id || "";
  $("#relationship-source").value = relation?.source_id || endpoints.sourceId || state.network.characters.find((item) => item.is_persona)?.id || "";
  $("#relationship-target").value = relation?.target_id || endpoints.targetId || state.network.characters.find((item) => !item.is_persona)?.id || "";
  $("#relationship-type").value = relation?.relation_type || "";
  $("#relationship-status").value = relation?.status || "active";
  $("#relationship-strength").value = relation?.strength ?? 0;
  $("#strength-output").value = relation?.strength ?? 0;
  $("#relationship-description").value = relation?.description || "";
  $("#delete-relationship-button").classList.toggle("hidden", !relation);
  $("#relationship-dialog").showModal();
}

function setGenerationBusy(busy, label = "") {
  state.generation.busy = busy;
  $("#generation-submit").disabled = busy;
  $("#generation-validate").disabled = busy;
  $("#generation-apply").disabled = busy || !state.generation.draft;
  $("#generation-status").textContent = label;
}

function openGenerationDialog() {
  generationPollToken += 1;
  state.generation = { draft: null, expected: "", expectedCount: 6, allowFillExisting: false, busy: false, jobId: null };
  $("#generation-count").value = 6;
  $("#generation-density").value = "balanced";
  $("#generation-fill-existing").checked = false;
  $("#generation-hint").value = "";
  $("#generation-persona-name").textContent = state.persona?.name || "";
  $("#generation-result").classList.add("hidden");
  $("#generation-preview").classList.add("hidden");
  $("#generation-errors").classList.add("hidden");
  $("#generation-errors").textContent = "";
  $("#generation-raw").value = "";
  $("#generation-expected").textContent = "";
  $("#generation-raw-panel").open = false;
  setGenerationBusy(false);
  $("#generation-dialog").showModal();
}

function generationEndpointOptions() {
  const options = [];
  const root = state.network.characters.find((item) => item.is_persona);
  options.push({ value: "persona", label: `人格 · ${root?.name || state.persona?.name || "当前人格"}` });
  state.network.characters.filter((item) => !item.is_persona).forEach((item) => {
    options.push({ value: item.id, label: `现有 · ${item.name}` });
  });
  (state.generation.draft?.characters || []).forEach((item) => {
    if (item.ref) options.push({ value: item.ref, label: `新人物 · ${item.name}` });
  });
  return options;
}

function generationOptionsMarkup(selected) {
  return generationEndpointOptions().map((item) => `<option value="${esc(item.value)}"${item.value === selected ? " selected" : ""}>${esc(item.label)}</option>`).join("");
}

function updateGenerationRawFromDraft() {
  if (!state.generation.draft) return;
  const clean = {
    characters: state.generation.draft.characters.map(({ _included, ...item }) => item),
    relationships: state.generation.draft.relationships.map(({ _included, ...item }) => item),
  };
  $("#generation-raw").value = JSON.stringify(clean, null, 2);
}

function renderGenerationCharacters() {
  const characters = state.generation.draft?.characters || [];
  const included = characters.filter((item) => item._included !== false).length;
  $("#generation-character-count").textContent = `${included} / ${characters.length} 项`;
  $("#generation-characters").innerHTML = characters.map((item, index) => {
    const active = item._included !== false;
    const identifier = item.ref || item.id;
    return `<details class="generation-item generation-character${active ? "" : " excluded"}" data-generation-character="${index}">
      <summary><input class="generation-toggle" type="checkbox"${active ? " checked" : ""} aria-label="保留人物" /><span class="generation-item-title"><strong>${esc(item.name)}</strong><span>${item.id ? "补全现有人物" : esc(item.ref)}</span></span></summary>
      <div class="generation-item-body">
        <label><span>姓名</span><input data-generation-field="name" maxlength="100" value="${esc(item.name)}"${item.id ? " disabled" : ""} /></label>
        <label><span>别名</span><input data-generation-list="aliases" value="${esc((item.aliases || []).join("，"))}" /></label>
        <label class="full"><span>身份简介</span><textarea data-generation-field="bio" maxlength="4000">${esc(item.bio || "")}</textarea></label>
        <label class="full"><span>性格</span><textarea data-generation-field="personality" maxlength="4000">${esc(item.personality || "")}</textarea></label>
        <label><span>偏好</span><textarea data-generation-list="preferences">${esc((item.preferences || []).join("\n"))}</textarea></label>
        <label><span>重要事实</span><textarea data-generation-list="facts">${esc((item.facts || []).join("\n"))}</textarea></label>
      </div>
    </details>`;
  }).join("");
  $$("[data-generation-character]").forEach((element) => {
    const index = Number(element.dataset.generationCharacter);
    const toggle = element.querySelector(".generation-toggle");
    toggle.onclick = (event) => event.stopPropagation();
    toggle.onchange = (event) => {
      event.stopPropagation();
      const character = state.generation.draft.characters[index];
      character._included = event.target.checked;
      if (!character._included) {
        const endpoint = character.ref || character.id;
        state.generation.draft.relationships.forEach((relation) => {
          if (relation.source === endpoint || relation.target === endpoint) relation._included = false;
        });
      }
      renderGenerationPreview();
    };
    element.querySelectorAll("[data-generation-field]").forEach((input) => {
      input.oninput = () => {
        state.generation.draft.characters[index][input.dataset.generationField] = input.value;
        updateGenerationRawFromDraft();
      };
    });
    element.querySelectorAll("[data-generation-list]").forEach((input) => {
      input.oninput = () => {
        state.generation.draft.characters[index][input.dataset.generationList] = splitList(input.value);
        updateGenerationRawFromDraft();
      };
    });
  });
}

function renderGenerationRelationships() {
  const relationships = state.generation.draft?.relationships || [];
  const activeCharacters = new Set(
    (state.generation.draft?.characters || [])
      .filter((item) => item._included !== false)
      .map((item) => item.ref || item.id),
  );
  relationships.forEach((item) => {
    const sourceAvailable = item.source === "persona" || activeCharacters.has(item.source) || byId(item.source);
    const targetAvailable = item.target === "persona" || activeCharacters.has(item.target) || byId(item.target);
    if (!sourceAvailable || !targetAvailable) item._included = false;
  });
  const included = relationships.filter((item) => item._included !== false).length;
  $("#generation-relationship-count").textContent = `${included} / ${relationships.length} 项`;
  $("#generation-relationships").innerHTML = relationships.map((item, index) => {
    const active = item._included !== false;
    return `<details class="generation-item generation-relationship${active ? "" : " excluded"}" data-generation-relationship="${index}">
      <summary><input class="generation-toggle" type="checkbox"${active ? " checked" : ""} aria-label="保留关系" /><span class="generation-item-title"><strong>${esc(item.type)}</strong><span>${esc(item.source)} → ${esc(item.target)}</span></span></summary>
      <div class="generation-item-body">
        <label><span>关系主体</span><select data-generation-field="source">${generationOptionsMarkup(item.source)}</select></label>
        <label><span>关联人物</span><select data-generation-field="target">${generationOptionsMarkup(item.target)}</select></label>
        <label><span>关联人物的身份</span><input data-generation-field="type" maxlength="100" value="${esc(item.type)}" /></label>
        <label><span>状态</span><select data-generation-field="status"><option value="active"${item.status === "active" ? " selected" : ""}>进行中</option><option value="uncertain"${item.status === "uncertain" ? " selected" : ""}>不确定</option><option value="ended"${item.status === "ended" ? " selected" : ""}>已结束</option></select></label>
        <label class="full"><span>关系亲密度</span><div class="generation-strength"><input data-generation-number="strength" type="range" min="0" max="100" value="${Number(item.strength) || 0}" /><output>${Number(item.strength) || 0}</output></div></label>
        <label class="full"><span>关系描述</span><textarea data-generation-field="description" maxlength="2000">${esc(item.description || "")}</textarea></label>
      </div>
    </details>`;
  }).join("");
  $$("[data-generation-relationship]").forEach((element) => {
    const index = Number(element.dataset.generationRelationship);
    const toggle = element.querySelector(".generation-toggle");
    toggle.onclick = (event) => event.stopPropagation();
    toggle.onchange = (event) => {
      event.stopPropagation();
      state.generation.draft.relationships[index]._included = event.target.checked;
      renderGenerationRelationships();
      updateGenerationRawFromDraft();
    };
    element.querySelectorAll("[data-generation-field]").forEach((input) => {
      input.oninput = () => {
        state.generation.draft.relationships[index][input.dataset.generationField] = input.value;
        updateGenerationRawFromDraft();
      };
    });
    element.querySelectorAll("[data-generation-number]").forEach((input) => {
      input.oninput = () => {
        state.generation.draft.relationships[index][input.dataset.generationNumber] = Number(input.value);
        input.nextElementSibling.value = input.value;
        updateGenerationRawFromDraft();
      };
    });
  });
}

function renderGenerationPreview() {
  renderGenerationCharacters();
  renderGenerationRelationships();
  updateGenerationRawFromDraft();
  $("#generation-preview").classList.remove("hidden");
  $("#generation-apply").disabled = state.generation.busy || !state.generation.draft;
}

function showGenerationResult(result) {
  state.generation.expected = result.expected || "";
  $("#generation-result").classList.remove("hidden");
  $("#generation-raw").value = result.raw || "";
  $("#generation-expected").textContent = state.generation.expected;
  const errors = result.errors || [];
  $("#generation-errors").textContent = errors.join("\n");
  $("#generation-errors").classList.toggle("hidden", errors.length === 0);
  if (!result.valid || !result.draft) {
    state.generation.draft = null;
    $("#generation-preview").classList.add("hidden");
    $("#generation-raw-panel").open = true;
    $("#generation-apply").disabled = true;
    return;
  }
  state.generation.draft = {
    characters: result.draft.characters.map((item) => ({ ...item, _included: true })),
    relationships: result.draft.relationships.map((item) => ({ ...item, _included: true })),
  };
  $("#generation-raw-panel").open = false;
  renderGenerationPreview();
}

function selectedGenerationDraft() {
  const draft = state.generation.draft;
  if (!draft) return null;
  return {
    characters: draft.characters.filter((item) => item._included !== false).map(({ _included, ...item }) => item),
    relationships: draft.relationships.filter((item) => item._included !== false).map(({ _included, ...item }) => item),
  };
}

async function generateNetworkDraft() {
  if (state.generation.jobId) {
    const pollToken = ++generationPollToken;
    setGenerationBusy(true, "正在重新查询生成任务…");
    await pollGenerationJob(state.generation.jobId, pollToken);
    return;
  }
  const count = Math.max(1, Math.min(32, Number($("#generation-count").value) || 6));
  const generationHint = $("#generation-hint").value.trim();
  if ([...generationHint].length > 2000) {
    toast("额外生成要求不能超过 2000 字", true);
    $("#generation-status").textContent = "生成要求过长";
    return;
  }
  $("#generation-count").value = count;
  state.generation.expectedCount = count;
  state.generation.allowFillExisting = $("#generation-fill-existing").checked;
  const pollToken = ++generationPollToken;
  setGenerationBusy(true, "正在生成…");
  try {
    const job = await bridge.apiPost("network/generation/generate", {
      persona_id: state.persona.persona_id,
      count,
      density: $("#generation-density").value,
      allow_fill_existing: state.generation.allowFillExisting,
      generation_hint: generationHint,
    });
    if (pollToken !== generationPollToken) return;
    state.generation.jobId = job.job_id;
    state.generation.expectedCount = Number(job.expected_count) || count;
    state.generation.allowFillExisting = Boolean(job.allow_fill_existing);
    $("#generation-count").value = state.generation.expectedCount;
    if (job.density) $("#generation-density").value = job.density;
    $("#generation-fill-existing").checked = state.generation.allowFillExisting;
    if (job.reused) $("#generation-status").textContent = "继续等待此前生成任务…";
    await pollGenerationJob(job.job_id, pollToken);
  } catch (error) {
    if (pollToken !== generationPollToken) return;
    toast(`${t("actionFailed")}: ${error.message}`, true);
    setGenerationBusy(false, "生成失败");
  }
}

const waitForGenerationPoll = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function pollGenerationJob(jobId, pollToken) {
  let failures = 0;
  while (pollToken === generationPollToken && $("#generation-dialog").open) {
    try {
      const job = await bridge.apiPost("network/generation/status", {
        persona_id: state.persona.persona_id,
        job_id: jobId,
      });
      if (pollToken !== generationPollToken) return;
      failures = 0;
      state.generation.expectedCount = Number(job.expected_count) || state.generation.expectedCount;
      state.generation.allowFillExisting = Boolean(job.allow_fill_existing);
      $("#generation-count").value = state.generation.expectedCount;
      if (job.density) $("#generation-density").value = job.density;
      $("#generation-fill-existing").checked = state.generation.allowFillExisting;
      if (job.status === "pending" || job.status === "running") {
        $("#generation-status").textContent = "正在生成…";
        await waitForGenerationPoll(1500);
        continue;
      }
      const result = job.result || { valid: false, raw: "", errors: ["生成任务没有返回结果"], expected: "" };
      state.generation.jobId = null;
      showGenerationResult(result);
      const label = job.status === "failed" ? "生成失败" : (result.valid ? "草稿已生成" : "草稿需要修正");
      setGenerationBusy(false, label);
      return;
    } catch (error) {
      if (pollToken !== generationPollToken) return;
      failures += 1;
      if (failures >= 5) {
        toast(`${t("actionFailed")}: ${error.message}`, true);
        setGenerationBusy(false, "状态查询失败");
        return;
      }
      $("#generation-status").textContent = `连接中断，正在重试（${failures}/5）…`;
      await waitForGenerationPoll(1500 * failures);
    }
  }
}

async function validateGenerationRaw() {
  state.generation.allowFillExisting = $("#generation-fill-existing").checked;
  setGenerationBusy(true, "正在校验…");
  try {
    const result = await bridge.apiPost("network/generation/validate", {
      persona_id: state.persona.persona_id,
      raw: $("#generation-raw").value,
      expected_count: state.generation.expectedCount,
      allow_fill_existing: state.generation.allowFillExisting,
    });
    showGenerationResult(result);
    $("#generation-status").textContent = result.valid ? "校验通过" : "校验未通过";
  } catch (error) {
    toast(`${t("actionFailed")}: ${error.message}`, true);
  } finally {
    setGenerationBusy(false, $("#generation-status").textContent);
  }
}

async function applyGenerationDraft() {
  const draft = selectedGenerationDraft();
  if (!draft) return;
  setGenerationBusy(true, "正在应用…");
  try {
    const result = await bridge.apiPost("network/generation/apply", {
      persona_id: state.persona.persona_id,
      draft,
      allow_fill_existing: state.generation.allowFillExisting,
    });
    if (!result.applied) {
      showGenerationResult({
        valid: false,
        raw: JSON.stringify(draft, null, 2),
        errors: result.errors || ["应用失败"],
        expected: result.expected || state.generation.expected,
      });
      $("#generation-status").textContent = "应用前校验未通过";
      return;
    }
    $("#generation-dialog").close();
    toast("关系网草稿已应用");
    await loadNetwork();
  } catch (error) {
    toast(`${t("actionFailed")}: ${error.message}`, true);
  } finally {
    setGenerationBusy(false, $("#generation-status").textContent);
  }
}

async function deleteCharacter(id) {
  if (!await confirmAction(t("confirmDeletePerson"), "删除")) return;
  try {
    await bridge.apiPost("network/character/delete", { persona_id: state.persona.persona_id, character_id: id });
    $("#character-dialog").close(); state.selected = null; toast(t("deleted")); await loadNetwork();
  } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
}

async function deleteLifeEvent(id) {
  if (!await confirmAction(t("confirmDeleteEvent"), "删除")) return;
  try {
    await bridge.apiPost("network/life-event/delete", { persona_id: state.persona.persona_id, event_id: id });
    $("#life-event-dialog").close(); toast(t("deleted")); await loadNetwork();
  } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
}

async function deleteRelationship(id) {
  if (!await confirmAction(t("confirmDeleteRelation"), "删除")) return;
  try {
    await bridge.apiPost("network/relationship/delete", { persona_id: state.persona.persona_id, relationship_id: id });
    $("#relationship-dialog").close(); toast(t("deleted")); await loadNetwork();
  } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
}

function uploadAvatar(characterId) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/jpeg,image/png,image/webp";
  input.onchange = async () => {
    if (!input.files[0]) return;
    try {
      await bridge.upload(`network/avatar/${characterId}`, input.files[0]);
      toast(t("saved")); await loadNetwork();
    } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };
  input.click();
}

function bindEvents() {
  $$(".tab").forEach((tab) => {
    tab.onclick = () => {
      $$(".tab").forEach((item) => item.classList.toggle("active", item === tab));
      $$(".view").forEach((view) => view.classList.toggle("active", view.id === `${tab.dataset.view}-view`));
      if (tab.dataset.view === "graph") window.setTimeout(() => state.cy?.resize(), 0);
    };
  });
  $$(".close-dialog").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
  $("#persona-select").onchange = async (event) => { state.persona = state.personas.find((item) => item.persona_id === event.target.value); await loadNetwork(); };
  $("#network-enabled").onchange = async (event) => {
    try { await bridge.apiPost("network/network/enabled", { persona_id: state.persona.persona_id, enabled: event.target.checked }); toast(t("saved")); }
    catch (error) { event.target.checked = !event.target.checked; toast(`${t("actionFailed")}: ${error.message}`, true); }
  };
  $("#refresh-button").onclick = async () => { try { await loadPersonas(); await loadNetwork(); } catch (error) { toast(`${t("loadFailed")}: ${error.message}`, true); } };
  $("#fit-button").onclick = () => state.cy?.fit(undefined, 48);
  $("#graph-search").oninput = applyGraphFilters;
  $("#status-filter").onchange = applyGraphFilters;
  $("#people-search").oninput = renderPeople;
  $("#add-person-button").onclick = () => openCharacterDialog();
  $("#generate-network-button").onclick = openGenerationDialog;
  $("#generation-submit").onclick = generateNetworkDraft;
  $("#generation-validate").onclick = validateGenerationRaw;
  $("#generation-apply").onclick = applyGenerationDraft;
  $("#generation-dialog").addEventListener("close", () => {
    generationPollToken += 1;
    $("#generation-hint").value = "";
  });
  $("#add-alias-button").onclick = () => { state.editingAliases.push({ alias: "", use_count: 1, last_used_at: "" }); renderAliasEditor(); $("#character-aliases .alias-row:last-child .alias-name")?.focus(); };
  $("#add-relation-button").onclick = () => openRelationshipDialog();
  $("#add-life-event-button").onclick = () => openLifeEventDialog();
  $("#merge-button").onclick = () => { fillCharacterSelects(); $("#merge-dialog").showModal(); };
  $("#relationship-strength").oninput = (event) => { $("#strength-output").value = event.target.value; };
  $("#life-event-importance").oninput = (event) => { $("#life-event-importance-output").value = event.target.value; };
  $("#delete-character-button").onclick = () => deleteCharacter($("#character-id").value);
  $("#delete-relationship-button").onclick = () => deleteRelationship($("#relationship-id").value);
  $("#delete-life-event-button").onclick = () => deleteLifeEvent($("#life-event-id").value);

  $("#character-form").onsubmit = async (event) => {
    event.preventDefault();
    const payload = {
      persona_id: state.persona.persona_id, id: $("#character-id").value || undefined,
      name: $("#character-name").value.trim(),
      alias_usages: state.editingAliases
        .map((item) => ({ alias: item.alias.trim(), use_count: Math.max(1, Number(item.use_count) || 1), last_used_at: item.last_used_at || "" }))
        .filter((item) => item.alias)
        .sort((left, right) => right.use_count - left.use_count || left.alias.localeCompare(right.alias, "zh-CN")),
      bio: $("#character-bio").value.trim(), personality: $("#character-personality").value.trim(),
      preferences: splitList($("#character-preferences").value), facts: splitList($("#character-facts").value),
      notes: $("#character-notes").value.trim(),
    };
    try { await bridge.apiPost("network/character/save", payload); $("#character-dialog").close(); toast(t("saved")); await loadNetwork(); }
    catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };

  $("#relationship-form").onsubmit = async (event) => {
    event.preventDefault();
    const source = $("#relationship-source").value;
    const target = $("#relationship-target").value;
    if (source === target) { toast(t("invalidMerge"), true); return; }
    const payload = {
      persona_id: state.persona.persona_id, id: $("#relationship-id").value || undefined,
      source, target, type: $("#relationship-type").value.trim(), strength: Number($("#relationship-strength").value),
      status: $("#relationship-status").value, description: $("#relationship-description").value.trim(),
    };
    try { await bridge.apiPost("network/relationship/save", payload); $("#relationship-dialog").close(); toast(t("saved")); await loadNetwork(); }
    catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };

  $("#life-event-form").onsubmit = async (event) => {
    event.preventDefault();
    const participants = [...$("#life-event-participants").selectedOptions].map((option) => option.value);
    if (new Set(participants).size < 2) { toast("人生经历至少需要两名参与人物", true); return; }
    const payload = {
      persona_id: state.persona.persona_id,
      id: $("#life-event-id").value || undefined,
      occurred_at: new Date($("#life-event-time").value).toISOString(),
      type: $("#life-event-type").value.trim(),
      participants,
      summary: $("#life-event-summary").value.trim(),
      importance: Number($("#life-event-importance").value),
      emotional_tone: $("#life-event-tone").value.trim(),
    };
    try { await bridge.apiPost("network/life-event/save", payload); $("#life-event-dialog").close(); toast(t("saved")); await loadNetwork(); }
    catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };

  $("#merge-form").onsubmit = async (event) => {
    event.preventDefault();
    const targetId = $("#merge-target").value;
    const duplicateId = $("#merge-duplicate").value;
    if (targetId === duplicateId) { toast(t("invalidMerge"), true); return; }
    if (!await confirmAction(t("confirmMerge"), "合并")) return;
    try {
      await bridge.apiPost("network/character/merge", { persona_id: state.persona.persona_id, target_id: targetId, duplicate_id: duplicateId });
      $("#merge-dialog").close(); toast(t("merged")); await loadNetwork();
    } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };

  $("#export-button").onclick = async () => {
    try { await bridge.download("network/export", { persona_id: state.persona.persona_id }); toast(t("exported")); }
    catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
  };
  $("#import-button").onclick = () => $("#import-file").click();
  $("#import-file").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const preview = await bridge.upload(`network/import/${state.persona.key}/preview`, file);
      if (!preview.can_apply) {
        toast(`${t("previewImport")}: ${preview.conflicts.join("; ")}`, true);
        return;
      }
      const details = `${t("previewImport")}\n${preview.new_characters} + ${preview.updated_characters} ↻ ${t("personCount")}\n${preview.new_relationships} + ${preview.updated_relationships} ↻ ${t("relationCount")}\n${preview.new_life_events} + ${preview.updated_life_events} ↻ 条经历\n\n${t("confirmImport")}`;
      if (await confirmAction(details, "导入")) {
        await bridge.apiPost("network/import/apply", { token: preview.token });
        toast(t("imported")); await loadNetwork();
      }
    } catch (error) { toast(`${t("actionFailed")}: ${error.message}`, true); }
    finally { event.target.value = ""; }
  };
}

async function start() {
  await bridge.ready();
  bindEvents();
  bindConnectionHandle();
  bridge.onContext(() => {
    renderSummary(); renderPeople(); renderInspector(); renderGraph();
  });
  try { await loadPersonas(false); await loadNetwork(); }
  catch (error) { console.error(error); toast(`${t("loadFailed")}: ${error.message}`, true); }
}

start();
