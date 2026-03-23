const state = {
  catalog: null,
  blocks: [],
  advancedConfig: null,
  selectedBlockId: null,
  inspectorPanels: {},
  validation: null,
  runs: [],
  activeRunId: null,
  activeRun: null,
  transientRunningBlockId: null,
  terminalMessage: "No run started yet.",
  terminalMessageTimer: null,
  terminal: { cwd: "", history: [] },
  pollingHandle: null,
  draggedTaskSection: null,
  draggedBlockId: null,
};

const elements = {
  workbench: document.getElementById("workbench"),
  mainStack: document.getElementById("main-stack"),
  workspacePill: document.getElementById("workspace-pill"),
  runtimePill: document.getElementById("runtime-pill"),
  environmentSummary: document.getElementById("environment-summary"),
  taskPalette: document.getElementById("task-palette"),
  templateList: document.getElementById("template-list"),
  configFileInput: document.getElementById("config-file-input"),
  configDropzone: document.getElementById("config-dropzone"),
  loadConfigButton: document.getElementById("load-config-button"),
  saveButton: document.getElementById("save-button"),
  inspectorSelection: document.getElementById("inspector-selection"),
  blockInspector: document.getElementById("block-inspector"),
  advancedConfig: document.getElementById("advanced-config"),
  clearButton: document.getElementById("clear-button"),
  workspaceSummary: document.getElementById("workspace-summary"),
  workspaceMessage: document.getElementById("workspace-message"),
  pipelineCanvas: document.getElementById("pipeline-canvas"),
  runButton: document.getElementById("run-button"),
  runPill: document.getElementById("run-pill"),
  runList: document.getElementById("run-list"),
  saveModal: document.getElementById("save-modal"),
  saveModalClose: document.getElementById("save-modal-close"),
  saveCancelButton: document.getElementById("save-cancel-button"),
  saveConfirmButton: document.getElementById("save-confirm-button"),
  saveFilename: document.getElementById("save-filename"),
  saveOutputDir: document.getElementById("save-output-dir"),
  saveSectionList: document.getElementById("save-section-list"),
  saveModalCount: document.getElementById("save-modal-count"),
  saveModalMessage: document.getElementById("save-modal-message"),
  resizerLeftMain: document.getElementById("resizer-left-main"),
  resizerMainRight: document.getElementById("resizer-main-right"),
  themeToggle: document.getElementById("theme-toggle"),
};

const LAYOUT_STORAGE = {
  left: "workbench.leftColPx",
  right: "workbench.rightColPx",
};

const THEME_STORAGE = "workbench.theme";

function initTheme() {
  const saved = localStorage.getItem(THEME_STORAGE) || "dark";
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  if (elements.themeToggle) {
    elements.themeToggle.textContent = theme === "light" ? "☽" : "☀";
    elements.themeToggle.title = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  }
  localStorage.setItem(THEME_STORAGE, theme);
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
}

const TASK_COMMANDS = {
  preprocessing: "preprocess",
  slide_encoding: "encode_slides",
  patient_encoding: "encode_patients",
  training: "train",
  crossval: "crossval",
  deployment: "deploy",
  statistics: "statistics",
  heatmaps: "heatmaps",
};

const TASK_META = {
  preprocessing:     { icon: "▦",  color: "#5ecad4" },
  slide_encoding:    { icon: "⬡",  color: "#7b8dff" },
  patient_encoding:  { icon: "◎",  color: "#a78bfa" },
  training:          { icon: "⚡", color: "#f4a84a" },
  crossval:          { icon: "↺",  color: "#52c99a" },
  deployment:        { icon: "⬆",  color: "#4db6f0" },
  statistics:        { icon: "∑",  color: "#f47da5" },
  heatmaps:          { icon: "◫",  color: "#f4826e" },
};

const TASK_PANEL_DEFS = [
  {
    id: "paths",
    title: "Files & Paths",
    defaultOpen: true,
    matcher: (field) => field.kind === "path" || field.name.endsWith("_dir") || field.name.endsWith("_path"),
  },
  {
    id: "targets",
    title: "Labels & Targets",
    defaultOpen: true,
    matcher: (field) => [
      "task",
      "ground_truth_label",
      "categories",
      "status_label",
      "time_label",
      "patient_label",
      "filename_label",
    ].includes(field.name),
  },
  {
    id: "method",
    title: "Method",
    defaultOpen: true,
    matcher: (field) => [
      "extractor",
      "encoder",
      "generate_hash",
      "use_vary_precision_transform",
    ].includes(field.name),
  },
  {
    id: "runtime",
    title: "Runtime & Sampling",
    defaultOpen: false,
    matcher: (field) => [
      "device",
      "cache_dir",
      "max_workers",
      "tile_size_um",
      "tile_size_px",
      "default_slide_mpp",
    ].includes(field.name),
  },
  {
    id: "other",
    title: "Other Settings",
    defaultOpen: false,
    matcher: () => true,
  },
];

const ADVANCED_PANEL_DEFS = [
  {
    id: "training_setup",
    title: "Training Setup",
    defaultOpen: true,
    matcher: () => true,
  },
];

function makeId() {
  return window.crypto?.randomUUID?.() || `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    const error = new Error(payload?.error || payload?.errors?.join("\n") || response.statusText);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }

  return payload;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function defaultParamsFor(section) {
  const task = state.catalog.tasks[section];
  const params = {};
  for (const field of task.fields) {
    if (field.default !== null && field.default !== undefined) {
      params[field.name] = deepClone(field.default);
    } else if (field.kind === "boolean") {
      params[field.name] = false;
    } else if (field.kind === "list") {
      params[field.name] = [];
    } else {
      params[field.name] = "";
    }
  }
  return params;
}

function makeBlock(section) {
  return {
    id: makeId(),
    section,
    enabled: true,
    ui: {
      multiTarget: false,
      showRuntime: false,
      autoCreateSlideTable: false,
    },
    params: defaultParamsFor(section),
  };
}

function pipelinePayload() {
  return {
    blocks: state.blocks.map((block) => ({
      id: block.id,
      section: block.section,
      enabled: block.enabled,
      params: block.params,
      ui: block.ui,
    })),
    advanced_config: state.advancedConfig,
  };
}

function insertBlock(section, index = state.blocks.length) {
  const block = makeBlock(section);
  const safeIndex = Math.max(0, Math.min(index, state.blocks.length));
  state.blocks.splice(safeIndex, 0, block);
  state.selectedBlockId = block.id;
  renderAll();
  return true;
}

function selectedBlock() {
  return state.blocks.find((block) => block.id === state.selectedBlockId) || null;
}

function statusClass(status) {
  if (["completed"].includes(status)) return "good";
  if (["queued", "running", "terminating", "stopping"].includes(status)) return "info";
  if (["warning", "stopped"].includes(status)) return "warn";
  if (["failed", "terminated"].includes(status)) return "bad";
  return "info";
}

function runPillClass(status) {
  if (status === "completed") return "pill-good";
  if (["running", "queued", "terminating", "stopping"].includes(status)) return "pill-run";
  if (status === "stopped") return "pill-warn";
  if (["failed", "terminated"].includes(status)) return "pill-bad";
  return "pill-idle";
}

function friendlyValue(field, value) {
  if (field.kind === "boolean") {
    return value ? "yes" : "no";
  }
  if (field.kind === "list") {
    if (Array.isArray(value)) {
      return value.join(", ");
    }
    return value || "unset";
  }
  if (value === undefined || value === null || value === "") {
    return "unset";
  }
  return String(value);
}

function panelStateKey(scope, panelId) {
  return `${scope}:${panelId}`;
}

function isPanelOpen(scope, panelId, defaultOpen = false) {
  const key = panelStateKey(scope, panelId);
  if (Object.hasOwn(state.inspectorPanels, key)) {
    return state.inspectorPanels[key];
  }
  return defaultOpen;
}

function setPanelOpen(scope, panelId, nextOpen) {
  state.inspectorPanels[panelStateKey(scope, panelId)] = nextOpen;
}

function groupFields(fields, definitions) {
  const grouped = definitions.map((definition) => ({ ...definition, fields: [] }));
  for (const field of fields) {
    const bucket = grouped.find((definition) => definition.matcher(field));
    if (bucket) {
      bucket.fields.push(field);
    }
  }
  return grouped.filter((group) => group.fields.length > 0);
}

function readStoredPx(key) {
  try {
    const raw = window.localStorage.getItem(key);
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  } catch (_error) {
    return null;
  }
}

function writeStoredPx(key, value) {
  try {
    window.localStorage.setItem(key, String(Math.round(value)));
  } catch (_error) {
    // Ignore storage failures.
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function applyStoredLayout() {
  const left = readStoredPx(LAYOUT_STORAGE.left);
  const right = readStoredPx(LAYOUT_STORAGE.right);
  if (left) {
    elements.workbench.style.setProperty("--left-col", `${left}px`);
  }
  if (right) {
    elements.workbench.style.setProperty("--right-col", `${right}px`);
  }
}

function initResizer(handle, side) {
  handle.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 1120px)").matches) {
      return;
    }

    event.preventDefault();
    const startX = event.clientX;
    const styles = getComputedStyle(elements.workbench);
    const startLeft = parseFloat(styles.getPropertyValue("--left-col")) || 360;
    const startRight = parseFloat(styles.getPropertyValue("--right-col")) || 420;
    const total = elements.workbench.getBoundingClientRect().width;
    const centerMin = 540;
    const resizerWidth = 24;

    handle.classList.add("dragging");
    document.body.classList.add("is-resizing");

    const onMove = (moveEvent) => {
      const delta = moveEvent.clientX - startX;
      if (side === "left") {
        const maxLeft = total - startRight - centerMin - resizerWidth;
        const next = clamp(startLeft + delta, 280, maxLeft);
        elements.workbench.style.setProperty("--left-col", `${next}px`);
      } else {
        const maxRight = total - startLeft - centerMin - resizerWidth;
        const next = clamp(startRight - delta, 320, maxRight);
        elements.workbench.style.setProperty("--right-col", `${next}px`);
      }
    };

    const onUp = () => {
      const computed = getComputedStyle(elements.workbench);
      const currentLeft = parseFloat(computed.getPropertyValue("--left-col")) || startLeft;
      const currentRight = parseFloat(computed.getPropertyValue("--right-col")) || startRight;
      writeStoredPx(LAYOUT_STORAGE.left, currentLeft);
      writeStoredPx(LAYOUT_STORAGE.right, currentRight);
      handle.classList.remove("dragging");
      document.body.classList.remove("is-resizing");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
  });
}

function createMetaPill(label, tone = "info") {
  const pill = document.createElement("span");
  pill.className = `meta-pill ${tone}`;
  pill.textContent = label;
  return pill;
}

function parseListInput(raw, presentation = "csv") {
  const text = String(raw || "");
  if (presentation === "lines") {
    return text.split("\n").map((item) => item.trim()).filter(Boolean);
  }
  return text.split(",").map((item) => item.trim()).filter(Boolean);
}

function clearTerminalMessageTimer() {
  if (!state.terminalMessageTimer) {
    return;
  }
  window.clearTimeout(state.terminalMessageTimer);
  state.terminalMessageTimer = null;
}

function terminalMessageMeta(message) {
  const text = String(message || "");
  if (/failed|error/i.test(text)) {
    return { tone: "error", dismissMs: 0 };
  }
  if (/warning/i.test(text)) {
    return { tone: "warning", dismissMs: 0 };
  }
  if (/loaded|saved|ready to start|started\.|created\.|cleared/i.test(text)) {
    return { tone: "success", dismissMs: 3200 };
  }
  return { tone: "info", dismissMs: 0 };
}

function setTerminalMessage(message) {
  clearTerminalMessageTimer();
  state.terminalMessage = message;
  renderWorkspaceMessage();

  const { dismissMs } = terminalMessageMeta(message);
  if (message && dismissMs > 0) {
    state.terminalMessageTimer = window.setTimeout(() => {
      state.terminalMessage = "No run selected.";
      state.terminalMessageTimer = null;
      renderWorkspaceMessage();
    }, dismissMs);
  }
}

function getPathSuggestions(fieldName, excludeBlockId) {
  const seen = new Set();
  const suggestions = [];
  for (const block of state.blocks) {
    if (block.id === excludeBlockId) continue;
    const val = String(block.params?.[fieldName] ?? "").trim();
    if (val && !seen.has(val)) {
      seen.add(val);
      suggestions.push(val);
    }
  }
  return suggestions;
}

function createFieldEditor({ field, value, onChange, disabled = false, helpOverride = null, blockId = null }) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  if (field.kind === "list" || field.help || field.kind === "path") {
    wrapper.classList.add("full-span");
  }

  const label = document.createElement("label");
  label.textContent = field.label + (field.required ? " *" : "");
  wrapper.append(label);

  let input;
  if (field.kind === "select") {
    input = document.createElement("select");
    const current = value ?? "";
    for (const optionValue of field.options || []) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue === "" ? "Auto" : optionValue;
      if (String(optionValue) === String(current)) {
        option.selected = true;
      }
      input.append(option);
    }
    input.addEventListener("change", () => onChange(input.value));
  } else if (field.kind === "boolean") {
    input = document.createElement("select");
    const current = Boolean(value);
    for (const optionValue of [true, false]) {
      const option = document.createElement("option");
      option.value = optionValue ? "true" : "false";
      option.textContent = optionValue ? "true" : "false";
      if (optionValue === current) {
        option.selected = true;
      }
      input.append(option);
    }
    input.addEventListener("change", () => onChange(input.value === "true"));
  } else if (field.kind === "list") {
    input = document.createElement("textarea");
    input.placeholder = field.placeholder || (field.presentation === "lines" ? "one value per line" : "comma,separated");
    input.value = Array.isArray(value)
      ? value.join(field.presentation === "lines" ? "\n" : ",")
      : value || "";
    input.addEventListener("input", () => onChange(parseListInput(input.value, field.presentation)));
  } else {
    input = document.createElement("input");
    input.type = field.kind === "integer" || field.kind === "number" ? "number" : "text";
    input.placeholder = field.placeholder || "";
    input.value = value ?? "";
    if (field.kind === "number") {
      input.step = "any";
    }
    input.addEventListener("input", () => {
      if (field.kind === "integer") {
        onChange(input.value === "" ? "" : Number.parseInt(input.value, 10));
        return;
      }
      if (field.kind === "number") {
        onChange(input.value === "" ? "" : Number(input.value));
        return;
      }
      onChange(input.value);
    });
  }

  input.disabled = disabled;
  wrapper.append(input);

  if (helpOverride || field.help) {
    const help = document.createElement("div");
    help.className = "field-help";
    help.textContent = helpOverride || field.help;
    wrapper.append(help);
  }

  if (field.kind === "path") {
    const suggestions = getPathSuggestions(field.name, blockId);

    if (suggestions.length > 0) {
      const dropdown = document.createElement("ul");
      dropdown.className = "path-suggestions";
      dropdown.setAttribute("role", "listbox");

      function buildItems(filter) {
        dropdown.innerHTML = "";
        const filtered = filter
          ? suggestions.filter((s) => s.toLowerCase().includes(filter.toLowerCase()))
          : suggestions;
        for (const s of filtered) {
          const li = document.createElement("li");
          li.className = "path-suggestion-item";
          li.setAttribute("role", "option");
          li.textContent = s;
          li.addEventListener("mousedown", (e) => {
            e.preventDefault();
            input.value = s;
            onChange(s);
            dropdown.classList.remove("open");
          });
          dropdown.append(li);
        }
        dropdown.classList.toggle("open", filtered.length > 0);
      }

      input.addEventListener("focus", () => buildItems(input.value));
      input.addEventListener("input", () => buildItems(input.value));
      input.addEventListener("blur", () => {
        // slight delay so mousedown on item fires first
        setTimeout(() => dropdown.classList.remove("open"), 120);
      });
      input.addEventListener("keydown", (e) => {
        const items = [...dropdown.querySelectorAll(".path-suggestion-item")];
        const active = dropdown.querySelector(".path-suggestion-item.active");
        const idx = items.indexOf(active);
        if (e.key === "ArrowDown") {
          e.preventDefault();
          const next = items[(idx + 1) % items.length];
          active?.classList.remove("active");
          next?.classList.add("active");
          next?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          const prev = items[(idx - 1 + items.length) % items.length];
          active?.classList.remove("active");
          prev?.classList.add("active");
          prev?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter" && active) {
          e.preventDefault();
          input.value = active.textContent;
          onChange(active.textContent);
          dropdown.classList.remove("open");
        } else if (e.key === "Escape") {
          dropdown.classList.remove("open");
        }
      });

      wrapper.style.position = "relative";
      wrapper.append(dropdown);
    }

    const meta = document.createElement("div");
    meta.className = "field-meta";
    meta.textContent = suggestions.length
      ? `${suggestions.length} suggestion${suggestions.length > 1 ? "s" : ""} from other cells.`
      : "Enter an absolute path for this field.";
    wrapper.append(meta);
  }

  return wrapper;
}

function blockFieldNames(block) {
  return new Set(state.catalog.tasks[block.section].fields.map((field) => field.name));
}

function supportsAutoSlideTable(block) {
  const names = blockFieldNames(block);
  return names.has("slide_table") && names.has("clini_table") && (names.has("feature_dir") || names.has("feat_dir"));
}

function slideTableFeatureDir(block) {
  return block.params.feature_dir || block.params.feat_dir || "";
}

function slideTableAutoCreateMissing(block) {
  const missing = [];
  if (!String(block.params.clini_table || "").trim()) {
    missing.push("Clinical Table");
  }
  if (!String(slideTableFeatureDir(block) || "").trim()) {
    missing.push("Feature Directory");
  }
  if (!String(block.params.patient_label || "").trim()) {
    missing.push("Patient Column");
  }
  if (!String(block.params.filename_label || "").trim()) {
    missing.push("Filename Column");
  }
  return missing;
}

function createSlideTableEditor(field, block) {
  const wrapper = document.createElement("div");
  wrapper.className = "field full-span slide-table-field";

  const label = document.createElement("label");
  label.textContent = field.label + (field.required ? " *" : "");
  wrapper.append(label);

  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = field.placeholder || "";
  input.value = block.params[field.name] ?? "";
  input.disabled = Boolean(block.ui?.autoCreateSlideTable);
  input.addEventListener("input", () => {
    block.params[field.name] = input.value;
    renderPipeline();
  });
  wrapper.append(input);

  const meta = document.createElement("div");
  meta.className = "field-meta";
  meta.textContent = block.ui?.autoCreateSlideTable
    ? "This path will be generated automatically right before STAMP runs."
    : "Enter an absolute path for this field.";
  wrapper.append(meta);

  if (!supportsAutoSlideTable(block)) {
    return wrapper;
  }

  const toggle = document.createElement("label");
  toggle.className = "field-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = Boolean(block.ui?.autoCreateSlideTable);
  checkbox.addEventListener("change", () => {
    block.ui = block.ui || {};
    block.ui.autoCreateSlideTable = checkbox.checked;
    renderInspector();
    renderPipeline();
  });
  const toggleText = document.createElement("span");
  toggleText.textContent = "Auto create slide table";
  toggle.append(checkbox, toggleText);
  wrapper.append(toggle);

  if (!checkbox.checked) {
    return wrapper;
  }

  const helper = document.createElement("div");
  helper.className = "slide-table-helper";

  const copy = document.createElement("div");
  copy.className = "field-help";
  copy.textContent = "A temp slide_table.csv will be created automatically right before STAMP runs, using the current Clinical Table, Feature Directory, Patient Column, and Filename Column.";
  helper.append(copy);

  const missing = slideTableAutoCreateMissing(block);
  if (missing.length > 0) {
    const warning = document.createElement("div");
    warning.className = "field-help slide-table-warning";
    warning.textContent = `Missing: ${missing.join(", ")}`;
    helper.append(warning);
  }
  wrapper.append(helper);

  return wrapper;
}

function blockTask(block) {
  return block.params.task || "classification";
}

function fieldVisibleForBlock(field, block) {
  const task = blockTask(block);
  if (field.name === "status_label" || field.name === "time_label") {
    return task === "survival";
  }
  if (field.name === "categories") {
    return task === "classification";
  }
  if (field.name === "ground_truth_label") {
    return task !== "survival";
  }
  return true;
}

function isMultiTargetBlock(block) {
  return blockTask(block) === "classification" && Boolean(block.ui?.multiTarget);
}

function pipelineRequiresBarspoon() {
  return state.blocks.some((block) => (
    ["training", "crossval", "deployment", "statistics"].includes(block.section)
    && isMultiTargetBlock(block)
  ));
}

function createGroundTruthEditor(field, block) {
  const wrapper = document.createElement("div");
  wrapper.className = "field full-span multi-target-field";

  const label = document.createElement("label");
  label.textContent = field.label + (field.required ? " *" : "");
  wrapper.append(label);

  const task = blockTask(block);
  const isClassification = task === "classification";
  const multiTarget = isMultiTargetBlock(block);

  if (isClassification) {
    const toggle = document.createElement("label");
    toggle.className = "field-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = multiTarget;
    checkbox.addEventListener("change", () => {
      block.ui = block.ui || {};
      block.ui.multiTarget = checkbox.checked;
      if (checkbox.checked) {
        const values = Array.isArray(block.params.ground_truth_label)
          ? block.params.ground_truth_label
          : parseListInput(block.params.ground_truth_label || "", "csv");
        block.params.ground_truth_label = values;
      } else {
        const first = Array.isArray(block.params.ground_truth_label)
          ? (block.params.ground_truth_label[0] || "")
          : String(block.params.ground_truth_label || "");
        block.params.ground_truth_label = first;
      }
      if (pipelineRequiresBarspoon()) {
        state.advancedConfig.model_name = "barspoon";
      } else if (!state.advancedConfig.model_name) {
        state.advancedConfig.model_name = "vit";
      }
      renderInspector();
      renderAdvancedConfig();
      renderPipeline();
    });
    const toggleText = document.createElement("span");
    toggleText.textContent = "Multi-target";
    toggle.append(checkbox, toggleText);
    wrapper.append(toggle);
  }

  const input = isClassification && multiTarget
    ? document.createElement("textarea")
    : document.createElement("input");

  if (input.tagName === "TEXTAREA") {
    input.placeholder = "KRAS,BRAF,NRAS";
    input.value = Array.isArray(block.params.ground_truth_label)
      ? block.params.ground_truth_label.join(",")
      : block.params.ground_truth_label || "";
    input.addEventListener("input", () => {
      block.params.ground_truth_label = parseListInput(input.value, "csv");
      if (pipelineRequiresBarspoon()) {
        state.advancedConfig.model_name = "barspoon";
        renderAdvancedConfig();
      }
      renderPipeline();
    });
  } else {
    input.type = "text";
    input.placeholder = "KRAS";
    input.value = Array.isArray(block.params.ground_truth_label)
      ? (block.params.ground_truth_label[0] || "")
      : (block.params.ground_truth_label || "");
    input.addEventListener("input", () => {
      block.params.ground_truth_label = input.value;
      renderPipeline();
    });
  }

  wrapper.append(input);

  const help = document.createElement("div");
  help.className = "field-help";
  help.textContent = isClassification && multiTarget
    ? "Separate labels with commas. The pipeline will feed them as a list and lock Model Backbone to barspoon."
    : "Use one label for regression or single-target classification.";
  wrapper.append(help);

  return wrapper;
}

function fieldKindFromValue(value) {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return "text";
}

function createModelParamEditor(modelName, paramName, value) {
  const pseudoField = {
    name: paramName,
    label: paramName.replaceAll("_", " "),
    kind: fieldKindFromValue(value),
    help: "",
    required: false,
    placeholder: "",
  };
  const editor = createFieldEditor({
    field: pseudoField,
    value,
    onChange: (nextValue) => {
      state.advancedConfig.model_params[modelName][paramName] = nextValue;
    },
  });
  editor.classList.add("model-param-field");
  return editor;
}

function renderTopbar() {
  const env = state.catalog.environment;
  const activeCount = state.blocks.filter((block) => block.enabled).length;
  const cellText = state.blocks.length === 0 ? "No cells" : `${activeCount}/${state.blocks.length} active cells`;
  elements.workspacePill.className = `pill ${activeCount > 0 ? "pill-good" : "pill-idle"}`;
  elements.workspacePill.textContent = cellText;
  elements.runtimePill.textContent = `${env.devices.join(" / ")} | Python ${env.python}`;
  elements.environmentSummary.textContent = `${env.platform} | ${env.cwd}`;
}

function renderTaskPalette() {
  elements.taskPalette.innerHTML = "";
  for (const [section, task] of Object.entries(state.catalog.tasks)) {
    const meta = TASK_META[section] || { icon: "◆", color: "#97aabd" };
    const card = document.createElement("article");
    card.className = "palette-card";
    card.dataset.section = section;
    card.draggable = true;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Add ${task.title} task`);

    const iconChip = document.createElement("div");
    iconChip.className = "palette-icon";
    iconChip.textContent = meta.icon;
    iconChip.style.setProperty("--task-color", meta.color);

    const body = document.createElement("div");
    body.className = "palette-body";
    const title = document.createElement("h4");
    title.textContent = task.title;
    const summary = document.createElement("p");
    summary.textContent = task.summary;
    body.append(title, summary);

    card.addEventListener("click", () => addBlock(section));
    card.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      addBlock(section);
    });
    card.addEventListener("dragstart", (event) => {
      state.draggedTaskSection = section;
      card.classList.add("dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "copy";
        event.dataTransfer.setData("text/plain", `task:${section}`);
      }
    });
    card.addEventListener("dragend", () => {
      state.draggedTaskSection = null;
      card.classList.remove("dragging");
      clearDropTargets();
    });

    card.append(iconChip, body);
    elements.taskPalette.append(card);
  }
}

function renderTemplates() {
  elements.templateList.innerHTML = "";
  for (const template of state.catalog.templates) {
    const card = document.createElement("article");
    card.className = "template-card";
    const head = document.createElement("div");
    head.className = "template-head";
    const copy = document.createElement("div");
    copy.className = "template-copy";
    const title = document.createElement("h4");
    title.textContent = template.title;
    const description = document.createElement("p");
    description.textContent = template.description;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost template-apply";
    button.textContent = "Apply";
    button.addEventListener("click", () => applyTemplate(template));
    copy.append(title, description);
    head.append(copy, button);
    card.append(head);
    elements.templateList.append(card);
  }
}

function renderAdvancedConfig() {
  if (!state.advancedConfig.model_name) {
    state.advancedConfig.model_name = "vit";
  }
  const forceBarspoon = pipelineRequiresBarspoon();
  if (forceBarspoon) {
    state.advancedConfig.model_name = "barspoon";
  }
  elements.advancedConfig.innerHTML = "";
  const grouped = groupFields(state.catalog.advanced_fields, ADVANCED_PANEL_DEFS);
  for (const panel of grouped) {
    const body = document.createElement("div");
    body.className = "field-grid inspector-field-grid compact-advanced-grid";
    for (const field of panel.fields) {
      const isLockedModel = field.name === "model_name" && forceBarspoon;
      const editor = createFieldEditor({
        field,
        value: state.advancedConfig[field.name],
        onChange: (value) => {
          state.advancedConfig[field.name] = value;
          if (field.name === "model_name") {
            renderAdvancedConfig();
          }
        },
        disabled: isLockedModel,
        helpOverride: isLockedModel
          ? "Locked to barspoon because a classification cell is using multi-target labels."
          : null,
      });
      editor.classList.remove("full-span");
      editor.classList.add("compact-field");
      if (field.name === "model_name" || field.name === "accelerator") {
        editor.classList.add("compact-field-wide");
      }
      body.append(editor);
    }
    elements.advancedConfig.append(createInspectorPanel("advanced", panel, body));
  }

  const modelName = state.advancedConfig.model_name;
  const modelParams = state.advancedConfig.model_params?.[modelName];
  if (modelName && modelParams) {
    const paramsBody = document.createElement("div");
    paramsBody.className = "field-grid inspector-field-grid compact-advanced-grid";
    for (const [paramName, paramValue] of Object.entries(modelParams)) {
      const editor = createModelParamEditor(modelName, paramName, paramValue);
      if (typeof paramValue === "number" || typeof paramValue === "boolean") {
        editor.classList.add("compact-field");
      } else {
        editor.classList.add("compact-field-wide");
      }
      paramsBody.append(editor);
    }
    elements.advancedConfig.append(
      createInspectorPanel(
        "advanced",
        {
          id: `model_params_${modelName}`,
          title: `${modelName} Hyperparameters`,
          defaultOpen: true,
          fields: Object.keys(modelParams).map((name) => ({ name })),
        },
        paramsBody,
      ),
    );
  }
}

function renderWorkspaceSummary() {
  elements.workspaceSummary.innerHTML = "";
  if (state.blocks.length === 0) {
    return;
  }
  elements.workspaceSummary.append(
    createMetaPill(`${state.blocks.length} cells`, "info"),
    createMetaPill(`${state.blocks.filter((block) => block.enabled).length} active`, "good"),
    createMetaPill(
      state.blocks.map((block) => state.catalog.tasks[block.section].title).join(" -> "),
      "info",
    ),
  );
}

function renderWorkspaceMessage() {
  const text = state.terminalMessage || "";
  const shouldHide = !text || text === "No run selected." || text === "No run started yet.";
  elements.workspaceMessage.hidden = shouldHide;
  if (shouldHide) {
    elements.workspaceMessage.innerHTML = "";
    return;
  }

  const { tone } = terminalMessageMeta(text);
  elements.workspaceMessage.innerHTML = `<div class="message-card ${tone}">${text.replaceAll("\n", "<br>")}</div>`;
}

function hasDisplayValue(field, value) {
  if (field.kind === "boolean") {
    return value === true;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  return value !== undefined && value !== null && value !== "";
}

function renderBlockSummary(task, block) {
  const summary = document.createElement("div");
  summary.className = "cell-summary";
  const health = validateBlockLocally(task, block);
  summary.append(createMetaPill(health.ready ? "Ready" : `Missing ${health.missing.length}`, health.ready ? "good" : "warn"));
  const fields = task.fields
    .filter((field) => {
      return hasDisplayValue(field, block.params[field.name]);
    })
    .slice(0, 5);

  if (fields.length === 0) {
    summary.append(createMetaPill("No parameters filled yet", "warn"));
    return summary;
  }

  for (const field of fields) {
    summary.append(createMetaPill(`${field.label}: ${friendlyValue(field, block.params[field.name])}`, "info"));
  }
  return summary;
}

function createInspectorPanel(scope, panel, content) {
  const details = document.createElement("details");
  details.className = "inspector-panel";
  details.open = isPanelOpen(scope, panel.id, panel.defaultOpen);
  details.addEventListener("toggle", () => {
    setPanelOpen(scope, panel.id, details.open);
    const toggle = details.querySelector(".inspector-panel-toggle");
    if (toggle) {
      toggle.textContent = details.open ? "▾" : "▸";
    }
  });

  const head = document.createElement("summary");
  head.className = "inspector-panel-head";

  const title = document.createElement("div");
  title.className = "inspector-panel-title";
  title.textContent = panel.title;

  const count = document.createElement("div");
  count.className = "inspector-panel-count";
  count.textContent = `${panel.fields.length}`;

  const toggle = document.createElement("div");
  toggle.className = "inspector-panel-toggle";
  toggle.textContent = details.open ? "▾" : "▸";

  head.append(title, count, toggle);
  details.append(head, content);
  return details;
}

function validateBlockLocally(task, block) {
  const missing = [];
  for (const field of task.fields) {
    if (!field.required) {
      continue;
    }
    const value = block.params[field.name];
    if (
      field.name === "slide_table"
      && block.ui?.autoCreateSlideTable
      && supportsAutoSlideTable(block)
    ) {
      continue;
    }
    const emptyList = Array.isArray(value) && value.length === 0;
    const emptyString = value === undefined || value === null || value === "";
    if (emptyList || emptyString) {
      missing.push(field.label);
    }
  }
  return {
    ready: missing.length === 0,
    missing,
  };
}

function renderInspector() {
  const block = selectedBlock();
  elements.blockInspector.innerHTML = "";

  if (!block) {
    elements.inspectorSelection.textContent = "No cell selected";
    elements.blockInspector.className = "inspector-body empty-state";
    elements.blockInspector.textContent = "Select a pipeline cell to edit its settings.";
    return;
  }

  const index = state.blocks.findIndex((item) => item.id === block.id);
  const task = state.catalog.tasks[block.section];
  const health = validateBlockLocally(task, block);
  elements.inspectorSelection.textContent = `Cell ${index + 1} · ${task.title}`;
  elements.blockInspector.className = "inspector-body";

  const stack = document.createElement("div");
  stack.className = "inspector-stack";

  const lead = document.createElement("div");
  lead.className = "inspector-lead";
  const taskMeta = TASK_META[block.section] || { icon: "◆", color: "#97aabd" };
  const leadIcon = document.createElement("div");
  leadIcon.className = "inspector-lead-icon";
  leadIcon.textContent = taskMeta.icon;
  leadIcon.style.setProperty("--task-color", taskMeta.color);
  const leadBody = document.createElement("div");
  leadBody.className = "inspector-lead-body";
  const leadTitle = document.createElement("div");
  leadTitle.className = "inspector-kicker";
  leadTitle.textContent = task.title;
  const leadText = document.createElement("p");
  leadText.className = "inspector-copy";
  leadText.textContent = task.summary;
  leadBody.append(leadTitle, leadText);
  lead.append(leadIcon, leadBody);

  const meta = document.createElement("div");
  meta.className = "meta-row";
  meta.append(
    createMetaPill(`Cell ${index + 1}`, "info"),
    createMetaPill(TASK_COMMANDS[block.section], "info"),
    createMetaPill(block.enabled ? "Active" : "Disabled", block.enabled ? "good" : "warn"),
    createMetaPill(health.ready ? "Ready to run" : "Needs input", health.ready ? "good" : "warn"),
  );

  const toolbar = document.createElement("div");
  toolbar.className = "inspector-toolbar";

  const toggleButton = document.createElement("button");
  toggleButton.type = "button";
  toggleButton.className = "ghost";
  toggleButton.textContent = block.enabled ? "Disable Cell" : "Enable Cell";
  toggleButton.addEventListener("click", () => {
    block.enabled = !block.enabled;
    renderTopbar();
    renderPipeline();
    renderInspector();
  });

  toolbar.append(toggleButton);

  const grid = document.createElement("div");
  grid.className = "inspector-group-stack";

  const visibleFields = task.fields.filter((field) => fieldVisibleForBlock(field, block));
  const grouped = groupFields(visibleFields, TASK_PANEL_DEFS);
  for (const panel of grouped) {
    const panelBody = document.createElement("div");
    panelBody.className = "field-grid inspector-field-grid";
    for (const field of panel.fields) {
      let editor;
      if (field.name === "ground_truth_label") {
        editor = createGroundTruthEditor(field, block);
      } else if (field.name === "slide_table") {
        editor = createSlideTableEditor(field, block);
      } else {
        editor = createFieldEditor({
          field,
          value: block.params[field.name],
          blockId: block.id,
          onChange: (value) => {
            block.params[field.name] = value;
            if (field.name === "task") {
              if (value !== "classification" && block.ui) {
                block.ui.multiTarget = false;
              }
              if (pipelineRequiresBarspoon()) {
                state.advancedConfig.model_name = "barspoon";
              } else if (!state.advancedConfig.model_name || state.advancedConfig.model_name === "") {
                state.advancedConfig.model_name = "vit";
              }
              renderAdvancedConfig();
              renderInspector();
            }
            renderPipeline();
          },
        });
      }
      panelBody.append(editor);
    }
    grid.append(createInspectorPanel(`block:${block.id}`, panel, panelBody));
  }

  const note = document.createElement("div");
  note.className = "inspector-note";
  note.textContent = health.ready
    ? "This cell has its required inputs and is ready for pipeline validation."
    : `Still required: ${health.missing.join(", ")}. Fill those fields here before running the pipeline.`;

  stack.append(lead, meta, toolbar, grid, note);
  elements.blockInspector.append(stack);
}

function renderPipeline() {
  renderWorkspaceSummary();
  elements.pipelineCanvas.innerHTML = "";

  if (state.blocks.length === 0) {
    elements.pipelineCanvas.append(createDropSlot(0, true));
    return;
  }

  elements.pipelineCanvas.append(createDropSlot(0));

  state.blocks.forEach((block, index) => {
    const task = state.catalog.tasks[block.section];
    const blockRunning = isActiveRunForBlock(block);
    const blockPending = isPendingBlock(block);
    const blockDisabled = !block.enabled;
    const card = document.createElement("article");
    card.className = `cell-card ${block.id === state.selectedBlockId ? "selected" : ""} ${blockPending ? "pending" : ""} ${blockDisabled ? "disabled" : ""}`;
    card.dataset.section = block.section;
    card.tabIndex = 0;
    card.draggable = true;

    card.addEventListener("click", (event) => {
      if (event.target.closest("button, input, textarea, select, a")) {
        return;
      }
      state.selectedBlockId = block.id;
      renderPipeline();
      renderInspector();
    });
    card.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      state.selectedBlockId = block.id;
      renderPipeline();
      renderInspector();
    });
    card.addEventListener("dragstart", (event) => {
      state.draggedBlockId = block.id;
      card.classList.add("dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", `block:${block.id}`);
      }
    });
    card.addEventListener("dragend", () => {
      state.draggedBlockId = null;
      card.classList.remove("dragging");
      clearDropTargets();
    });
    card.addEventListener("dragenter", (event) => {
      const payload = dragPayloadFromEvent(event);
      if (!payload) {
        return;
      }
      event.preventDefault();
      const target = cardDropIndex(card, index, event);
      clearDropTargets();
      card.classList.add(target.position === "before" ? "drop-before" : "drop-after");
      elements.pipelineCanvas.classList.add("drop-active");
    });
    card.addEventListener("dragover", (event) => {
      const payload = dragPayloadFromEvent(event);
      if (!payload) {
        return;
      }
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = payload.kind === "block" ? "move" : "copy";
      }
      const target = cardDropIndex(card, index, event);
      clearDropTargets();
      card.classList.add(target.position === "before" ? "drop-before" : "drop-after");
      elements.pipelineCanvas.classList.add("drop-active");
    });
    card.addEventListener("dragleave", (event) => {
      if (event.relatedTarget && card.contains(event.relatedTarget)) {
        return;
      }
      card.classList.remove("drop-before", "drop-after");
      if (!elements.pipelineCanvas.querySelector(".drop-slot.active, .cell-card.drop-before, .cell-card.drop-after")) {
        elements.pipelineCanvas.classList.remove("drop-active");
      }
    });
    card.addEventListener("drop", (event) => {
      const payload = dragPayloadFromEvent(event);
      if (!payload) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const target = cardDropIndex(card, index, event);
      applyDropPayload(payload, target.index);
    });

    const shell = document.createElement("div");
    shell.className = "cell-shell";

    const gutter = document.createElement("div");
    gutter.className = "cell-index";
    const quickRunButton = document.createElement("button");
    quickRunButton.type = "button";
    quickRunButton.className = `cell-run-icon ${blockRunning ? "running" : "idle"}`;
    quickRunButton.setAttribute("aria-label", blockRunning ? `Terminate ${task.title} cell` : `Run ${task.title} cell`);
    quickRunButton.title = blockRunning ? "Terminate this cell" : "Run this cell";
    quickRunButton.textContent = blockRunning ? "■" : "▶";
    quickRunButton.addEventListener("click", (event) => {
      event.stopPropagation();
      runSingleBlock(block);
    });
    const gutterIndex = document.createElement("div");
    gutterIndex.className = "cell-gutter-index";
    gutterIndex.textContent = `${index + 1}`;
    gutter.append(quickRunButton, gutterIndex);

    const main = document.createElement("div");
    main.className = "cell-main";

    const head = document.createElement("div");
    head.className = "cell-head";
    const headLead = document.createElement("div");
    headLead.className = "cell-head-lead";
    const taskMeta = TASK_META[block.section] || { icon: "◆", color: "#97aabd" };
    const iconChip = document.createElement("div");
    iconChip.className = "cell-icon-chip";
    iconChip.textContent = taskMeta.icon;
    iconChip.style.setProperty("--task-color", taskMeta.color);
    const copy = document.createElement("div");
    copy.className = "cell-head-copy";
    const title = document.createElement("h3");
    title.textContent = task.title;
    copy.append(title);
    headLead.append(iconChip, copy);

    const headActions = document.createElement("div");
    headActions.className = "cell-head-actions";

    const runtimeButton = document.createElement("button");
    runtimeButton.type = "button";
    runtimeButton.className = "ghost cell-runtime-toggle";
    runtimeButton.textContent = block.ui?.showRuntime ? "Hide Runtime" : "Show Runtime";
    runtimeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      block.ui = block.ui || {};
      block.ui.showRuntime = !block.ui.showRuntime;
      renderPipeline();
    });

    const enableLabel = document.createElement("label");
    enableLabel.className = "cell-toggle";
    const enableInput = document.createElement("input");
    enableInput.type = "checkbox";
    enableInput.checked = block.enabled;
    enableInput.addEventListener("click", (event) => event.stopPropagation());
    enableInput.addEventListener("change", () => {
      block.enabled = enableInput.checked;
      renderPipeline();
      renderInspector();
      renderTopbar();
    });
    const enableText = document.createElement("span");
    enableText.textContent = "Enable";
    enableLabel.append(enableInput, enableText);

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "ghost cell-remove-icon";
    removeButton.setAttribute("aria-label", `Remove ${task.title} cell`);
    removeButton.title = "Remove cell";
    removeButton.textContent = "x";
    removeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      removeBlock(block.id);
    });

    headActions.append(enableLabel, removeButton);
    head.append(headLead, headActions);
    main.append(head);
    main.append(renderBlockSummary(task, block));
    if (block.ui?.showRuntime) {
      main.append(renderBlockRuntime(block));
    }
    const runtimeToggleRow = document.createElement("div");
    runtimeToggleRow.className = "cell-runtime-toggle-row";
    runtimeToggleRow.append(runtimeButton);
    main.append(runtimeToggleRow);

    shell.append(gutter, main);
    card.append(shell);
    elements.pipelineCanvas.append(card);
    elements.pipelineCanvas.append(createDropSlot(index + 1));
  });
}

function currentStage(run) {
  if (!run) {
    return null;
  }
  return run.stages.find((stage) => stage.status === "running")
    || run.stages.find((stage) => stage.status === "failed")
    || run.stages.find((stage) => stage.status === "terminated")
    || run.stages.find((stage) => stage.status === "pending")
    || run.stages[run.stages.length - 1]
    || null;
}

function renderRunPanel() {
  const run = state.activeRun;
  elements.runPill.className = `pill ${runPillClass(run?.status || "idle")}`;
  elements.runPill.textContent = run?.status || "Idle";

  elements.runList.innerHTML = "";
  if (state.runs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "workspace-empty";
    empty.innerHTML = "<p>No runs yet.</p>";
    elements.runList.append(empty);
  } else {
    for (const entry of state.runs) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "run-card";
      card.addEventListener("click", async () => {
        state.activeRunId = entry.run_id;
        await refreshActiveRun();
      });

      const top = document.createElement("div");
      top.className = "run-top";
      const name = document.createElement("strong");
      name.textContent = entry.run_id;
      const pill = document.createElement("span");
      pill.className = `meta-pill ${statusClass(entry.status)}`;
      pill.textContent = entry.status;
      top.append(name, pill);

      const desc = document.createElement("p");
      const label = entry.scope === "block" && entry.block_title ? `${entry.block_title}: ` : "";
      desc.textContent = label + entry.stages.map((stage) => stage.command).join(" -> ");
      card.append(top, desc);
      elements.runList.append(card);
    }
  }

  const canRun = state.blocks.some((block) => block.enabled);
  const runnable = Boolean(run && ["running", "queued", "stopping", "terminating"].includes(run.status));
  const runIcon = elements.runButton.querySelector(".button-icon");
  const runText = elements.runButton.querySelector("span:last-child");
  if (runnable) {
    elements.runButton.classList.add("danger");
    elements.runButton.classList.remove("primary");
    if (runIcon) runIcon.textContent = "■";
    if (runText) runText.textContent = "Interrupt";
    elements.runButton.disabled = false;
  } else {
    elements.runButton.classList.add("primary");
    elements.runButton.classList.remove("danger");
    if (runIcon) runIcon.textContent = "▶";
    if (runText) runText.textContent = "Run All";
    elements.runButton.disabled = !canRun;
  }
}

function isRunnableRunStatus(status) {
  return ["queued", "running", "stopping", "terminating"].includes(status);
}

function isActiveRunForBlock(block) {
  return Boolean(
    state.transientRunningBlockId === block.id
    || (
      state.activeRun
      && state.activeRun.block_id === block.id
      && isRunnableRunStatus(state.activeRun.status)
    ),
  );
}

function hasActiveSingleBlockRun() {
  return Boolean(
    state.transientRunningBlockId
    || (
      state.activeRun
      && state.activeRun.scope === "block"
      && state.activeRun.block_id
      && isRunnableRunStatus(state.activeRun.status)
    ),
  );
}

function isPendingBlock(block) {
  return Boolean(
    hasActiveSingleBlockRun()
    && (
      state.transientRunningBlockId
        ? state.transientRunningBlockId !== block.id
        : state.activeRun.block_id !== block.id
    ),
  );
}

function syncRuntimeVisibilityFromRun() {
  const run = state.activeRun;
  if (!run) {
    return;
  }

  if (run.scope === "block") {
    const block = state.blocks.find((item) => item.id === run.block_id);
    if (block) {
      block.ui = block.ui || {};
      block.ui.showRuntime = true;
    }
    return;
  }

  const enabledBlocks = state.blocks.filter((item) => item.enabled);
  let lastVisibleStage = -1;
  run.stages.forEach((stage, index) => {
    if (stage.status && stage.status !== "pending") {
      lastVisibleStage = index;
    }
  });

  enabledBlocks.forEach((block, index) => {
    if (index <= lastVisibleStage) {
      block.ui = block.ui || {};
      block.ui.showRuntime = true;
    }
  });
}

function runtimeForBlock(block) {
  const run = state.activeRun;
  if (!run) {
    return null;
  }

  if (run.scope === "block") {
    if (run.block_id !== block.id) {
      return null;
    }
    return {
      runId: run.run_id,
      status: run.status,
      command: run.stages[0]?.command || TASK_COMMANDS[block.section],
      logs: run.logs || [],
      warnings: run.warnings || [],
      errors: run.errors || [],
    };
  }

  const enabledBlocks = state.blocks.filter((item) => item.enabled);
  const stageIndex = enabledBlocks.findIndex((item) => item.id === block.id);
  if (stageIndex === -1 || stageIndex >= run.stages.length) {
    return null;
  }

  const stage = run.stages[stageIndex];
  return {
    runId: run.run_id,
    status: stage.status || run.status,
    command: stage.command || TASK_COMMANDS[block.section],
    logs: stage.logs || [],
    warnings: stageIndex === 0 ? (run.warnings || []) : [],
    errors: stage.status === "failed" || stage.status === "terminated" ? (run.errors || []) : [],
  };
}

function renderBlockRuntime(block) {
  const runtime = runtimeForBlock(block);
  const panel = document.createElement("div");
  panel.className = "cell-runtime";

  const output = document.createElement("pre");
  output.className = "terminal-output cell-runtime-output";

  if (!runtime) {
    output.textContent = block.enabled
      ? "No runtime attached to this cell yet."
      : "This disabled cell has no runtime in the current run.";
  } else {
    const warnings = runtime.warnings?.length ? `\n[warnings]\n${runtime.warnings.join("\n")}` : "";
    const errors = runtime.errors?.length ? `\n[errors]\n${runtime.errors.join("\n\n")}` : "";
    output.textContent = (runtime.logs.join("\n") || "Waiting for stage output...") + warnings + errors;
  }

  panel.append(output);
  return panel;
}

function renderAll() {
  renderTopbar();
  renderTaskPalette();
  renderTemplates();
  renderAdvancedConfig();
  renderWorkspaceMessage();
  renderPipeline();
  renderInspector();
  renderRunPanel();
}

function addBlock(section) {
  insertBlock(section);
}

function saveCandidates() {
  const sectionCounts = new Map();
  for (const block of state.blocks) {
    sectionCounts.set(block.section, (sectionCounts.get(block.section) || 0) + 1);
  }

  const seenBySection = new Map();
  return state.blocks.map((block, index) => {
    const seen = seenBySection.get(block.section) || 0;
    seenBySection.set(block.section, seen + 1);
    return {
      blockId: block.id,
      section: block.section,
      title: state.catalog.tasks[block.section].title,
      index,
      duplicateIndex: seen + 1,
      duplicateCount: sectionCounts.get(block.section) || 1,
      enabled: block.enabled,
    };
  });
}

function setSaveModalMessage(message = "", tone = "muted") {
  elements.saveModalMessage.className = `modal-message ${tone} small`;
  elements.saveModalMessage.textContent = message;
}

function openSaveModal() {
  const candidates = saveCandidates();
  if (candidates.length === 0) {
    setTerminalMessage("[workbench] Add at least one cell before saving a config.");
    return;
  }

  elements.saveSectionList.innerHTML = "";
  const seenCheckedSections = new Set();
  for (const candidate of candidates) {
    const label = document.createElement("label");
    label.className = "save-section-option";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = candidate.blockId;
    checkbox.dataset.section = candidate.section;
    const shouldCheck = !seenCheckedSections.has(candidate.section);
    checkbox.checked = shouldCheck;
    if (shouldCheck) {
      seenCheckedSections.add(candidate.section);
    }
    checkbox.addEventListener("change", () => {
      if (!checkbox.checked) {
        return;
      }
      elements.saveSectionList
        .querySelectorAll(`input[type="checkbox"][data-section="${candidate.section}"]`)
        .forEach((peer) => {
          if (peer !== checkbox) {
            peer.checked = false;
          }
        });
    });

    const copy = document.createElement("div");
    copy.className = "save-section-copy";

    const title = document.createElement("div");
    title.className = "save-section-title";
    title.textContent = candidate.title;

    const titleRow = document.createElement("div");
    titleRow.className = "save-section-title-row";

    const indexBadge = document.createElement("span");
    indexBadge.className = "save-section-index";
    indexBadge.textContent = `${candidate.index + 1}`;

    const meta = document.createElement("div");
    meta.className = "save-section-meta";
    const cellLabel = `Cell ${candidate.index + 1}`;
    if (candidate.duplicateCount > 1) {
      meta.textContent = `${cellLabel} · ${candidate.duplicateIndex}/${candidate.duplicateCount} for this task`;
    } else {
      meta.textContent = `${cellLabel} · ${candidate.enabled ? "enabled" : "disabled"}`;
    }

    titleRow.append(indexBadge, title);
    copy.append(titleRow, meta);
    label.append(checkbox, copy);
    elements.saveSectionList.append(label);
  }

  elements.saveModalCount.textContent = `${candidates.length} pipeline cell${candidates.length === 1 ? "" : "s"}`;
  elements.saveFilename.value = elements.saveFilename.value?.trim() || "config.yaml";
  if (!elements.saveOutputDir.value.trim()) {
    elements.saveOutputDir.value = state.catalog.environment.cwd;
  }
  setSaveModalMessage("");
  elements.saveModal.hidden = false;
}

function closeSaveModal() {
  elements.saveModal.hidden = true;
  setSaveModalMessage("");
}

async function saveConfigFile() {
  const selectedBlockIds = Array.from(
    elements.saveSectionList.querySelectorAll('input[type="checkbox"]:checked'),
  ).map((input) => input.value);

  if (selectedBlockIds.length === 0) {
    setSaveModalMessage("Select at least one task cell to save.", "warning");
    return;
  }

  const outputDir = elements.saveOutputDir.value.trim();
  const filename = elements.saveFilename.value.trim() || "config.yaml";
  if (!outputDir) {
    setSaveModalMessage("Enter an output directory.", "warning");
    return;
  }

  elements.saveConfirmButton.disabled = true;
  setSaveModalMessage("Saving config...", "success");
  try {
    const payload = await request("/api/export-config", {
      method: "POST",
      body: JSON.stringify({
        ...pipelinePayload(),
        selected_block_ids: selectedBlockIds,
        output_dir: outputDir,
        filename,
      }),
    });
    const warningText = payload.warnings?.length ? `\n${payload.warnings.join("\n")}` : "";
    setTerminalMessage(`[workbench] Saved config to ${payload.path}.${warningText}`);
    closeSaveModal();
  } catch (error) {
    setSaveModalMessage(error.message, "error");
  } finally {
    elements.saveConfirmButton.disabled = false;
  }
}

async function loadConfigFile(file) {
  if (!file) {
    setTerminalMessage("[workbench] Choose a config.yaml file first.");
    return;
  }

  try {
    setTerminalMessage(`[workbench] Loading ${file.name}...`);
    const content = await file.text();
    const payload = await request("/api/import-config", {
      method: "POST",
      body: JSON.stringify({
        content,
        filename: file.name,
      }),
    });

    state.blocks = (payload.blocks || []).map((block) => ({
      ...block,
      enabled: block.enabled !== false,
      ui: {
        multiTarget: Boolean(block.ui?.multiTarget),
        autoCreateSlideTable: Boolean(block.ui?.autoCreateSlideTable),
      },
    }));
    state.selectedBlockId = state.blocks[0]?.id || null;
    state.advancedConfig = deepClone(payload.advanced_config || state.catalog.advanced_defaults);
    state.validation = null;
    renderAll();

    const warningText = payload.warnings?.length ? `\n${payload.warnings.join("\n")}` : "";
    setTerminalMessage(
      `[workbench] Loaded ${payload.source}. Imported ${state.blocks.length} pipeline cell(s).${warningText}`,
    );
  } catch (error) {
    setTerminalMessage(`[workbench] Failed to load config.\n${error.message}`);
  }
}

function openConfigPicker() {
  elements.configFileInput.click();
}

function applyTemplate(template) {
  state.blocks = template.sections.map((section) => makeBlock(section));
  state.selectedBlockId = state.blocks[0]?.id || null;
  state.validation = null;
  renderAll();
}

function moveBlock(fromIndex, toIndex) {
  if (toIndex < 0 || toIndex >= state.blocks.length) {
    return;
  }
  const [block] = state.blocks.splice(fromIndex, 1);
  state.blocks.splice(toIndex, 0, block);
  renderAll();
}

function reorderBlock(blockId, toIndex) {
  const fromIndex = state.blocks.findIndex((block) => block.id === blockId);
  if (fromIndex === -1) {
    return;
  }
  const [block] = state.blocks.splice(fromIndex, 1);
  const adjustedIndex = fromIndex < toIndex ? toIndex - 1 : toIndex;
  const safeIndex = Math.max(0, Math.min(adjustedIndex, state.blocks.length));
  state.blocks.splice(safeIndex, 0, block);
  state.selectedBlockId = block.id;
  renderAll();
}

function removeBlock(blockId) {
  state.blocks = state.blocks.filter((block) => block.id !== blockId);
  if (state.selectedBlockId === blockId) {
    state.selectedBlockId = state.blocks[0]?.id || null;
  }
  renderAll();
}

function clearDropTargets() {
  document.querySelectorAll(".drop-slot.active, .notebook-canvas.drop-active, .cell-card.dragging, .cell-card.drop-before, .cell-card.drop-after").forEach((node) => {
    node.classList.remove("active");
    node.classList.remove("drop-active");
    node.classList.remove("dragging");
    node.classList.remove("drop-before");
    node.classList.remove("drop-after");
  });
}

function dragPayloadFromEvent(event) {
  const fromTransfer = event.dataTransfer?.getData("text/plain");
  if (fromTransfer?.startsWith("task:")) {
    return { kind: "task", value: fromTransfer.slice(5) };
  }
  if (fromTransfer?.startsWith("block:")) {
    return { kind: "block", value: fromTransfer.slice(6) };
  }
  if (state.draggedBlockId) {
    return { kind: "block", value: state.draggedBlockId };
  }
  if (state.draggedTaskSection) {
    return { kind: "task", value: state.draggedTaskSection };
  }
  return null;
}

function applyDropPayload(payload, index) {
  if (!payload) {
    return;
  }
  clearDropTargets();
  if (payload.kind === "block") {
    const blockId = payload.value;
    state.draggedBlockId = null;
    reorderBlock(blockId, index);
    return;
  }
  state.draggedTaskSection = null;
  insertBlock(payload.value, index);
}

function cardDropIndex(card, index, event) {
  const rect = card.getBoundingClientRect();
  const insertBefore = event.clientY < rect.top + (rect.height / 2);
  return {
    index: insertBefore ? index : index + 1,
    position: insertBefore ? "before" : "after",
  };
}

function createDropSlot(index, isEmpty = false) {
  const slot = document.createElement("div");
  slot.className = `drop-slot ${isEmpty ? "empty-drop" : ""}`;
  slot.innerHTML = isEmpty
    ? "<div class=\"empty-drop-copy\"><h3>Start Building</h3><p>Drag a task from the left library into this workspace, or click a task to create the first cell.</p></div>"
    : "";

  slot.addEventListener("dragenter", (event) => {
    const payload = dragPayloadFromEvent(event);
    if (!payload) {
      return;
    }
    event.preventDefault();
    slot.classList.add("active");
    elements.pipelineCanvas.classList.add("drop-active");
  });

  slot.addEventListener("dragover", (event) => {
    const payload = dragPayloadFromEvent(event);
    if (!payload) {
      return;
    }
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = payload.kind === "block" ? "move" : "copy";
    }
    slot.classList.add("active");
    elements.pipelineCanvas.classList.add("drop-active");
  });

  slot.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && slot.contains(event.relatedTarget)) {
      return;
    }
    slot.classList.remove("active");
    if (!elements.pipelineCanvas.querySelector(".drop-slot.active")) {
      elements.pipelineCanvas.classList.remove("drop-active");
    }
  });

  slot.addEventListener("drop", (event) => {
    const payload = dragPayloadFromEvent(event);
    if (!payload) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    applyDropPayload(payload, index);
  });

  return slot;
}

async function validatePipeline() {
  setTerminalMessage("[workbench] Validating pipeline...");
  try {
    state.validation = await request("/api/validate", {
      method: "POST",
      body: JSON.stringify(pipelinePayload()),
    });
  } catch (error) {
    state.validation = error.payload || {
      valid: false,
      errors: [error.message],
      warnings: [],
      config_preview: "",
    };
  }
  if (!state.validation?.valid) {
    const errors = state.validation?.errors?.length ? state.validation.errors.join("\n") : "Validation failed.";
    setTerminalMessage(`[workbench] Validation failed.\n${errors}`);
  } else {
    const warnings = state.validation?.warnings?.length
      ? `\n[workbench] Warnings:\n${state.validation.warnings.join("\n")}`
      : "";
    setTerminalMessage(`[workbench] Validation passed. Ready to start.${warnings}`);
  }
  return Boolean(state.validation?.valid);
}

async function runPipeline() {
  const isValid = await validatePipeline();
  if (!isValid) {
    return;
  }
  setTerminalMessage("[workbench] Creating run...");
  try {
    const run = await request("/api/runs", {
      method: "POST",
      body: JSON.stringify(pipelinePayload()),
    });
    state.activeRunId = run.run_id;
    state.blocks.forEach((block) => {
      if (block.enabled) {
        block.ui = block.ui || {};
        block.ui.showRuntime = true;
      }
    });
    setTerminalMessage(`[workbench] Run ${run.run_id} created. Waiting for stage output...`);
    await refreshRuns();
    await refreshActiveRun();
  } catch (error) {
    state.validation = {
      valid: false,
      errors: [error.message],
      warnings: [],
      config_preview: state.validation?.config_preview || "",
    };
    setTerminalMessage(`[workbench] Failed to create run.\n${state.validation.errors.join("\n")}`);
  }
}

async function runSingleBlock(block) {
  const task = state.catalog.tasks[block.section];

  if (isActiveRunForBlock(block)) {
    state.transientRunningBlockId = block.id;
    renderPipeline();
    setTerminalMessage(`[workbench] Terminating ${task.title} cell...`);
    try {
      await request(`/api/runs/${state.activeRun.run_id}/terminate`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      state.transientRunningBlockId = null;
      await refreshRuns();
      await refreshActiveRun();
    } catch (error) {
      state.transientRunningBlockId = null;
      renderPipeline();
      setTerminalMessage(`[workbench] Failed to terminate ${task.title} cell.\n${error.message}`);
    }
    return;
  }

  state.transientRunningBlockId = block.id;
  block.ui = block.ui || {};
  block.ui.showRuntime = true;
  renderPipeline();
  setTerminalMessage(`[workbench] Validating ${task.title} cell...`);
  try {
    const run = await request("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        blocks: [{
          id: block.id,
          section: block.section,
          enabled: true,
          params: block.params,
        }],
        advanced_config: state.advancedConfig,
        scope: "block",
        block_id: block.id,
        block_title: task.title,
      }),
    });
    state.activeRunId = run.run_id;
    setTerminalMessage(`[workbench] ${task.title} cell started. Waiting for stage output...`);
    await refreshRuns();
    await refreshActiveRun();
  } catch (error) {
    state.transientRunningBlockId = null;
    renderPipeline();
    setTerminalMessage(`[workbench] Failed to run ${task.title} cell.\n${error.message}`);
  }
}

async function startRun() {
  if (state.activeRun && state.activeRun.status === "stopped") {
    setTerminalMessage(`[workbench] Restarting run ${state.activeRun.run_id}...`);
    try {
      await request(`/api/runs/${state.activeRun.run_id}/start`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      await refreshRuns();
      await refreshActiveRun();
      return;
    } catch (error) {
      state.validation = {
        valid: false,
        errors: [error.message],
        warnings: [],
        config_preview: state.validation?.config_preview || "",
      };
      setTerminalMessage(`[workbench] Failed to restart run.\n${state.validation.errors.join("\n")}`);
      return;
    }
  }
  setTerminalMessage("[workbench] Starting a new run...");
  await runPipeline();
}

async function stopRun() {
  if (!state.activeRunId) {
    return;
  }
  try {
    await request(`/api/runs/${state.activeRunId}/stop`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    await refreshRuns();
    await refreshActiveRun();
  } catch (error) {
    state.validation = { valid: false, errors: [error.message], warnings: [], config_preview: "" };
    setTerminalMessage(`[workbench] Failed to stop run.\n${state.validation.errors.join("\n")}`);
  }
}

async function terminateRun() {
  if (!state.activeRunId) {
    return;
  }
  try {
    await request(`/api/runs/${state.activeRunId}/terminate`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    await refreshRuns();
    await refreshActiveRun();
  } catch (error) {
    state.validation = { valid: false, errors: [error.message], warnings: [], config_preview: "" };
    setTerminalMessage(`[workbench] Failed to terminate run.\n${state.validation.errors.join("\n")}`);
  }
}

async function handleRunAllButton() {
  if (state.activeRun && ["running", "queued", "stopping", "terminating"].includes(state.activeRun.status)) {
    await terminateRun();
    return;
  }
  await startRun();
}

async function refreshRuns() {
  const payload = await request("/api/runs");
  state.runs = payload.runs;
  const activeRunning = state.runs.find((run) => isRunnableRunStatus(run.status));
  if (!state.activeRunId) {
    state.activeRunId = activeRunning?.run_id || null;
  } else if (!state.runs.some((run) => run.run_id === state.activeRunId)) {
    state.activeRunId = activeRunning?.run_id || null;
  }
}

async function refreshActiveRun() {
  if (!state.activeRunId) {
    state.activeRun = null;
    state.transientRunningBlockId = null;
    renderRunPanel();
    renderPipeline();
    return;
  }
  try {
    state.activeRun = await request(`/api/runs/${state.activeRunId}`);
  } catch (_error) {
    state.activeRun = null;
  }
  syncRuntimeVisibilityFromRun();
  if (!state.activeRun || !isRunnableRunStatus(state.activeRun.status)) {
    state.transientRunningBlockId = null;
  }
  renderRunPanel();
  renderPipeline();
}

function startPolling() {
  if (state.pollingHandle) {
    window.clearInterval(state.pollingHandle);
  }
  state.pollingHandle = window.setInterval(async () => {
    try {
      await refreshRuns();
      if (state.activeRunId) {
        await refreshActiveRun();
      } else {
        renderRunPanel();
      }
    } catch (_error) {
      // Keep the current UI state if polling fails.
    }
  }, 1500);
}

function wireEvents() {
  elements.saveButton.addEventListener("click", () => openSaveModal());
  elements.saveModalClose.addEventListener("click", () => closeSaveModal());
  elements.saveCancelButton.addEventListener("click", () => closeSaveModal());
  elements.saveConfirmButton.addEventListener("click", () => saveConfigFile());
  elements.saveModal.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.dataset.closeSaveModal === "true") {
      closeSaveModal();
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.saveModal.hidden) {
      closeSaveModal();
    }
  });
  elements.configDropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) {
      return;
    }
    openConfigPicker();
  });
  elements.configDropzone.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    openConfigPicker();
  });
  elements.loadConfigButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openConfigPicker();
  });
  elements.configFileInput.addEventListener("change", () => {
    const file = elements.configFileInput.files?.[0];
    if (file) {
      loadConfigFile(file);
    }
  });
  elements.configDropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.configDropzone.classList.add("is-dragover");
  });
  elements.configDropzone.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && elements.configDropzone.contains(event.relatedTarget)) {
      return;
    }
    elements.configDropzone.classList.remove("is-dragover");
  });
  elements.configDropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.configDropzone.classList.remove("is-dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      loadConfigFile(file);
    }
  });
  elements.clearButton.addEventListener("click", () => {
    clearTerminalMessageTimer();
    state.blocks = [];
    state.selectedBlockId = null;
    state.validation = null;
    state.terminalMessage = "No run selected.";
    renderAll();
  });
  elements.runButton.addEventListener("click", () => handleRunAllButton());
  elements.themeToggle.addEventListener("click", () => toggleTheme());
}

async function bootstrap() {
  const payload = await request("/api/bootstrap");
  state.catalog = payload;
  state.blocks = [];
  state.advancedConfig = deepClone(payload.advanced_defaults);
  state.runs = payload.runs || [];
  state.activeRunId = state.runs.find((run) => isRunnableRunStatus(run.status))?.run_id || null;
  if (!state.activeRunId) {
    state.terminalMessage = "No run selected.";
  }
  applyStoredLayout();
  renderAll();
  if (state.activeRunId) {
    await refreshActiveRun();
  }
  startPolling();
}

function init() {
  initTheme();
  wireEvents();
  initResizer(elements.resizerLeftMain, "left");
  initResizer(elements.resizerMainRight, "right");
  bootstrap().catch((error) => {
    elements.pipelineCanvas.innerHTML = `<div class="message-card error">Failed to load workbench: ${error.message}</div>`;
  });
}

window.addEventListener("DOMContentLoaded", init);
