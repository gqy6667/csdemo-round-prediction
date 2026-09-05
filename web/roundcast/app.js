'use strict';

const MODEL_PAIRS = [['pre_round', 'xgboost'], ['pre_round', 'lightgbm'], ['post_first_kill', 'xgboost'], ['post_first_kill', 'lightgbm']];
const modelKey = (stage, algorithm) => `${algorithm === 'xgboost' ? 'xgb' : 'lgbm'}_${stage}`;
const emptyComparisons = () => Object.fromEntries(MODEL_PAIRS.map(([stage, algorithm]) =>
  [modelKey(stage, algorithm), { stage, algorithm, status: 'idle', result: null, error: '' }]));
function validSnapshot(detail, selection) {
  const f = detail?.features;
  if (detail?.example_id !== selection.example_id || detail?.stage !== selection.stage || !f || Array.isArray(f)
      || typeof f.map_name !== 'string' || !f.map_name.startsWith('de_')
      || !Number.isInteger(f.round_num) || f.round_num < 1
      || !['ct_score','t_score','ct_cash','t_cash','ct_eq_value','t_eq_value'].every(k =>
        typeof f[k] === 'number' && Number.isFinite(f[k]) && f[k] >= 0)) return false;
  if (selection.stage === 'pre_round') return !Object.keys(f).some(k => k.startsWith('first_kill_'));
  return [-1,1].includes(f.first_kill_advantage_ct) && [0,1,false,true].includes(f.first_kill_headshot)
    && typeof f.first_kill_time === 'number' && Number.isFinite(f.first_kill_time) && f.first_kill_time >= 0 && f.first_kill_time <= 180
    && typeof f.first_kill_weapon === 'string' && f.first_kill_weapon.trim().length > 0;
}

class RoundcastController {
  constructor(api, render = () => {}) {
    this.api = api;
    this.render = render;
    this.generation = 0; this.caseGeneration = 0;
    this.outcomeGeneration = 0;
    this.state = { selection: { example_id: 'A', stage: 'pre_round', algorithm: 'xgboost' },
      status: 'loading', ready: false, result: null, detail: null, outcome: null, outcomeStatus: 'idle', error: '',
      models: [], examples: [], comparisons: emptyComparisons(), comparing: false, detailError: '' };
  }
  notify() { this.render(this.state); }
  valid(selection) {
    return MODEL_PAIRS.some(([stage, algorithm]) => stage === selection.stage && algorithm === selection.algorithm)
      && this.state.examples.some(e => e.example_id === selection.example_id && e.inference_ready)
      && this.state.models.some(m => m.stage === selection.stage && m.algorithm === selection.algorithm
        && m.model_id === modelKey(selection.stage, selection.algorithm) && m.inference_ready && m.available_examples.includes(selection.example_id));
  }
  sync() {
    const row = this.state.comparisons[modelKey(this.state.selection.stage, this.state.selection.algorithm)];
    this.state.status = this.state.detailError ? 'error' : !this.state.ready ? 'loading' : row.status;
    this.state.result = this.state.ready && row.status === 'success' ? row.result : null;
    this.state.error = this.state.detailError || (this.state.ready ? row.error : '');
    this.notify();
  }
  async readDetail() {
    const token = ++this.generation, selection = { ...this.state.selection };
    Object.assign(this.state, { ready: false, detail: null, result: null, detailError: '' }); this.sync();
    try {
      const path = `/api/examples/${selection.example_id}` + (selection.stage === 'pre_round' ? '' : `/snapshots/${selection.stage}`);
      const detail = await this.api(path);
      if (token !== this.generation) return;
      if (!validSnapshot(detail, selection)) throw Error('回合快照与当前选择不一致或展示字段无效，请重新连接');
      Object.assign(this.state, { ready: true, detail });
    } catch (error) {
      if (token !== this.generation) return;
      this.state.detailError = error.message;
    }
    this.sync();
  }
  async select(change) {
    const selection = { ...this.state.selection, ...change };
    if (!this.valid(selection)) return false;
    if (selection.example_id !== this.state.selection.example_id) {
      ++this.caseGeneration; ++this.outcomeGeneration;
      Object.assign(this.state, { comparisons: emptyComparisons(), comparing: false, outcome: null, outcomeStatus: 'idle' });
    }
    this.state.selection = selection;
    await this.readDetail();
    return true;
  }
  async load() {
    const token = ++this.generation; ++this.caseGeneration; ++this.outcomeGeneration;
    Object.assign(this.state, { status: 'loading', ready: false, result: null, detail: null, outcome: null, outcomeStatus: 'idle', error: '',
      comparisons: emptyComparisons(), comparing: false, detailError: '' });
    this.notify();
    try {
      const [models, examples] = await Promise.all([this.api('/api/models'), this.api('/api/examples')]);
      if (token !== this.generation) return;
      if (!Array.isArray(models.models) || !Array.isArray(examples.examples)) throw Error('模型清单不可用');
      Object.assign(this.state, { models: models.models, examples: examples.examples });
      if (!this.valid(this.state.selection)) throw Error('当前组合尚未就绪');
      await this.readDetail();
    } catch (error) {
      if (token !== this.generation) return;
      Object.assign(this.state, { status: 'error', detailError: error.message, error: error.message });
    }
    this.notify();
  }
  async run() {
    return this.runPair(this.state.selection.stage, this.state.selection.algorithm);
  }
  async runPair(stage, algorithm) {
    const selection = { example_id: this.state.selection.example_id, stage, algorithm };
    if (!this.state.ready || this.state.comparing || !this.valid(selection)) return;
    return this.predict(selection, this.caseGeneration);
  }
  async predict(selection, epoch) {
    const key = modelKey(selection.stage, selection.algorithm), row = this.state.comparisons[key];
    if (row.status === 'running') return;
    Object.assign(row, { status: 'running', result: null, error: '' }); this.sync();
    try {
      const result = await this.api('/api/predict', selection);
      if (epoch !== this.caseGeneration || this.state.comparisons[key] !== row) return;
      const p = result?.prediction;
      if (!result || result.status !== 'success' || result.model_id !== key || !result.request_id
          || Object.keys(selection).some(k => result[k] !== selection[k])
          || !p || ![p.ct_win_probability, p.t_win_probability].every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1)
          || Math.abs(p.ct_win_probability + p.t_win_probability - 1) > 1e-12) throw Error('返回结果与当前选择不一致，请重新运行');
      Object.assign(row, { status: 'success', result });
    } catch (error) {
      if (epoch !== this.caseGeneration || this.state.comparisons[key] !== row) return;
      Object.assign(row, { status: 'error', result: null, error: error.message });
    }
    this.sync();
    return row.result;
  }
  async runAll() {
    if (!this.state.ready || this.state.comparing || Object.values(this.state.comparisons).some(row => row.status === 'running')) return;
    const epoch = this.caseGeneration, example_id = this.state.selection.example_id;
    Object.assign(this.state, { comparisons: emptyComparisons(), comparing: true }); this.sync();
    for (const [stage, algorithm] of MODEL_PAIRS) {
      if (epoch !== this.caseGeneration) return;
      await this.predict({ example_id, stage, algorithm }, epoch);
    }
    if (epoch === this.caseGeneration) { this.state.comparing = false; this.sync(); }
  }
  async reveal() {
    if (!this.state.ready || this.state.outcomeStatus === 'loading') return;
    const token = ++this.outcomeGeneration, example_id = this.state.selection.example_id;
    Object.assign(this.state, { outcome: null, outcomeStatus: 'loading' }); this.notify();
    try {
      const outcome = await this.api(`/api/examples/${example_id}/outcome`);
      if (token !== this.outcomeGeneration) return;
      if (outcome.example_id !== this.state.selection.example_id || !['CT', 'T'].includes(outcome.winning_side)) throw Error('赛果不可用');
      Object.assign(this.state, { outcome, outcomeStatus: 'success' });
    } catch {
      if (token !== this.outcomeGeneration) return;
      Object.assign(this.state, { outcome: null, outcomeStatus: 'error' });
    }
    this.notify();
  }
}

function registerRoundcastTool(context, controller) {
  if (!context?.registerTool) return () => {};
  const lifecycle = new AbortController();
  const tool = { name: 'run_current_round_prediction', title: '运行当前回合预测',
    description: 'Run the same local frozen-model prediction as the visible button. Does not reveal the outcome.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false, untrustedContentHint: true },
    async execute(input) {
      if (!input || Array.isArray(input) || typeof input !== 'object' || Object.keys(input).length) throw Error('No input fields are accepted');
      if (!controller.state.ready || controller.state.status === 'running') throw Error('Current combination is unavailable or busy');
      const result = await controller.run();
      if (!result) throw Error('Prediction failed; inspect the visible error');
      return { request_id: result.request_id, model_id: result.model_id, prediction: result.prediction };
    } };
  try { Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(() => lifecycle.abort()); }
  catch { lifecycle.abort(); }
  return () => lifecycle.abort();
}
if (typeof module !== 'undefined') module.exports = { RoundcastController, registerRoundcastTool };

if (typeof document !== 'undefined') {
  const byId = id => document.getElementById(id);
  const setText = (id, text) => { byId(id).textContent = text; };
  const percent = value => `${(value * 100).toFixed(2)}%`;
  const stageLabel = stage => stage === 'pre_round' ? '购买结束' : '首杀后';
  const algorithmLabel = algorithm => algorithm === 'xgboost' ? 'XGBoost' : 'LightGBM';
  const api = async (url, body) => {
    let response;
    try {
      response = await fetch(url, { method: body ? 'POST' : 'GET', cache: 'no-store',
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(15000) });
    } catch { throw Error('无法连接本地服务，请确认服务已启动后重试'); }
    const data = await response.json();
    if (!response.ok) throw Error(data.message || '请求失败，请重试');
    return data;
  };
  function economy(features) {
    const chart = byId('economy-chart'); chart.replaceChildren();
    for (const [label, key] of [['装备价值', 'eq_value'], ['剩余现金', 'cash']]) {
      const row = document.createElement('div'); row.className = 'economy-row';
      const name = document.createElement('span'); name.className = 'economy-label'; name.textContent = label; row.append(name);
      const lines = document.createElement('div'); lines.className = 'economy-lines'; row.append(lines);
      const max = Math.max(features[`ct_${key}`], features[`t_${key}`], 1);
      for (const side of ['ct', 't']) {
        const value = features[`${side}_${key}`];
        const line = document.createElement('div'); line.className = 'economy-line';
        const sideLabel = document.createElement('span'); sideLabel.textContent = side.toUpperCase();
        const track = document.createElement('div'); track.className = 'track';
        const fill = document.createElement('i'); fill.style.width = `${value / max * 100}%`; track.append(fill);
        const number = document.createElement('strong'); number.textContent = `$${value.toLocaleString('en-US')}`;
        line.append(sideLabel, track, number); lines.append(line);
      }
      chart.append(row);
    }
  }
  function render(state) {
    const labels = { loading: '正在连接', idle: '待运行', running: '运行中', success: '本次运行成功', error: '运行失败' };
    setText('status', labels[state.status]); byId('status').dataset.status = state.status;
    byId('run').disabled = !state.ready || state.status === 'running' || state.comparing;
    byId('run-all').disabled = !state.ready || state.comparing || Object.values(state.comparisons).some(row => row.status === 'running');
    byId('reveal').disabled = !state.ready || state.outcomeStatus === 'loading';
    const selection = state.selection;
    for (const [control, key] of [['case-select', 'example_id'], ['stage-select', 'stage'], ['algorithm-select', 'algorithm']]) {
      byId(control).value = selection[key];
    }
    setText('current-combination', `案例 ${selection.example_id} · ${stageLabel(selection.stage)} · ${algorithmLabel(selection.algorithm)}`);
    for (const button of document.querySelectorAll('[data-stage]')) {
      const active = button.dataset.stage === selection.stage;
      button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    }
    const f = state.detail?.features;
    setText('map-name', f ? f.map_name.replace('de_', '').toUpperCase() : '—');
    setText('round-num', f ? `ROUND ${String(f.round_num).padStart(2, '0')} / 案例 ${selection.example_id}` : '等待回合数据');
    setText('score', f ? `CT ${f.ct_score} : ${f.t_score} T` : '—');
    if (f) economy(f); else byId('economy-chart').replaceChildren();
    const post = Boolean(f && selection.stage === 'post_first_kill');
    byId('first-kill-panel').hidden = !post;
    setText('first-kill-side', post ? (f.first_kill_advantage_ct === 1 ? 'CT' : 'T') : '—');
    setText('first-kill-time', post ? `${f.first_kill_time.toFixed(2)} 秒` : '—');
    setText('first-kill-weapon', post ? f.first_kill_weapon : '—');
    setText('first-kill-headshot', post ? (f.first_kill_headshot ? '是' : '否') : '—');
    const result = state.result; const p = result?.prediction;
    setText('ct-probability', p ? percent(p.ct_win_probability) : '—');
    setText('t-probability', p ? percent(p.t_win_probability) : '—');
    byId('ct-fill').style.width = p ? `${p.ct_win_probability * 100}%` : '0%';
    byId('t-fill').style.width = p ? `${p.t_win_probability * 100}%` : '0%';
    byId('probability-bar').classList.toggle('empty', !p);
    byId('probability-bar').setAttribute('aria-label', p ? `CT ${percent(p.ct_win_probability)}，T ${percent(p.t_win_probability)}` : '当前没有成功预测');
    setText('prediction-note', state.error || (p ? `当前更倾向 ${p.predicted_side} 获胜。这是模型估计，不是确定结果。` : state.status === 'running' ? '正在校验本地文件并执行模型，旧结果已清除。' : '点击运行，使用已保存的真实模型进行本次预测。'));
    setText('threshold', p ? percent(p.decision_threshold) : '—');
    setText('calibration', result ? result.calibration_method : '—');
    setText('elapsed', result ? `${result.inference_ms.toFixed(1)} ms` : '—');
    setText('model-version', result ? result.model_version : '—');
    setText('validation', result ? `通过 · ${result.validation.required_input_field_count} 项输入 / ${result.validation.encoded_model_feature_count} 编码特征` : '—');
    setText('model-hash', result ? `${result.model_sha256.slice(0, 16)}…` : '—');
    setText('request-id', result ? result.request_id : '—');
    setText('source-detail', result ? JSON.stringify({ identity: result.identity, model_sha256: result.model_sha256, calibrator_sha256: result.calibrator_sha256, feature_version: result.feature_version, source: result.source }, null, 2) : '尚未运行');
    setText('outcome', state.outcome ? `${state.outcome.winning_side} 获胜 · 真实历史赛果` : state.outcomeStatus === 'error' ? '无法获取赛果，请确认本地服务已启动后重试。' : state.outcomeStatus === 'loading' ? '正在读取赛果…' : '暂不揭示。先观察输入，再比较预测与实际结果。');
    renderComparisons(state);
    window.roundcastChat?.setModelState(state);
  }
  function renderComparisons(state) {
    const body = byId('comparison-body'); body.replaceChildren();
    const names = { idle: '未运行', running: '运行中', success: '成功', error: '失败' };
    let completed = 0;
    for (const [stage, algorithm] of MODEL_PAIRS) {
      const key = modelKey(stage, algorithm), row = state.comparisons[key], result = row.result;
      const tr = document.createElement('tr'); tr.dataset.model = key;
      tr.className = 'comparison-row';
      tr.classList.toggle('selected', stage === state.selection.stage && algorithm === state.selection.algorithm);
      for (const text of [stageLabel(stage), algorithmLabel(algorithm), result ? percent(result.prediction.ct_win_probability) : '—',
        result ? percent(result.prediction.t_win_probability) : '—']) {
        const td = document.createElement('td'); td.textContent = text; tr.append(td);
      }
      const status = document.createElement('td'); const badge = document.createElement('span');
      badge.className = 'status'; badge.dataset.status = row.status; badge.textContent = names[row.status]; status.append(badge);
      if (row.error) { const error = document.createElement('small'); error.className = 'comparison-error'; error.textContent = row.error; status.append(error); }
      if (result) { const source = document.createElement('small'); source.className = 'comparison-request'; source.textContent = result.request_id.slice(0, 8); status.append(source); completed++; }
      const actions = document.createElement('td');
      const select = document.createElement('button'); select.type = 'button'; select.className = 'text-button'; select.textContent = '查看';
      select.addEventListener('click', () => controller.select({ stage, algorithm }));
      const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'text-button row-run';
      retry.textContent = row.status === 'error' ? '重试' : result ? '重跑' : '运行';
      retry.disabled = !state.ready || state.comparing || row.status === 'running';
      retry.addEventListener('click', () => controller.runPair(stage, algorithm));
      actions.append(select, retry); tr.append(status, actions); body.append(tr);
    }
    setText('comparison-status', state.comparing ? `正在对比 · ${completed}/4 成功` : `${completed}/4 成功`);
    for (const [stage, id] of [['pre_round', 'pre-difference'], ['post_first_kill', 'post-difference']]) {
      const xgb = state.comparisons[modelKey(stage, 'xgboost')].result, lgb = state.comparisons[modelKey(stage, 'lightgbm')].result;
      const delta = xgb && lgb ? (lgb.prediction.ct_win_probability - xgb.prediction.ct_win_probability) * 100 : null;
      setText(id, delta === null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} 个百分点`);
    }
  }
  const controller = new RoundcastController(api, render);
  byId('run').addEventListener('click', () => controller.run());
  byId('reveal').addEventListener('click', () => controller.reveal());
  byId('reload').addEventListener('click', () => controller.load());
  byId('run-all').addEventListener('click', () => controller.runAll());
  for (const [control, key] of [['case-select', 'example_id'], ['stage-select', 'stage'], ['algorithm-select', 'algorithm']]) {
    byId(control).addEventListener('change', event => controller.select({ [key]: event.target.value }));
  }
  for (const button of document.querySelectorAll('[data-stage]')) button.addEventListener('click', () => controller.select({ stage: button.dataset.stage }));
  controller.load();
  const unregisterTool = registerRoundcastTool(document.modelContext, controller);
  window.addEventListener('pagehide', unregisterTool, { once: true });
}
