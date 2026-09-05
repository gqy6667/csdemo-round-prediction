'use strict';

const MODEL_PAIRS = [['pre_round', 'xgboost'], ['pre_round', 'lightgbm'], ['post_first_kill', 'xgboost'], ['post_first_kill', 'lightgbm']];
const modelKey = (stage, algorithm) => `${algorithm === 'xgboost' ? 'xgb' : 'lgbm'}_${stage}`;
const emptyComparisons = () => Object.fromEntries(MODEL_PAIRS.map(([stage, algorithm]) =>
  [modelKey(stage, algorithm), { stage, algorithm, status: 'idle', result: null, error: '' }]));
const DEFAULT_ROUTE = { view: 'viewer', example_id: 'A', stage: 'pre_round', algorithm: 'xgboost' };
function parseRoute(hash = '') {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const route = { view: params.get('view') || 'viewer', example_id: params.get('case') || 'A',
    stage: params.get('stage') || 'pre_round', algorithm: params.get('algorithm') || 'xgboost' };
  return ['viewer', 'analyst', 'technical'].includes(route.view) && ['A', 'B', 'C'].includes(route.example_id)
    && MODEL_PAIRS.some(([stage, algorithm]) => stage === route.stage && algorithm === route.algorithm) ? route : { ...DEFAULT_ROUTE };
}
function routeHash(route) {
  const selection = route.selection || route;
  return '#' + new URLSearchParams({ view: route.view, case: selection.example_id, stage: selection.stage, algorithm: selection.algorithm });
}
const INPUT_FIELDS = [
  ['map_name', '地图', '回合', '名称'], ['round_num', '回合序号', '回合', '回合'],
  ['ct_score', 'CT 比分', '回合', '分'], ['t_score', 'T 比分', '回合', '分'],
  ...[['eq_value', '装备价值', '游戏币'], ['cash', '剩余现金', '游戏币'], ['armor', '有护甲人数', '人'],
    ['helmets', '有头盔人数', '人'], ['grenades', '投掷物库存条目', '条目'], ['ak47', 'AK-47 持有人数', '人'],
    ['m4a4', 'M4A4 持有人数', '人'], ['m4a1_s', 'M4A1-S 持有人数', '人'], ['awp', 'AWP 持有人数', '人'],
    ['rifles', '步枪持有人数', '人'], ['smgs', '冲锋枪持有人数', '人']].flatMap(([key, label, unit]) =>
      ['ct', 't'].map(side => [`${side}_${key}`, `${side.toUpperCase()} ${label}`, '购买结束', unit])),
  ['ct_defuse_kits', 'CT 携带拆弹器人数', '购买结束', '人'],
  ['first_kill_advantage_ct', '首杀阵营编码', '首杀', '+1 = CT / −1 = T'],
  ['first_kill_time', '首杀时间（源事件）', '首杀', '秒'],
  ['first_kill_headshot', '首杀是否爆头', '首杀', '1 = 是 / 0 = 否'],
  ['first_kill_weapon', '首杀武器', '首杀', '名称']
];
function analysisData(state) {
  const probabilities = MODEL_PAIRS.map(([stage, algorithm]) => {
    const row = state.comparisons[modelKey(stage, algorithm)], r = row?.result;
    const valid = row?.status === 'success' && r?.example_id === state.selection.example_id
      && r.stage === stage && r.algorithm === algorithm && r.model_id === modelKey(stage, algorithm);
    return { stage, algorithm, probability: valid ? r.prediction.ct_win_probability : null, status: row?.status || 'idle' };
  });
  const changes = Object.fromEntries(['xgboost', 'lightgbm'].map(algorithm => {
    const values = probabilities.filter(r => r.algorithm === algorithm).map(r => r.probability);
    return [algorithm, values.every(v => v !== null) ? (values[1] - values[0]) * 100 : null];
  }));
  const f = state.ready && validSnapshot(state.detail, state.selection) ? state.detail.features : null;
  const fields = f ? INPUT_FIELDS.filter(([key]) => state.selection.stage === 'post_first_kill' || !key.startsWith('first_kill_'))
    .map(([key, label, group, unit]) => ({ key, label, group, unit, value: f[key] ?? null })) : [];
  const economy = f ? [['eq_value', '装备价值'], ['cash', '剩余现金']].map(([key, label]) =>
    ({ key, label, ct: f[`ct_${key}`], t: f[`t_${key}`], difference: f[`ct_${key}`] - f[`t_${key}`] })) : [];
  return { probabilities, changes, fields, economy };
}
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

const METRIC_FIELDS = [
  ['accuracy', 'Accuracy', '准确率 · 越高越好'], ['auc', 'AUC', '区分能力 · 越高越好'],
  ['log_loss', 'Log Loss', '概率损失 · 越低越好'], ['brier_score', 'Brier', '概率平方误差 · 越低越好'],
  ['ece10', 'ECE10', '10 分箱校准误差 · 越低越好']
];
function technicalData(state) {
  const s = state.selection, key = modelKey(s.stage, s.algorithm);
  const rows = state.metrics?.models?.filter(row => row?.model_id === key) || [];
  const row = rows.length === 1 ? rows[0] : null;
  const validMetrics = ['success', 'partial'].includes(state.metricsStatus) && row?.status === 'success'
    && row.stage === s.stage && row.algorithm === s.algorithm && row.scope === 'frozen_test_set'
    && row.split === 'test' && row.split_unit === 'series_id'
    && row.n_test === (s.stage === 'pre_round' ? 4172 : 4170)
    && typeof row.source?.path === 'string' && /^[a-f0-9]{64}$/.test(row.source?.sha256)
    && row.metrics && Object.keys(row.metrics).length === METRIC_FIELDS.length
    && METRIC_FIELDS.every(([name]) => typeof row.metrics[name] === 'number' && Number.isFinite(row.metrics[name])
      && row.metrics[name] >= 0 && (name === 'log_loss' || row.metrics[name] <= 1));
  const r = state.result;
  const prediction = state.ready && state.status === 'success' && r?.status === 'success' && r.model_id === key
    && r.example_id === s.example_id && r.stage === s.stage && r.algorithm === s.algorithm ? r : null;
  const outcome = state.outcomeStatus === 'success' && state.outcome?.example_id === s.example_id
    && ['CT', 'T'].includes(state.outcome.winning_side) ? state.outcome : null;
  const correct = prediction && outcome && ['CT', 'T'].includes(prediction.prediction?.predicted_side)
    ? prediction.prediction.predicted_side === outcome.winning_side : null;
  return { metrics: validMetrics ? row : null, prediction, outcome, correct };
}

function validRuntime(result, selection) {
  const text = value => typeof value === 'string' && value.trim().length > 0;
  return text(result.request_id) && text(result.model_version) && text(result.feature_version)
    && [result.model_sha256, result.calibrator_sha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash))
    && typeof result.inference_ms === 'number' && Number.isFinite(result.inference_ms) && result.inference_ms >= 0
    && result.calibration_method === 'uncalibrated' && result.validation?.status === 'passed'
    && result.validation.required_input_field_count === (selection.stage === 'pre_round' ? 27 : 31)
    && result.validation.encoded_model_feature_count === (selection.stage === 'pre_round' ? 43 : 82)
    && result.prediction.decision_threshold === 0.5
    && result.prediction.predicted_side === (result.prediction.ct_win_probability >= 0.5 ? 'CT' : 'T')
    && ['series_id', 'game_id', 'round_id'].every(key => text(result.identity?.[key]))
    && result.source && typeof result.source === 'object' && !Array.isArray(result.source);
}

class RoundcastController {
  constructor(api, render = () => {}, route = DEFAULT_ROUTE) {
    this.api = api;
    this.render = render;
    this.generation = 0; this.caseGeneration = 0;
    this.outcomeGeneration = 0; this.metricsGeneration = 0;
    const initial = parseRoute(routeHash(route));
    this.state = { view: initial.view, selection: { example_id: initial.example_id, stage: initial.stage, algorithm: initial.algorithm },
      status: 'loading', ready: false, result: null, detail: null, outcome: null, outcomeStatus: 'idle', error: '',
      models: [], examples: [], comparisons: emptyComparisons(), comparing: false, detailError: '',
      metrics: null, metricsStatus: 'idle', metricsError: '' };
  }
  notify() { this.render(this.state); }
  async navigate(change) {
    const view = change.view ?? this.state.view;
    const selection = Object.fromEntries(['example_id', 'stage', 'algorithm'].map(k => [k, change[k] ?? this.state.selection[k]]));
    const changed = Object.keys(selection).some(k => selection[k] !== this.state.selection[k]);
    if (!['viewer', 'analyst', 'technical'].includes(view) || (changed && !this.valid(selection))) return false;
    this.state.view = view;
    if (changed) return this.select(selection);
    this.notify();
    return true;
  }
  valid(selection) {
    return MODEL_PAIRS.some(([stage, algorithm]) => stage === selection.stage && algorithm === selection.algorithm)
      && this.state.examples.some(e => e?.example_id === selection.example_id && e.inference_ready === true)
      && this.state.models.some(m => m?.stage === selection.stage && m.algorithm === selection.algorithm
        && m.model_id === modelKey(selection.stage, selection.algorithm) && m.inference_ready === true
        && Array.isArray(m.available_examples) && m.available_examples.includes(selection.example_id));
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
      models: [], examples: [], comparisons: emptyComparisons(), comparing: false, detailError: '' });
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
  async loadMetrics() {
    const token = ++this.metricsGeneration;
    Object.assign(this.state, { metrics: null, metricsStatus: 'loading', metricsError: '' }); this.notify();
    try {
      const report = await this.api('/api/metrics');
      if (token !== this.metricsGeneration) return;
      if (!['success', 'partial', 'unavailable'].includes(report?.status) || !Array.isArray(report.models))
        throw Error('正式指标响应无效，请重试');
      Object.assign(this.state, { metrics: report, metricsStatus: report.status });
    } catch (error) {
      if (token !== this.metricsGeneration) return;
      Object.assign(this.state, { metrics: null, metricsStatus: 'error', metricsError: error.message });
    }
    this.notify();
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
          || Math.abs(p.ct_win_probability + p.t_win_probability - 1) > 1e-12
          || !validRuntime(result, selection)) throw Error('返回结果与当前选择不一致或运行记录不完整，请重新运行');
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
if (typeof module !== 'undefined') module.exports = { RoundcastController, registerRoundcastTool, parseRoute, routeHash, analysisData, technicalData, INPUT_FIELDS };

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
    const hash = routeHash(state);
    if (window.location.hash !== hash) window.history.pushState(null, '', hash);
    for (const view of ['viewer', 'analyst', 'technical']) byId(`${view}-view`).hidden = state.view !== view;
    for (const link of document.querySelectorAll('[data-view]')) {
      link.href = routeHash({ ...state, view: link.dataset.view });
      if (link.dataset.view === state.view) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    }
    const titles = {
      viewer: ['回合观察', 'ROUND OBSERVER', '读懂一个回合。', '选择真实历史回合，比较购买结束与首杀后的模型估计。'],
      analyst: ['比赛分析', 'ROUND ANALYSIS', '比较这个回合。', '同一回合的两时点估计、购买结束经济与原始输入。'],
      technical: ['技术分析', 'MODEL EVIDENCE', '让预测有据可查。', '区分整个测试集的表现，与当前回合的一次模型运行。']
    }[state.view];
    document.title = `ROUNDCAST · ${titles[0]}`;
    setText('view-eyebrow', titles[1]); setText('view-title', titles[2]); setText('view-description', titles[3]);
    const labels = { loading: '正在连接', idle: '待运行', running: '运行中', success: '本次运行成功', error: '运行失败' };
    setText('status', labels[state.status]); byId('status').dataset.status = state.status;
    byId('run').disabled = !state.ready || state.status === 'running' || state.comparing;
    byId('run-all').disabled = !state.ready || state.comparing || Object.values(state.comparisons).some(row => row.status === 'running');
    byId('reveal').disabled = !state.ready || state.outcomeStatus === 'loading';
    const selection = state.selection;
    for (const [control, key] of [['case-select', 'example_id'], ['stage-select', 'stage'], ['algorithm-select', 'algorithm']]) {
      byId(control).value = selection[key];
      byId(control).disabled = !controller.valid(selection);
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
    renderAnalysis(state);
    renderTechnical(state);
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
  function renderAnalysis(state) {
    setText('analysis-selection', byId('current-combination').textContent);
    setText('analysis-status', byId('status').textContent); byId('analysis-status').dataset.status = state.status;
    setText('analysis-message', state.detailError || '点击“运行全部对比”填入四项真实预测；未成功的组合不画点。');
    setText('analysis-outcome', byId('outcome').textContent);
    byId('analysis-reveal').disabled = byId('reveal').disabled;
    const data = analysisData(state), svg = byId('analysis-chart'); svg.replaceChildren();
    const element = (tag, attributes, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
      if (text !== undefined) node.textContent = text;
      svg.append(node); return node;
    };
    element('title', {}, '当前回合 CT 胜率：四个离散预测组合');
    element('desc', {}, '横轴为 CT 胜率 0 至 100%，圆点为 XGBoost，方点为 LightGBM。空缺不代表零。两时点不连接为连续曲线。');
    for (const tick of [0, 25, 50, 75, 100]) {
      const x = 200 + tick * 4;
      element('line', { x1: x, x2: x, y1: 40, y2: 225, stroke: '#e3e6eb', 'stroke-dasharray': tick === 50 ? '3 4' : '0' });
      element('text', { x, y: 24, 'text-anchor': 'middle' }, `${tick}%`);
    }
    element('text', { x: 6, y: 90 }, '购买结束'); element('text', { x: 6, y: 190 }, '首杀后');
    data.probabilities.forEach((row, i) => {
      const y = [65, 105, 165, 205][i], key = modelKey(row.stage, row.algorithm);
      element('text', { x: 94, y: y + 5 }, algorithmLabel(row.algorithm));
      element('text', { x: 710, y: y + 5, 'text-anchor': 'end' }, row.probability === null ? '—' : percent(row.probability));
      if (row.probability === null) return;
      const x = 200 + row.probability * 400;
      if (row.stage === state.selection.stage && row.algorithm === state.selection.algorithm)
        element('circle', { cx: x, cy: y, r: 11, class: 'chart-selected' });
      const point = element(row.algorithm === 'xgboost' ? 'circle' : 'rect',
        row.algorithm === 'xgboost' ? { cx: x, cy: y, r: 6, class: 'chart-point' }
          : { x: x - 6, y: y - 6, width: 12, height: 12, class: 'chart-point lightgbm' });
      point.dataset.model = key; point.dataset.probability = String(row.probability);
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${stageLabel(row.stage)} · ${algorithmLabel(row.algorithm)} · CT ${percent(row.probability)}`; point.append(title);
    });
    for (const [algorithm, id] of [['xgboost', 'analysis-xgb-change'], ['lightgbm', 'analysis-lgb-change']]) {
      const value = data.changes[algorithm];
      setText(id, value === null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)} 个百分点`);
    }
    const economy = byId('analysis-economy'); economy.replaceChildren();
    for (const row of data.economy) {
      const item = document.createElement('div'); item.className = 'analysis-economy-item'; item.dataset.metric = row.key;
      const label = document.createElement('h3'); label.textContent = row.label;
      const difference = document.createElement('strong'); difference.dataset.difference = row.difference;
      difference.textContent = `${row.difference >= 0 ? '+' : '−'}$${Math.abs(row.difference).toLocaleString('en-US')}`;
      item.append(label, difference);
      const max = Math.max(row.ct, row.t, 1);
      for (const side of ['ct', 't']) {
        const line = document.createElement('div'); line.className = 'economy-line';
        const name = document.createElement('span'); name.textContent = side.toUpperCase();
        const track = document.createElement('div'); track.className = 'track';
        const fill = document.createElement('i'); fill.style.width = `${row[side] / max * 100}%`; track.append(fill);
        const value = document.createElement('strong'); value.textContent = `$${row[side].toLocaleString('en-US')}`;
        line.append(name, track, value); item.append(line);
      }
      economy.append(item);
    }
    if (!data.economy.length) economy.textContent = state.detailError || '等待当前回合快照';
    const body = byId('analysis-snapshot'); body.replaceChildren(); let group;
    for (const field of data.fields) {
      const tr = document.createElement('tr'); tr.dataset.field = field.key;
      if (group !== field.group) { tr.className = 'snapshot-group-start'; group = field.group; }
      const value = field.value === null ? '未提供' : typeof field.value === 'boolean' ? String(Number(field.value)) : String(field.value);
      for (const text of [field.group, field.label, value, field.unit, field.key]) {
        const td = document.createElement('td'); td.textContent = text; tr.append(td);
      }
      body.append(tr);
    }
    const missing = data.fields.filter(f => f.value === null).length;
    setText('analysis-field-count', data.fields.length ? `${data.fields.length} 项输入${missing ? ` · ${missing} 项未提供` : ''}` : '等待快照');
  }
  function renderTechnical(state) {
    const { metrics, prediction: r, outcome, correct } = technicalData(state);
    const metricLoading = state.metricsStatus === 'loading';
    setText('metrics-selection', `${stageLabel(state.selection.stage)} · ${algorithmLabel(state.selection.algorithm)}`);
    setText('metrics-status', metrics ? '来源已校验' : metricLoading ? '读取中' : '指标不可用');
    byId('metrics-status').dataset.status = metrics ? 'success' : metricLoading ? 'loading' : 'error';
    byId('metrics-reload').disabled = metricLoading;
    setText('metrics-scope', metrics ? `${metrics.n_test.toLocaleString('en-US')} 个测试回合 · 按 series_id 划分` : '整个冻结测试集 · 当前来源尚不可用');
    const grid = byId('technical-metrics'); grid.replaceChildren();
    for (const [name, label, meaning] of METRIC_FIELDS) {
      const item = document.createElement('div'); item.className = 'metric-item'; item.dataset.metric = name;
      const dt = document.createElement('dt'); dt.textContent = label;
      const dd = document.createElement('dd'); dd.dataset.value = metrics ? String(metrics.metrics[name]) : '';
      dd.textContent = metrics ? name === 'accuracy' ? percent(metrics.metrics[name]) : metrics.metrics[name].toFixed(6) : '—';
      const note = document.createElement('small'); note.textContent = meaning;
      item.append(dt, dd, note); grid.append(item);
    }
    setText('metrics-message', metrics ? '读取正式报告中的固定结果，不重新拟合或用当前案例计算。' :
      metricLoading ? '正在校验指标文件与测试样本来源…' : (state.metricsError || '所选模型的正式指标缺失、校验失败或响应无效。预测功能可独立使用。'));
    setText('metrics-source', metrics ? JSON.stringify({ model_id: metrics.model_id, model_version: metrics.model_version,
      scope: metrics.scope, n_test: metrics.n_test, source: metrics.source, sample_source: metrics.sample_source }, null, 2) : '尚无通过校验的指标来源');
    setText('technical-selection', byId('current-combination').textContent);
    setText('technical-status', byId('status').textContent); byId('technical-status').dataset.status = state.status;
    setText('technical-message', state.error || (r ? `CT ${percent(r.prediction.ct_win_probability)} · T ${percent(r.prediction.t_win_probability)}` : '点击“运行当前组合”生成本次记录；正式指标不代表已执行预测。'));
    setText('technical-version', r ? r.model_version : '—');
    setText('technical-validation', r ? `通过 · ${r.validation.required_input_field_count} 项输入 / ${r.validation.encoded_model_feature_count} 编码特征` : '—');
    setText('technical-elapsed', r ? `${r.inference_ms.toFixed(1)} ms` : '—');
    setText('technical-calibration', r ? r.calibration_method === 'uncalibrated' ? 'uncalibrated · 未额外校准' : r.calibration_method : '—');
    setText('technical-source', r ? JSON.stringify({ model_id: r.model_id, model_sha256: r.model_sha256,
      calibrator_sha256: r.calibrator_sha256, feature_version: r.feature_version, request_id: r.request_id }, null, 2) : '尚无当前组合的成功运行记录');
    setText('technical-correctness', correct === null ? '尚未评判' : correct ? '本次预测正确' : '本次预测错误');
    byId('technical-correctness').dataset.correct = correct === null ? '' : String(correct);
    setText('technical-outcome-text', correct !== null ? `预测 ${r.prediction.predicted_side} · 实际 ${outcome.winning_side} 获胜 · 阈值 ${percent(r.prediction.decision_threshold)}` :
      outcome ? `实际 ${outcome.winning_side} 获胜；等待当前组合成功预测。` : byId('outcome').textContent);
    byId('technical-reveal').disabled = byId('reveal').disabled;
  }
  const controller = new RoundcastController(api, render, parseRoute(window.location.hash));
  window.history.replaceState(null, '', routeHash(controller.state));
  const followLocation = async () => {
    const route = parseRoute(window.location.hash);
    window.history.replaceState(null, '', routeHash(route));
    if (!await controller.navigate(route)) window.history.replaceState(null, '', routeHash(controller.state));
  };
  window.addEventListener('popstate', followLocation);
  window.addEventListener('hashchange', followLocation);
  byId('run').addEventListener('click', () => controller.run());
  byId('reveal').addEventListener('click', () => controller.reveal());
  byId('analysis-reveal').addEventListener('click', () => controller.reveal());
  byId('technical-reveal').addEventListener('click', () => controller.reveal());
  byId('metrics-reload').addEventListener('click', () => controller.loadMetrics());
  byId('reload').addEventListener('click', () => controller.load());
  byId('run-all').addEventListener('click', () => controller.runAll());
  for (const [control, key] of [['case-select', 'example_id'], ['stage-select', 'stage'], ['algorithm-select', 'algorithm']]) {
    byId(control).addEventListener('change', event => controller.select({ [key]: event.target.value }));
  }
  for (const button of document.querySelectorAll('[data-stage]')) button.addEventListener('click', () => controller.select({ stage: button.dataset.stage }));
  controller.load();
  controller.loadMetrics();
  const unregisterTool = registerRoundcastTool(document.modelContext, controller);
  window.addEventListener('pagehide', unregisterTool, { once: true });
}
