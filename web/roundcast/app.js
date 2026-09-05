'use strict';

class RoundcastController {
  constructor(api, render = () => {}) {
    this.api = api;
    this.render = render;
    this.generation = 0;
    this.outcomeGeneration = 0;
    this.state = { selection: { example_id: 'A', stage: 'pre_round', algorithm: 'xgboost' },
      status: 'loading', ready: false, result: null, detail: null, outcome: null, outcomeStatus: 'idle', error: '' };
  }
  notify() { this.render(this.state); }
  async load() {
    const token = ++this.generation;
    ++this.outcomeGeneration;
    Object.assign(this.state, { status: 'loading', ready: false, result: null, detail: null, outcome: null, outcomeStatus: 'idle', error: '' });
    this.notify();
    try {
      const [models, examples, detail] = await Promise.all([
        this.api('/api/models'), this.api('/api/examples'), this.api('/api/examples/A')]);
      if (token !== this.generation) return;
      if (!models.models.some(m => m.model_id === 'xgb_pre_round' && m.inference_ready && m.available_examples.includes('A'))
          || !examples.examples.some(e => e.example_id === 'A' && e.inference_ready)
          || detail.example_id !== 'A' || detail.stage !== 'pre_round' || !detail.features) throw Error('当前组合尚未就绪');
      Object.assign(this.state, { detail, models: models.models, status: 'idle', ready: true });
    } catch (error) {
      if (token !== this.generation) return;
      Object.assign(this.state, { status: 'error', error: error.message });
    }
    this.notify();
  }
  async run() {
    if (!this.state.ready || this.state.status === 'running') return;
    const token = ++this.generation;
    const selection = { ...this.state.selection };
    Object.assign(this.state, { status: 'running', result: null, error: '' }); this.notify();
    try {
      const result = await this.api('/api/predict', selection);
      if (token !== this.generation) return;
      const p = result?.prediction;
      if (!result || result.status !== 'success' || result.model_id !== 'xgb_pre_round' || !result.request_id
          || Object.keys(selection).some(k => result[k] !== selection[k])
          || !p || ![p.ct_win_probability, p.t_win_probability].every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1)
          || Math.abs(p.ct_win_probability + p.t_win_probability - 1) > 1e-12) throw Error('返回结果与当前选择不一致，请重新运行');
      Object.assign(this.state, { status: 'success', result });
    } catch (error) {
      if (token !== this.generation) return;
      Object.assign(this.state, { status: 'error', result: null, error: error.message });
    }
    this.notify();
    return this.state.result;
  }
  async reveal() {
    if (!this.state.ready || this.state.outcomeStatus === 'loading') return;
    const token = ++this.outcomeGeneration;
    Object.assign(this.state, { outcome: null, outcomeStatus: 'loading' }); this.notify();
    try {
      const outcome = await this.api('/api/examples/A/outcome');
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
    byId('run').disabled = !state.ready || state.status === 'running';
    byId('reveal').disabled = !state.ready || state.outcomeStatus === 'loading';
    const f = state.detail?.features;
    setText('map-name', f ? f.map_name.replace('de_', '').toUpperCase() : '—');
    setText('round-num', f ? `ROUND ${String(f.round_num).padStart(2, '0')} / 案例 A` : '等待回合数据');
    setText('score', f ? `CT ${f.ct_score} : ${f.t_score} T` : '—');
    if (f) economy(f); else byId('economy-chart').replaceChildren();
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
    setText('validation', result ? `通过 · ${result.validation.required_base_feature_count} 项输入 / ${result.validation.encoded_model_feature_count} 编码特征` : '—');
    setText('model-hash', result ? `${result.model_sha256.slice(0, 16)}…` : '—');
    setText('request-id', result ? result.request_id : '—');
    setText('source-detail', result ? JSON.stringify({ identity: result.identity, model_sha256: result.model_sha256, calibrator_sha256: result.calibrator_sha256, feature_version: result.feature_version, source: result.source }, null, 2) : '尚未运行');
    setText('outcome', state.outcome ? `${state.outcome.winning_side} 获胜 · 真实历史赛果` : state.outcomeStatus === 'error' ? '无法获取赛果，请确认本地服务已启动后重试。' : state.outcomeStatus === 'loading' ? '正在读取赛果…' : '暂不揭示。先观察输入，再比较预测与实际结果。');
    window.roundcastChat?.setModelState(state);
  }
  const controller = new RoundcastController(api, render);
  byId('run').addEventListener('click', () => controller.run());
  byId('reveal').addEventListener('click', () => controller.reveal());
  byId('reload').addEventListener('click', () => controller.load());
  controller.load();
  const unregisterTool = registerRoundcastTool(document.modelContext, controller);
  window.addEventListener('pagehide', unregisterTool, { once: true });
}
