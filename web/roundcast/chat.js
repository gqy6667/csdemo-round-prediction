'use strict';

const CHAT_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function getChatContext(state) {
  const selection = state?.selection || {};
  const result = state?.status === 'success' ? state.result : null;
  const matches = result && Object.keys(selection).every(key => result[key] === selection[key]);
  const request_id = matches && CHAT_UUID.test(result.request_id) ? result.request_id : null;
  const features = state?.detail?.features;
  const stage = selection.stage === 'post_first_kill' ? '首杀后' : '购买结束';
  const algorithm = selection.algorithm === 'lightgbm' ? 'LightGBM' : 'XGBoost';
  const round = features ? `${String(features.map_name).replace('de_', '').toUpperCase()} R${features.round_num}` : '等待回合数据';
  return { example_id: selection.example_id, stage: selection.stage, algorithm: selection.algorithm, request_id,
    ready: Boolean(state?.ready && features && !['loading', 'running'].includes(state.status)),
    label: `案例 ${selection.example_id || '—'} · ${round} · ${stage} · ${algorithm} · ${request_id ? `预测 ${request_id.slice(0, 8)}` : '未附预测'}` };
}
function boundedHistory(history) {
  const pairs = [];
  for (let i = 0; i + 1 < history.length; i += 2) {
    const [user, assistant] = history.slice(i, i + 2);
    if (user.role !== 'user' || assistant.role !== 'assistant' || typeof user.text !== 'string' || typeof assistant.text !== 'string') continue;
    pairs.push([{ role: 'user', text: user.text }, { role: 'assistant', text: assistant.text }]);
  }
  let result = pairs.slice(-6).flat();
  while (result.reduce((total, entry) => total + entry.text.length, 0) > 24000) result = result.slice(2);
  return result;
}

class RoundcastChatController {
  constructor(api, render = () => {}, pause = ms => new Promise(resolve => setTimeout(resolve, ms))) {
    this.api = api; this.render = render; this.pause = pause; this.active = null;
    this.history = []; this.historyUpdatedAt = 0; this.statusGeneration = 0;
    this.state = { available: false, checking: false, connection: '尚未检查本地 Codex', phase: 'idle',
      context: getChatContext(null), messages: [], error: '' };
  }
  notify() { this.render(this.state); }
  setModelState(state) {
    const context = getChatContext(state), previous = this.state.context;
    if (context.example_id !== previous.example_id || context.stage !== previous.stage) {
      this.history = []; this.historyUpdatedAt = 0;
    }
    this.state.context = context; this.notify();
  }
  async checkStatus() {
    const generation = ++this.statusGeneration;
    this.state.checking = true; this.notify();
    try {
      const status = await this.api('/api/chat/status');
      if (generation !== this.statusGeneration) return;
      this.state.available = status.available === true;
      this.state.connection = typeof status.message === 'string' ? status.message : '本地 Codex 状态不可用';
    } catch {
      if (generation !== this.statusGeneration) return;
      this.state.available = false; this.state.connection = '无法检查本地 Codex，请确认本地服务已启动。';
    }
    this.state.checking = false; this.notify();
  }
  async send(input) {
    if (this.active || !this.state.available || !this.state.context.ready) return false;
    const message = typeof input === 'string' ? input.trim() : '';
    if (!message || message.length > 2000) {
      this.state.error = '请输入 1–2000 个字符的问题。'; this.notify(); return false;
    }
    if (Date.now() - this.historyUpdatedAt > 30 * 60 * 1000) this.history = [];
    const context = { ...this.state.context };
    const active = { context, message, jobId: null, cancelRequested: false, cancelPending: false };
    this.active = active;
    Object.assign(this.state, { phase: 'starting', error: '' });
    this.state.messages.push({ role: 'user', text: message, contextLabel: context.label });
    this.state.messages = this.state.messages.slice(-48); this.notify();
    try {
      const result = await this.api('/api/chat', { message, example_id: context.example_id,
        stage: context.stage, algorithm: context.algorithm, request_id: context.request_id,
        history: boundedHistory(this.history) });
      if (this.active !== active) return false;
      if (!CHAT_UUID.test(result.job_id) || result.status !== 'running') throw Error('对话请求未能建立，请重新检查连接。');
      active.jobId = result.job_id;
      if (active.cancelRequested) { await this.cancel(); return true; }
      this.state.phase = 'running'; this.notify();
      await this.poll(active);
    } catch (error) {
      if (this.active === active) this.finish(active, null, error.message || '发送失败，请重试。');
    }
    return true;
  }
  async poll(active) {
    while (this.active === active && !active.cancelRequested) {
      try {
        const result = await this.api(`/api/chat/jobs/${active.jobId}`);
        if (this.active !== active || active.cancelRequested) return;
        if (result.job_id !== active.jobId) throw Error('回复标识不匹配，请停止本次对话后重试。');
        if (result.status === 'success') {
          if (typeof result.reply !== 'string' || !result.reply.trim() || result.reply.length > 16000) throw Error('回复内容不可用，请停止本次对话后重试。');
          this.finish(active, result.reply); return;
        }
        if (['error', 'cancelled'].includes(result.status)) {
          this.finish(active, null, result.message || (result.status === 'cancelled' ? '本次回复已停止。' : 'Codex 暂时无法回答，请重试。')); return;
        }
        if (result.status !== 'running') throw Error('对话状态不可用，请停止后重试。');
      } catch (error) {
        if (this.active !== active || active.cancelRequested) return;
        if (error.status === 404) { this.finish(active, null, '请求已失效，无法恢复此回复，可重新发送。'); return; }
        this.state.phase = 'poll_error'; this.state.error = error.message || '无法获取回复，请停止后重试。'; this.notify(); return;
      }
      await this.pause(1200);
    }
  }
  finish(active, reply, error = '') {
    if (this.active !== active) return;
    if (reply) {
      this.state.messages.push({ role: 'assistant', text: reply, contextLabel: active.context.label });
      const contextNote = `[上下文：${active.context.label}]\n`;
      if (active.context.example_id === this.state.context.example_id && active.context.stage === this.state.context.stage) {
        this.history = boundedHistory([...this.history, { role: 'user', text: contextNote + active.message }, { role: 'assistant', text: reply }]);
        this.historyUpdatedAt = Date.now();
      }
    }
    this.active = null; this.state.phase = 'idle'; this.state.error = error; this.notify();
  }
  async cancel() {
    const active = this.active;
    if (!active || active.cancelPending) return;
    active.cancelRequested = true; this.state.phase = 'cancelling'; this.state.error = ''; this.notify();
    if (!active.jobId) return;
    active.cancelPending = true;
    try {
      const result = await this.api('/api/chat/cancel', { job_id: active.jobId });
      if (result.job_id !== active.jobId || !['cancelled', 'success', 'error'].includes(result.status)) throw Error('无法确认停止状态，请重试停止。');
      this.finish(active, null, '本次回复已停止。');
    } catch (error) {
      if (this.active !== active) return;
      if (error.status === 404) { this.finish(active, null, '请求已失效，无法恢复此回复，可重新发送。'); return; }
      this.state.phase = 'poll_error'; this.state.error = '未能确认停止。请重试“停止回复”，暂不发送新问题。';
      this.notify();
    } finally { active.cancelPending = false; }
  }
  clear() {
    if (this.active) return false;
    this.history = []; this.historyUpdatedAt = 0;
    this.state.messages = []; this.state.error = ''; this.notify(); return true;
  }
}

function shouldSendOnShortcut(event) {
  return Boolean(event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing && event.keyCode !== 229);
}
if (typeof module !== 'undefined') module.exports = { RoundcastChatController, getChatContext, boundedHistory, shouldSendOnShortcut };

if (typeof document !== 'undefined') {
  const element = id => document.getElementById(id);
  const input = element('chat-input');
  const transcript = element('chat-messages');
  let renderedMessages = '';
  const api = async (url, body) => {
    let response, data;
    try {
      response = await fetch(url, { method: body ? 'POST' : 'GET', cache: 'no-store',
        headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(15000) });
      data = await response.json();
    } catch { throw Error('无法连接本地对话服务，请检查服务后重试。'); }
    if (!response.ok) throw Object.assign(Error(data.message || '对话请求失败，请重试。'), { status: response.status });
    return data;
  };
  function render(state) {
    const busy = state.phase !== 'idle';
    const phases = { idle: '', starting: '正在建立对话…', running: 'Codex 正在思考…', cancelling: '正在停止回复…', poll_error: '回复连接中断，请停止后重试。' };
    element('chat-status').textContent = state.checking ? '正在检查本地连接…' : state.connection;
    element('chat-status').dataset.status = state.available ? 'success' : 'error';
    element('chat-reconnect').disabled = state.checking;
    element('chat-context').textContent = state.context.label;
    element('chat-progress').textContent = phases[state.phase];
    element('chat-error').textContent = state.error;
    element('chat-clear').disabled = busy || state.messages.length === 0;
    element('chat-cancel').hidden = !busy;
    element('chat-cancel').disabled = state.phase === 'cancelling';
    element('chat-send').disabled = busy || !state.available || !state.context.ready || !input.value.trim() || input.value.trim().length > 2000;
    element('chat-send').textContent = busy ? '回复中' : '发送给 Codex';
    element('chat-counter').textContent = `${input.value.length} / 2000`;
    element('chat-empty').hidden = state.messages.length > 0;
    const signature = JSON.stringify(state.messages);
    if (signature !== renderedMessages) {
      renderedMessages = signature; transcript.replaceChildren();
      for (const message of state.messages) {
        const article = document.createElement('article'); article.className = `chat-message chat-${message.role}`;
        const header = document.createElement('div'); header.className = 'chat-message-header';
        const name = document.createElement('strong'); name.textContent = message.role === 'user' ? '你' : 'Codex';
        const context = document.createElement('span'); context.textContent = message.contextLabel;
        const content = document.createElement('p'); content.className = 'chat-message-content'; content.textContent = message.text;
        header.append(name, context); article.append(header, content); transcript.append(article);
      }
      transcript.scrollTop = transcript.scrollHeight;
    }
  }
  const controller = new RoundcastChatController(api, render);
  window.roundcastChat = controller;
  function send() {
    if (element('chat-send').disabled) return;
    const pending = controller.send(input.value);
    if (controller.active) { input.value = ''; controller.notify(); }
    void pending;
  }
  element('chat-form').addEventListener('submit', event => { event.preventDefault(); send(); });
  input.addEventListener('keydown', event => { if (shouldSendOnShortcut(event)) { event.preventDefault(); send(); } });
  input.addEventListener('input', () => controller.notify());
  for (const button of document.querySelectorAll('[data-chat-question]')) {
    button.addEventListener('click', () => { input.value = button.dataset.chatQuestion; input.focus(); controller.notify(); });
  }
  element('chat-reconnect').addEventListener('click', () => controller.checkStatus());
  element('chat-cancel').addEventListener('click', () => controller.cancel());
  element('chat-clear').addEventListener('click', () => { if (controller.clear()) { input.value = ''; controller.notify(); } });
  controller.checkStatus();
}
