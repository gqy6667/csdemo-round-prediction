const test = require('node:test');
const assert = require('node:assert/strict');
const { RoundcastChatController, getChatContext, boundedHistory, shouldSendOnShortcut } = require('../web/roundcast/chat.js');
const job = 'd920f73d-71c2-43af-96d6-665a701b16ce';
const request = 'adc83880-34f6-4713-a2f1-52713ae138e3';
function model(result = null) {
  return { selection: { example_id: 'A', stage: 'pre_round', algorithm: 'xgboost' },
    status: result ? 'success' : 'idle', ready: true, result,
    detail: { features: { map_name: 'de_ancient', round_num: 4 } }, outcome: { winning_side: 'CT' } };
}
function prediction() { return { ...model().selection, request_id: request, prediction: { ct_win_probability: 0.63 } }; }
function fixture(handler = async () => ({ job_id: job, status: 'success', reply: '实际解释' })) {
  const calls = [];
  const api = async (url, body) => {
    calls.push({ url, body });
    if (url === '/api/chat/status') return { available: true, message: '已登录' };
    if (url === '/api/chat') return { job_id: job, status: 'running' };
    if (url === '/api/chat/cancel') return { job_id: job, status: 'cancelled' };
    return handler(url, body);
  };
  const c = new RoundcastChatController(api, () => {}, async () => {});
  c.setModelState(model(prediction()));
  return { c, calls };
}
test('send attaches immutable current IDs only and success keeps its original context label', async () => {
  const { c, calls } = fixture(); await c.checkStatus(); await c.send(' 为什么胜率高？ ');
  const body = calls.find(call => call.url === '/api/chat').body;
  assert.deepEqual(body, { message: '为什么胜率高？', ...model().selection, request_id: request, history: [] });
  assert.equal(c.state.phase, 'idle'); assert.equal(c.state.messages[1].text, '实际解释');
  assert.match(c.state.messages[1].contextLabel, /ANCIENT/);
  const changed = model(); changed.selection.example_id = 'B'; c.setModelState(changed);
  assert.match(c.state.messages[1].contextLabel, /案例 A/);
  assert(!JSON.stringify(body).includes('winning_side'));
});
test('new, failed, or mismatched predictions never reuse a prior request ID', () => {
  const m = model(prediction()); assert.equal(getChatContext(m).request_id, request);
  m.status = 'running'; assert.equal(getChatContext(m).request_id, null); assert.equal(getChatContext(m).ready, false);
  m.status = 'error'; assert.equal(getChatContext(m).request_id, null);
  m.status = 'success'; m.result.example_id = 'B'; assert.equal(getChatContext(m).request_id, null);
});
test('invalid, unavailable, or duplicate sends never start another job', async () => {
  let finish; const { c, calls } = fixture(() => new Promise(resolve => finish = resolve));
  await c.send('未连接'); await c.checkStatus(); await c.send(' '); await c.send('字'.repeat(2001));
  const pending = c.send('第一个问题'); await Promise.resolve(); await c.send('重复');
  assert.equal(calls.filter(call => call.url === '/api/chat').length, 1);
  finish({ job_id: job, status: 'success', reply: '回答' }); await pending;
});
test('prior history is paired, bounded, and excludes failed conversation turns', async () => {
  const pairs = Array.from({ length: 8 }, (_, i) => [{ role: 'user', text: `q${i}` }, { role: 'assistant', text: `a${i}` }]).flat();
  assert.deepEqual(boundedHistory(pairs), pairs.slice(4));
  assert.deepEqual(boundedHistory([{ role: 'user', text: 'q' }, { role: 'assistant', text: 'x'.repeat(24000) }]), []);
  let fails = true;
  const { c, calls } = fixture(async () => fails ? ({ job_id: job, status: 'error', message: '连接失败' }) : ({ job_id: job, status: 'success', reply: 'ok' }));
  await c.checkStatus(); await c.send('失败问题'); fails = false; await c.send('成功问题');
  assert.deepEqual(calls.filter(call => call.url === '/api/chat')[1].body.history, []);
  await c.send('继续');
  assert.equal(calls.filter(call => call.url === '/api/chat')[2].body.history.length, 2);
});
test('cancelling a delayed reply cannot append an answer after clearing', async () => {
  let finish; const { c } = fixture(() => new Promise(resolve => finish = resolve));
  await c.checkStatus(); const pending = c.send('等待中'); await Promise.resolve();
  await c.cancel(); assert.equal(c.state.phase, 'idle'); c.clear();
  finish({ job_id: job, status: 'success', reply: '迟到回答' }); await pending;
  assert.deepEqual(c.state.messages, []); assert.equal(c.history.length, 0);
});
test('cancel during request creation waits for the job ID and prevents clearing too early', async () => {
  let create; const calls = [];
  const c = new RoundcastChatController(async (url, body) => {
    calls.push({ url, body });
    if (url === '/api/chat/status') return { available: true, message: 'ok' };
    if (url === '/api/chat') return new Promise(resolve => create = resolve);
    if (url === '/api/chat/cancel') return { job_id: job, status: 'cancelled' };
    throw Error('Should never poll a cancelled job');
  });
  c.setModelState(model()); await c.checkStatus();
  const pending = c.send('解释当前输入'); await c.cancel();
  assert.equal(c.state.phase, 'cancelling'); assert.equal(c.clear(), false);
  create({ job_id: job, status: 'running' }); await pending;
  assert.equal(c.state.phase, 'idle'); assert.equal(c.clear(), true);
  assert.equal(calls.filter(call => call.url === '/api/chat/cancel').length, 1);
});
test('failed polling and cancellation keep new sends blocked until stop is confirmed', async () => {
  let cancelFails = true; const { c, calls } = fixture(async () => { throw Error('offline'); });
  const api = c.api;
  c.api = async (url, body) => { if (url === '/api/chat/cancel' && cancelFails) throw Error('offline'); return api(url, body); };
  await c.checkStatus(); await c.send('问题'); assert.equal(c.state.phase, 'poll_error');
  await c.cancel(); await c.send('不应发送'); assert.equal(c.clear(), false);
  assert.equal(calls.filter(call => call.url === '/api/chat').length, 1);
  cancelFails = false; await c.cancel(); assert.equal(c.clear(), true);
});
test('stale login checks and expired history cannot overwrite current connection or context', async () => {
  const { c, calls } = fixture(); const api = c.api; let first;
  c.api = (url, body) => url === '/api/chat/status' && !first ? new Promise(resolve => first = resolve) : api(url, body);
  const old = c.checkStatus(); await c.checkStatus(); first({ available: false, message: 'stale' }); await old;
  assert.equal(c.state.available, true); assert.notEqual(c.state.connection, 'stale');
  c.history = [{ role: 'user', text: 'old' }, { role: 'assistant', text: 'old' }]; c.historyUpdatedAt = Date.now() - 31 * 60 * 1000;
  await c.send('新问题'); assert.deepEqual(calls.find(call => call.url === '/api/chat').body.history, []);
});
test('Enter preserves newlines, Ctrl/Cmd+Enter sends, and IME composition never sends', () => {
  assert.equal(shouldSendOnShortcut({ key: 'Enter' }), false);
  assert.equal(shouldSendOnShortcut({ key: 'Enter', ctrlKey: true }), true);
  assert.equal(shouldSendOnShortcut({ key: 'Enter', metaKey: true }), true);
  assert.equal(shouldSendOnShortcut({ key: 'Enter', ctrlKey: true, isComposing: true }), false);
  assert.equal(shouldSendOnShortcut({ key: 'Enter', ctrlKey: true, keyCode: 229 }), false);
});
test('an explicit missing job on poll releases the active request without claiming cancellation', async () => {
  let missing = true;
  const { c, calls } = fixture(async () => {
    if (missing) throw Object.assign(Error('not found'), { status: 404 });
    return { job_id: job, status: 'success', reply: '新的回复' };
  });
  await c.checkStatus(); await c.send('服务重启前的问题');
  assert.equal(c.state.phase, 'idle'); assert.equal(c.active, null);
  assert.match(c.state.error, /请求已失效/); assert(!c.state.error.includes('已停止'));
  assert.equal(c.clear(), true); missing = false; await c.send('重发问题');
  assert.equal(calls.filter(call => call.url === '/api/chat').length, 2);
  assert.equal(c.state.messages[1].text, '新的回复');
});
test('an explicit missing job on cancel permits clearing and a later send', async () => {
  const { c, calls } = fixture(async () => { throw Error('offline'); });
  const api = c.api;
  c.api = async (url, body) => {
    if (url === '/api/chat/cancel') throw Object.assign(Error('not found'), { status: 404 });
    return api(url, body);
  };
  await c.checkStatus(); await c.send('问题'); await c.cancel();
  assert.equal(c.active, null); assert.equal(c.state.phase, 'idle');
  assert.match(c.state.error, /请求已失效/); assert(!c.state.error.includes('已停止'));
  assert.equal(c.clear(), true); await c.send('重发问题');
  assert.equal(calls.filter(call => call.url === '/api/chat').length, 2);
  await c.cancel();
});
test('reply limit matches the backend: 16000 characters accepted, 16001 rejected', async () => {
  for (const length of [16000, 16001]) {
    const { c } = fixture(async () => ({ job_id: job, status: 'success', reply: '字'.repeat(length) }));
    await c.checkStatus(); await c.send('边界测试');
    assert.equal(c.state.messages.filter(message => message.role === 'assistant').length, length === 16000 ? 1 : 0);
    assert.equal(c.state.phase, length === 16000 ? 'idle' : 'poll_error');
    if (c.active) await c.cancel();
  }
});
