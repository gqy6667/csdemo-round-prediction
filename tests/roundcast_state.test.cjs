const test = require('node:test');
const assert = require('node:assert/strict');
const { RoundcastController, registerRoundcastTool } = require('../web/roundcast/app.js');
const selection = { example_id: 'A', stage: 'pre_round', algorithm: 'xgboost' };
const response = () => ({ ...selection, status: 'success', model_id: 'xgb_pre_round', request_id: 'test-only',
  prediction: { ct_win_probability: 0.6, t_win_probability: 0.4, decision_threshold: 0.5, predicted_side: 'CT' } });
function fixture(predict = async () => response()) {
  const calls = [];
  const api = async (url, body) => {
    calls.push([url, body]);
    if (url === '/api/models') return { models: [{ ...selection, model_id: 'xgb_pre_round', inference_ready: true, available_examples: ['A'] }] };
    if (url === '/api/examples') return { examples: [{ example_id: 'A', inference_ready: true }] };
    if (url.endsWith('/outcome')) return { example_id: 'A', winning_side: 'CT' };
    if (url === '/api/examples/A') return { example_id: 'A', stage: 'pre_round', features: { map_name: 'de_ancient', round_num: 4 } };
    return predict();
  };
  return { controller: new RoundcastController(api), calls };
}
test('starts empty; prediction and outcome only follow explicit actions', async () => {
  const { controller: c, calls } = fixture();
  await c.load();
  assert.equal(c.state.status, 'idle'); assert.equal(c.state.result, null); assert.equal(c.state.outcome, null);
  assert.equal(calls.filter(([u]) => u === '/api/predict' || u.endsWith('/outcome')).length, 0);
  await c.run(); assert.equal(c.state.result.prediction.ct_win_probability, 0.6);
  await c.run(); assert.equal(calls.filter(([u]) => u === '/api/predict').length, 2);
  await c.reveal(); assert.equal(c.state.outcome.winning_side, 'CT');
});
test('duplicate clicks produce one request and running state has no old result', async () => {
  let resolve; const { controller: c, calls } = fixture(() => new Promise(r => resolve = r));
  await c.load(); const pending = c.run(); await c.run();
  assert.equal(c.state.status, 'running'); assert.equal(c.state.result, null);
  assert.equal(calls.filter(([u]) => u === '/api/predict').length, 1);
  resolve(response()); await pending; assert.equal(c.state.status, 'success');
});
test('failed rerun removes old success', async () => {
  let fail = false; const { controller: c } = fixture(async () => { if (fail) throw Error('offline'); return response(); });
  await c.load(); await c.run(); fail = true; await c.run();
  assert.equal(c.state.status, 'error'); assert.equal(c.state.result, null);
});
test('refresh invalidates a slow previous prediction', async () => {
  let resolve; const { controller: c } = fixture(() => new Promise(r => resolve = r));
  await c.load(); const pending = c.run(); await c.load(); resolve(response()); await pending;
  assert.equal(c.state.status, 'idle'); assert.equal(c.state.result, null);
});
test('wrong combination, malformed probability or response cannot become success', async () => {
  for (const bad of [{ ...response(), example_id: 'B' }, { ...response(), prediction: { ct_win_probability: NaN, t_win_probability: 0.4 } }, null]) {
    const { controller: c } = fixture(async () => bad); await c.load(); await c.run();
    assert.equal(c.state.status, 'error'); assert.equal(c.state.result, null);
  }
});
test('optional page tool shares the run action and rejects extra input', async () => {
  const { controller: c } = fixture(); await c.load();
  let tool, signal;
  const cleanup = registerRoundcastTool({ registerTool(value, options) { tool = value; signal = options.signal; } }, c);
  assert.equal(tool.name, 'run_current_round_prediction');
  await assert.rejects(() => tool.execute({ path: '/private' }));
  assert.equal(c.state.result, null);
  const result = await tool.execute({});
  assert.equal(result.request_id, c.state.result.request_id);
  assert.equal(c.state.status, 'success'); cleanup(); assert(signal.aborted);
});
