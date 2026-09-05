const test = require('node:test');
const assert = require('node:assert/strict');
const {RoundcastController, parseRoute, technicalData} = require('../web/roundcast/app.js');
const metadata = require('./roundcast_prediction_fixture.cjs');
const pairs = [['pre_round','xgboost'],['pre_round','lightgbm'],['post_first_kill','xgboost'],['post_first_kill','lightgbm']];
const key = (stage, algorithm) => `${algorithm === 'xgboost' ? 'xgb' : 'lgbm'}_${stage}`;
const report = () => ({status:'success',models:pairs.map(([stage,algorithm])=>({model_id:key(stage,algorithm),stage,algorithm,status:'success',
  scope:'frozen_test_set',split:'test',split_unit:'series_id',n_test:stage==='pre_round'?4172:4170,
  metrics:{accuracy:.7,auc:.8,log_loss:.5,brier_score:.2,ece10:.01},source:{path:'reports/source.csv',sha256:'a'.repeat(64)}}))});
function fixture(metricsApi=async()=>report()) {
  const calls=[];
  const api=async(url,body)=>{
    calls.push(url);
    if(url==='/api/metrics')return metricsApi();
    if(url==='/api/models')return {models:pairs.map(([stage,algorithm])=>({stage,algorithm,model_id:key(stage,algorithm),inference_ready:true,available_examples:['A','B','C']}))};
    if(url==='/api/examples')return {examples:['A','B','C'].map(example_id=>({example_id,inference_ready:true}))};
    if(body)return {...metadata(body),...body,status:'success',model_id:key(body.stage,body.algorithm),request_id:'run-1',
      prediction:{ct_win_probability:.4,t_win_probability:.6,predicted_side:'T',decision_threshold:.5}};
    const example_id=url.split('/')[3];
    if(url.endsWith('/outcome'))return {example_id,winning_side:'CT'};
    return {example_id,stage:'pre_round',features:{map_name:'de_nuke',round_num:17,ct_score:7,t_score:9,ct_cash:0,t_cash:200,ct_eq_value:21000,t_eq_value:17000}};
  };
  return {c:new RoundcastController(api),calls};
}
test('technical deep link and three-view navigation preserve results without automatic actions',async()=>{
  assert.equal(parseRoute('#view=technical&case=B&stage=post_first_kill&algorithm=lightgbm').view,'technical');
  const {c,calls}=fixture();await c.load();await c.loadMetrics();await c.run();const result=c.state.result;
  const count=calls.length;
  for(const view of ['technical','analyst','viewer']){assert.equal(await c.navigate({view}),true);assert.equal(c.state.result,result);}
  assert.equal(calls.length,count);assert(!calls.some(url=>url.endsWith('/outcome')));
});
test('formal metrics load independently and failures never block prediction',async()=>{
  const {c}=fixture(async()=>{throw Error('metrics offline');});await c.load();await c.loadMetrics();
  assert.equal(c.state.ready,true);assert.equal(c.state.metricsStatus,'error');assert.equal(technicalData(c.state).metrics,null);
  await c.run();assert.equal(c.state.status,'success');assert.equal(technicalData(c.state).prediction.request_id,'run-1');
  // Metrics can finish before model catalogs without making prediction ready.
  const early=fixture();let releaseModels;const gate=new Promise(resolve=>releaseModels=resolve),original=early.c.api;
  early.c.api=async(url,body)=>{if(url==='/api/models')await gate;return original(url,body);};
  const loading=early.c.load();await early.c.loadMetrics();
  assert.equal(early.c.state.ready,false);assert.equal(technicalData(early.c.state).metrics.n_test,4172);
  releaseModels();await loading;assert.equal(early.c.state.ready,true);assert.equal(early.c.state.result,null);
  // A late metrics failure must not clear a completed prediction or reveal.
  let rejectMetrics;const late=fixture(()=>new Promise((_,reject)=>rejectMetrics=reject));
  await late.c.load();const metricsPending=late.c.loadMetrics();await late.c.run();await late.c.reveal();
  const result=late.c.state.result,outcome=late.c.state.outcome;
  rejectMetrics(Error('late metrics failure'));await metricsPending;
  assert.equal(late.c.state.ready,true);assert.equal(late.c.state.result,result);assert.equal(late.c.state.outcome,outcome);
  assert.equal(technicalData(late.c.state).correct,false);assert.equal(late.c.state.metricsStatus,'error');
});
test('metrics match selected model and stage; unavailable or duplicate rows do not become zero',async()=>{
  const {c}=fixture();await c.load();await c.loadMetrics();
  assert.equal(technicalData(c.state).metrics.n_test,4172);
  c.state.selection={example_id:'A',stage:'post_first_kill',algorithm:'lightgbm'};
  assert.equal(technicalData(c.state).metrics.n_test,4170);
  const row=c.state.metrics.models[3];row.stage='pre_round';assert.equal(technicalData(c.state).metrics,null);
  row.stage='post_first_kill';row.metrics.accuracy=0;row.metrics.log_loss=2;
  assert.equal(technicalData(c.state).metrics.metrics.accuracy,0);
  row.metrics.auc=null;assert.equal(technicalData(c.state).metrics,null);row.metrics.auc=.8;
  c.state.metrics.models.push({...row});assert.equal(technicalData(c.state).metrics,null);
});
test('metrics reload clears stale success and ignores late earlier response',async()=>{
  let resolveFirst,resolveSecond;
  const {c}=fixture(()=>new Promise(resolve=>{if(!resolveFirst)resolveFirst=resolve;else resolveSecond=resolve;}));
  await c.load();const first=c.loadMetrics(),second=c.loadMetrics();
  assert.equal(c.state.metricsStatus,'loading');assert.equal(technicalData(c.state).metrics,null);
  resolveSecond(report());await second;resolveFirst({status:'unavailable',models:[]});await first;
  assert.equal(c.state.metricsStatus,'success');assert.equal(technicalData(c.state).metrics.n_test,4172);
});
test('current correctness requires matching successful prediction and explicitly revealed outcome',async()=>{
  const {c}=fixture();await c.load();await c.loadMetrics();assert.equal(technicalData(c.state).correct,null);
  await c.reveal();assert.equal(technicalData(c.state).correct,null);
  await c.run();assert.equal(technicalData(c.state).correct,false);
  c.state.result.prediction.predicted_side='CT';assert.equal(technicalData(c.state).correct,true);
  c.state.result.example_id='B';assert.equal(technicalData(c.state).prediction,null);assert.equal(technicalData(c.state).correct,null);
  c.state.result.example_id='A';c.state.outcomeStatus='error';assert.equal(technicalData(c.state).correct,null);
  c.state.outcomeStatus='success';c.state.status='running';assert.equal(technicalData(c.state).correct,null);
});
