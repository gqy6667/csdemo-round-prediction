const test = require('node:test');
const assert = require('node:assert/strict');
const { RoundcastController } = require('../web/roundcast/app.js');
const metadata = require('./roundcast_prediction_fixture.cjs');
const pairs = [['pre_round','xgboost'],['pre_round','lightgbm'],['post_first_kill','xgboost'],['post_first_kill','lightgbm']];
const id = (stage, algorithm) => `${algorithm === 'xgboost' ? 'xgb' : 'lgbm'}_${stage}`;
const features = {map_name:'de_ancient',round_num:4,ct_score:2,t_score:1,ct_cash:6000,t_cash:400,ct_eq_value:23900,t_eq_value:14650};
const reply = s => ({...metadata(s),...s,status:'success',model_id:id(s.stage,s.algorithm),request_id:'request-'+Math.random(),
  prediction:{ct_win_probability:s.algorithm==='xgboost'?.6:.65,t_win_probability:s.algorithm==='xgboost'?.4:.35,decision_threshold:.5,predicted_side:'CT'}});
function fixture(hook = async (url, body) => body ? reply(body) : null) {
  const calls = [];
  const api = async (url, body) => {
    calls.push({url,body});
    if (url === '/api/models') return {models:pairs.map(([stage,algorithm])=>({stage,algorithm,model_id:id(stage,algorithm),inference_ready:true,available_examples:['A','B','C']}))};
    if (url === '/api/examples') return {examples:['A','B','C'].map(example_id=>({example_id,inference_ready:true}))};
    const overridden = await hook(url,body); if(overridden) return overridden;
    const [,example,stage] = url.match(/examples\/([ABC])(?:\/snapshots\/(pre_round|post_first_kill))?/) || [];
    if(url.endsWith('/outcome')) return {example_id:example,winning_side:'CT'};
    return {example_id:example,stage:stage||'pre_round',features:{...features,map_name:example==='A'?'de_ancient':'de_mirage',round_num:example==='A'?4:11,
      ...(stage==='post_first_kill'?{first_kill_advantage_ct:1,first_kill_time:25,first_kill_headshot:1,first_kill_weapon:'AK47'}:{})}};
  };
  return {c:new RoundcastController(api),calls};
}
test('all twelve selections only load stage-correct snapshots until explicitly run',async()=>{
  const {c,calls}=fixture();await c.load();
  for(const example_id of ['A','B','C']) for(const [stage,algorithm] of pairs){
    await c.select({example_id,stage,algorithm});
    assert.equal(c.state.detail.example_id,example_id); assert.equal(c.state.detail.stage,stage);
    assert.equal(c.state.result,null); assert.equal(c.state.ready,true);
  }
  assert.equal(calls.filter(v=>v.body||v.url.endsWith('/outcome')).length,0);
});
test('compare issues four unique calls and selected hero is not last response',async()=>{
  const {c,calls}=fixture();await c.load();await c.runAll();
  const requests=calls.filter(v=>v.url==='/api/predict');assert.equal(requests.length,4);
  assert.equal(new Set(requests.map(v=>JSON.stringify(v.body))).size,4);
  assert.equal(c.state.result.model_id,'xgb_pre_round');
  assert.equal(Object.values(c.state.comparisons).filter(v=>v.status==='success').length,4);
  await c.select({algorithm:'lightgbm'});assert.equal(c.state.result.model_id,'lgbm_pre_round');
});
test('partial comparison failure stays empty and one retry only runs one model',async()=>{
  let fail=true; const {c,calls}=fixture(async(url,body)=>{
    if(!body)return null;if(fail&&body.stage==='post_first_kill'&&body.algorithm==='lightgbm')throw Error('missing model');return reply(body);
  });await c.load();await c.runAll();
  assert.equal(c.state.comparisons.lgbm_post_first_kill.status,'error');
  assert.equal(c.state.comparisons.lgbm_post_first_kill.result,null);
  assert.equal(Object.values(c.state.comparisons).filter(v=>v.status==='success').length,3);
  fail=false;await c.runPair('post_first_kill','lightgbm');
  assert.equal(calls.filter(v=>v.body).length,5);assert.equal(c.state.comparisons.lgbm_post_first_kill.status,'success');
});
test('switch A B A invalidates old predictions and comparison does not continue old case',async()=>{
  let resolve;const {c,calls}=fixture(async(url,body)=>body?new Promise(r=>resolve=()=>r(reply(body))):null);
  await c.load();const pending=c.runAll();await Promise.resolve();
  await c.select({example_id:'B'});await c.select({example_id:'A'});resolve();await pending;
  assert.equal(c.state.result,null);assert.equal(c.state.comparing,false);
  assert.equal(calls.filter(v=>v.body).length,1);
  assert(Object.values(c.state.comparisons).every(v=>v.result===null));
});
test('late stage details and outcome never overwrite current selection',async()=>{
  let detailResolve,outcomeResolve;
  const {c}=fixture(async(url)=>{
    if(url.endsWith('/post_first_kill'))return new Promise(r=>detailResolve=r);
    if(url.endsWith('/outcome'))return new Promise(r=>outcomeResolve=r);
    return null;
  });await c.load();const detail=c.select({stage:'post_first_kill'});
  assert.equal(c.state.detail,null);assert.equal(c.state.result,null);
  await c.select({stage:'pre_round'});
  detailResolve({example_id:'A',stage:'post_first_kill',features:{first_kill_side:'CT'}});await detail;
  assert.equal(c.state.detail.stage,'pre_round');
  const outcome=c.reveal();await c.select({example_id:'B'});
  outcomeResolve({example_id:'A',winning_side:'CT'});await outcome;
  assert.equal(c.state.outcome,null);assert.equal(c.state.outcomeStatus,'idle');
});
test('duplicate comparison clicks and rerun clear all old results',async()=>{
  const {c,calls}=fixture();await c.load();await c.runAll();const first=c.state.result.request_id;
  const pending=c.runAll();await c.runAll();await pending;
  assert.equal(calls.filter(v=>v.body).length,8);assert.notEqual(c.state.result.request_id,first);
});
test('malformed post-first-kill display fields fail closed before publishing ready state',async()=>{
  const good={...features,first_kill_advantage_ct:1,first_kill_time:25,first_kill_headshot:1,first_kill_weapon:'AK47'};
  for(const bad of [{first_kill_time:null},{first_kill_time:'25'},{first_kill_time:181},
    {first_kill_advantage_ct:0},{first_kill_headshot:'false'},{first_kill_weapon:''},{ct_cash:NaN}]){
    const {c}=fixture(async url=>url.endsWith('/post_first_kill')?{example_id:'A',stage:'post_first_kill',features:{...good,...bad}}:null);
    await c.load();await c.select({stage:'post_first_kill'});
    assert.equal(c.state.ready,false);assert.equal(c.state.status,'error');assert.equal(c.state.detail,null);
  }
});

test('reconnect cannot use old ready catalogs when selecting a new case',async()=>{
  const {c,calls}=fixture();await c.load();let resolveCatalog;const original=c.api;
  c.api=async(url,body)=>url==='/api/models'?new Promise(resolve=>resolveCatalog=resolve):original(url,body);
  const pending=c.load();const before=calls.length;
  assert.equal(await c.select({example_id:'B'}),false);
  await c.navigate({view:'technical'});assert.equal(c.state.view,'technical');
  assert.equal(c.state.ready,false);assert.equal(calls.length,before);
  resolveCatalog({models:pairs.map(([stage,algorithm])=>({stage,algorithm,model_id:id(stage,algorithm),inference_ready:false,available_examples:[]}))});
  await pending;await c.run();assert.equal(c.state.ready,false);assert.equal(c.state.status,'error');
  assert.equal(calls.filter(call=>call.body).length,0);assert(c.state.models.every(m=>!m.inference_ready));
});

test('view-only navigation remains available with failed or malformed catalogs',async()=>{
  for(const catalog of [null,{models:[{stage:'pre_round',algorithm:'xgboost',model_id:'xgb_pre_round',inference_ready:true}]}]){
    const {c}=fixture();const original=c.api;
    c.api=async(url,body)=>{if(url==='/api/models'){if(!catalog)throw Error('catalog offline');return catalog;}return original(url,body);};
    await c.load();assert.equal(c.state.ready,false);
    assert.equal(await c.navigate({view:'technical'}),true);assert.equal(c.state.view,'technical');
    assert.equal(await c.select({example_id:'B'}),false);assert.equal(c.state.ready,false);
  }
});
