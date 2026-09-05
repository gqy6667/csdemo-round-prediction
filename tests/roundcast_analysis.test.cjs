const test = require('node:test');
const assert = require('node:assert/strict');
const {RoundcastController, parseRoute, routeHash, analysisData} = require('../web/roundcast/app.js');
const metadata = require('./roundcast_prediction_fixture.cjs');
const pairs = [['pre_round','xgboost'],['pre_round','lightgbm'],['post_first_kill','xgboost'],['post_first_kill','lightgbm']];
const key = (stage, algorithm) => `${algorithm === 'xgboost' ? 'xgb' : 'lgbm'}_${stage}`;
function fixture(route) {
  const calls=[];
  const api=async(url,body)=>{
    calls.push({url,body});
    if(url==='/api/models')return {models:pairs.map(([stage,algorithm])=>({stage,algorithm,model_id:key(stage,algorithm),inference_ready:true,available_examples:['A','B','C']}))};
    if(url==='/api/examples')return {examples:['A','B','C'].map(example_id=>({example_id,inference_ready:true}))};
    if(body)return {...metadata(body),...body,status:'success',model_id:key(body.stage,body.algorithm),request_id:`${body.example_id}-${key(body.stage,body.algorithm)}`,
      prediction:{ct_win_probability:body.stage==='pre_round'?.7:.4,t_win_probability:body.stage==='pre_round'?.3:.6,decision_threshold:.5,predicted_side:body.stage==='pre_round'?'CT':'T'}};
    const example_id=url.split('/')[3], stage=url.endsWith('/post_first_kill')?'post_first_kill':'pre_round';
    if(url.endsWith('/outcome'))return {example_id,winning_side:'T'};
    return {example_id,stage,features:{map_name:'de_mirage',round_num:11,ct_score:5,t_score:5,ct_cash:1000,t_cash:2000,ct_eq_value:20000,t_eq_value:12000,
      ...(stage==='post_first_kill'?{first_kill_advantage_ct:-1,first_kill_time:21.5,first_kill_headshot:0,first_kill_weapon:'AK-47'}:{})}};
  };
  return {c:new RoundcastController(api,()=>{},route),calls};
}
test('route restores only known view and selection fields, never results or spoilers',()=>{
  const route=parseRoute('#view=analyst&case=B&stage=post_first_kill&algorithm=lightgbm&winning_side=T');
  assert.deepEqual(route,{view:'analyst',example_id:'B',stage:'post_first_kill',algorithm:'lightgbm'});
  assert.deepEqual(parseRoute(routeHash(route)),route);
  for(const hash of ['#view=technical&case=Z&stage=live&algorithm=evil','#view=%',''])
    assert.deepEqual(parseRoute(hash),{view:'viewer',example_id:'A',stage:'pre_round',algorithm:'xgboost'});
});
test('initial route and view-only navigation do not infer or fetch hidden outcome',async()=>{
  const route={view:'analyst',example_id:'B',stage:'post_first_kill',algorithm:'lightgbm'};
  const {c,calls}=fixture(route);await c.load();
  assert.equal(c.state.view,'analyst');assert.equal(c.state.selection.example_id,'B');assert.equal(c.state.result,null);
  const count=calls.length;await c.navigate({view:'viewer'});await c.navigate({view:'analyst'});
  assert.equal(calls.length,count);assert.equal(c.state.detail.stage,'post_first_kill');
  assert.equal(calls.filter(r=>r.body||r.url.endsWith('/outcome')).length,0);
});
test('shared navigation retains same-case results and clears results/outcome on case change',async()=>{
  const {c}=fixture();await c.load();await c.runAll();await c.reveal();const result=c.state.result;
  await c.navigate({view:'analyst'});assert.equal(c.state.result,result);assert.equal(c.state.outcome.winning_side,'T');
  await c.navigate({view:'viewer',stage:'post_first_kill',algorithm:'lightgbm'});
  assert.equal(c.state.result.model_id,'lgbm_post_first_kill');assert.equal(c.state.selection.example_id,'A');
  await c.navigate({view:'analyst',example_id:'B'});
  assert.equal(c.state.result,null);assert.equal(c.state.outcome,null);assert.equal(c.state.detail.example_id,'B');
  assert.equal(await c.navigate({view:'admin'}),false);assert.equal(c.state.view,'analyst');
});
test('analysis only publishes successful current-case probabilities; missing is not zero',async()=>{
  const {c}=fixture();await c.load();let data=analysisData(c.state);
  assert(data.probabilities.every(r=>r.probability===null));assert.equal(data.changes.xgboost,null);
  await c.runAll();data=analysisData(c.state);
  assert.equal(data.probabilities[0].probability,.7);assert(Math.abs(data.changes.xgboost+30)<1e-10);
  c.state.comparisons.lgbm_post_first_kill.status='error';data=analysisData(c.state);
  assert.equal(data.probabilities[3].probability,null);assert.equal(data.changes.lightgbm,null);
  c.state.comparisons.xgb_pre_round.result.example_id='C';
  assert.equal(analysisData(c.state).probabilities[0].probability,null);
});
test('analysis snapshot is read-only, stage bounded and excludes unapproved fields',async()=>{
  const {c}=fixture();await c.load();c.state.detail.features.winning_side='T';
  let data=analysisData(c.state);assert.equal(data.fields.length,27);
  assert(!data.fields.some(f=>f.key==='winning_side'||f.key.startsWith('first_kill_')));
  assert.equal(data.economy[0].difference,8000);assert.equal(data.economy[1].difference,-1000);
  assert.equal(data.fields.find(f=>f.key==='ct_armor').value,null);
  await c.select({stage:'post_first_kill'});data=analysisData(c.state);
  assert.equal(data.fields.length,31);assert.equal(data.fields.find(f=>f.key==='first_kill_advantage_ct').value,-1);
  assert.equal(data.economy[0].difference,8000);
  c.state.ready=false;data=analysisData(c.state);assert.equal(data.fields.length,0);assert.equal(data.economy.length,0);
});
