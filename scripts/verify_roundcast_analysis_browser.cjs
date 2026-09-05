// T07 acceptance: reuse the running local preview; never send a Codex message.
const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const playwright=require(process.env.ROUNDCAST_PLAYWRIGHT||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const reference=JSON.parse(fs.readFileSync(path.join(root,'examples/roundcast_v1_cases.json'),'utf8')).reference_probabilities;
const url=process.env.ROUNDCAST_URL||'http://127.0.0.1:8765/';
const output=path.join(root,'reports/roundcast_interactive_v1');
const prefix=process.env.ROUNDCAST_EVIDENCE_PREFIX||'t07';
(async()=>{
  const browser=await playwright.chromium.launch({executablePath:process.env.ROUNDCAST_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const evidence={status:'running',matrix:[],checks:[],pageErrors:[],screenshots:[]};
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1050}});page.setDefaultTimeout(15000);
    page.on('pageerror',e=>evidence.pageErrors.push(e.message));
    const predicts=[],reads=[];
    page.on('request',r=>{if(r.url().includes('/api/'))reads.push(r.url());if(r.url().endsWith('/api/predict'))predicts.push(r.postDataJSON());});
    const ready=()=>page.waitForFunction(()=>!document.querySelector('#run').disabled);
    const view=async name=>{await page.locator(`[data-view="${name}"]`).click();await page.locator(`#${name==='viewer'?'viewer':'analyst'}-view`).waitFor({state:'visible'});};
    await page.goto(url+'#view=analyst&case=B&stage=post_first_kill&algorithm=lightgbm');await ready();
    assert.equal(await page.locator('#analyst-view').isVisible(),true);
    assert.equal(await page.locator('#case-select').inputValue(),'B');assert.equal(await page.locator('#stage-select').inputValue(),'post_first_kill');
    assert.equal(await page.locator('#analysis-snapshot tr').count(),31);assert.equal(predicts.length,0);
    assert.equal(reads.filter(x=>x.endsWith('/outcome')).length,0);
    evidence.checks.push('deep-link-restores-view-and-selection-without-prediction-or-outcome');
    for(const example of ['A','B','C']){
      await page.selectOption('#case-select',example);await ready();
      assert.equal(await page.locator('#analysis-chart .chart-point').count(),0);
      const start=predicts.length;await page.locator('#run-all').click();
      await page.locator('#comparison-status').filter({hasText:'4/4 成功'}).waitFor();await ready();
      assert.equal(predicts.length-start,4);
      for(const point of await page.locator('#analysis-chart .chart-point').all()){
        const model=await point.getAttribute('data-model'),probability=Number(await point.getAttribute('data-probability'));
        assert(Math.abs(probability-reference[example][model])<=1e-8);
        evidence.matrix.push({example_id:example,model_id:model,ct_probability:probability,reference:reference[example][model]});
        const row=page.locator(`#comparison-body tr[data-model="${model}"] td`).nth(2);
        assert.equal(await row.textContent(),(probability*100).toFixed(2)+'%');
      }
      const priorReads=reads.length,requestId=await page.locator('#request-id').textContent();
      await view('viewer');assert.equal(await page.locator('#request-id').textContent(),requestId);
      await page.goBack();await page.locator('#analyst-view').waitFor({state:'visible'});
      await page.goForward();await page.locator('#viewer-view').waitFor({state:'visible'});await view('analyst');
      assert.equal(reads.length,priorReads);assert.equal(await page.locator('#request-id').textContent(),requestId);
      for(const stage of ['pre_round','post_first_kill']){
        await page.selectOption('#stage-select',stage);await ready();
        const snapshot=await (await page.request.get(url+`api/examples/${example}/snapshots/${stage}`)).json();
        const rows=await page.locator('#analysis-snapshot tr').all();assert.equal(rows.length,stage==='pre_round'?27:31);
        for(const row of rows){const key=await row.getAttribute('data-field');assert.equal(await row.locator('td').nth(2).textContent(),String(snapshot.features[key]));}
        for(const key of ['eq_value','cash'])assert.equal(Number(await page.locator(`[data-metric="${key}"] [data-difference]`).getAttribute('data-difference')),snapshot.features[`ct_${key}`]-snapshot.features[`t_${key}`]);
        const x=(reference[example].xgb_post_first_kill-reference[example].xgb_pre_round)*100;
        assert.equal(await page.locator('#analysis-xgb-change').textContent(),`${x>=0?'+':''}${x.toFixed(2)} 个百分点`);
      }
      if(example==='B')for(const width of [1440,1280,390]){
        await page.setViewportSize({width,height:1050});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
        const filename=`${prefix}-analyst-${width}.png`;await page.screenshot({path:path.join(output,filename),fullPage:true});evidence.screenshots.push(filename);
      }
      await page.setViewportSize({width:1440,height:1050});
    }
    assert.equal(evidence.matrix.length,12);
    evidence.checks.push('all-12-chart-points-match-frozen-reference-and-comparison-table','view-back-forward-preserves-request-id-without-api-calls','all-six-snapshots-match-exact-27-31-fields','economy-differences-and-stage-changes-match-original-values','three-widths-without-page-overflow');
    await page.locator('#analysis-reveal').click();await page.locator('#analysis-outcome').filter({hasText:'获胜 · 真实历史赛果'}).waitFor();
    await view('viewer');assert((await page.locator('#outcome').textContent()).includes('获胜 ·'));await view('analyst');
    const retainedURL=page.url(),priorPredicts=predicts.length,priorOutcomes=reads.filter(x=>x.endsWith('/outcome')).length;
    await page.reload();await ready();assert.equal(page.url(),retainedURL);assert.equal(predicts.length,priorPredicts);
    assert.equal(reads.filter(x=>x.endsWith('/outcome')).length,priorOutcomes);assert.equal(await page.locator('#analysis-chart .chart-point').count(),0);
    assert(!(await page.locator('#analysis-outcome').textContent()).includes('获胜 ·'));
    evidence.checks.push('reveal-shared-between-views','reload-restores-selection-not-results-outcome-or-chat');
    // One explicit failure: no fabricated successful prediction.
    await page.route('**/api/predict',route=>route.request().postDataJSON().algorithm==='lightgbm'&&route.request().postDataJSON().stage==='post_first_kill'
      ?route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({status:'error',message:'受控验收：单项失败'})}):route.continue());
    await page.locator('#run-all').click();await ready();
    assert.equal(await page.locator('#analysis-chart .chart-point').count(),3);assert.equal(await page.locator('#analysis-lgb-change').textContent(),'—');
    await page.unroute('**/api/predict');const beforeRetry=predicts.length;
    await page.locator('tr[data-model="lgbm_post_first_kill"] .row-run').click();await ready();
    assert.equal(await page.locator('#analysis-chart .chart-point').count(),4);assert.equal(predicts.length,beforeRetry+1);
    evidence.checks.push('partial-failure-removes-point-and-delta','single-retry-restores-point-with-one-request');
    let release,started;const gate=new Promise(r=>release=r),hit=new Promise(r=>started=r);
    await page.route('**/api/predict',async route=>{started();await gate;await route.continue();});
    const beforeBatch=predicts.length;await page.locator('#run-all').click();await hit;await view('viewer');await view('analyst');release();await ready();
    await page.unroute('**/api/predict');assert.equal(predicts.length,beforeBatch+4);
    assert.equal(await page.locator('#analysis-chart .chart-point').count(),4);
    evidence.checks.push('view-switch-during-batch-preserves-four-runs');
    // Target the narrow interval after a native hash changes but before hashchange runs.
    // The prediction response is still real; only navigation timing is injected.
    await page.evaluate(()=>{
      const original=window.fetch;
      window.t07RouteRaceLog=[];
      const trace=phase=>window.t07RouteRaceLog.push({phase,hash:location.hash,viewerVisible:!document.querySelector('#viewer-view').hidden});
      window.addEventListener('popstate',()=>trace('popstate'),{once:true});
      window.addEventListener('hashchange',()=>trace('hashchange'),{once:true});
      window.t07RouteRaceDone=new Promise(resolve=>window.addEventListener('hashchange',()=>queueMicrotask(resolve),{once:true}));
      window.fetch=async(...args)=>{
        const response=await original(...args);
        if(String(args[0]).endsWith('/api/predict')){
          window.fetch=original;const json=response.json.bind(response);
          response.json=async()=>{const data=await json();trace('before-hash');location.hash=location.hash.replace('view=analyst','view=viewer');trace('after-hash');return data;};
        }
        return response;
      };
    });
    await page.locator('#run').click();await page.evaluate(()=>window.t07RouteRaceDone);await ready();
    assert.equal(await page.locator('#viewer-view').isVisible(),true,'prediction completion must not overwrite a pending native navigation');
    assert(page.url().includes('view=viewer'));
    evidence.navigationRaceTrace=await page.evaluate(()=>window.t07RouteRaceLog);
    evidence.checks.push('prediction-completion-does-not-overwrite-pending-native-navigation');
    // URL state only; no arbitrary endpoint, result, or hidden outcome survives normalization.
    await page.evaluate(()=>{location.hash='view=technical&case=Z&stage=live&algorithm=invalid&winning_side=T';});await ready();
    await page.waitForFunction(()=>location.hash==='#view=viewer&case=A&stage=pre_round&algorithm=xgboost');
    assert.equal(await page.locator('#viewer-view').isVisible(),true);assert.equal(await page.locator('#ct-probability').textContent(),'—');
    assert.deepEqual(evidence.pageErrors,[]);evidence.checks.push('invalid-hash-normalized-to-supported-selection','no-page-errors');
    evidence.status='passed';fs.writeFileSync(path.join(output,`${prefix}-browser-evidence.json`),JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify({status:evidence.status,realCombinations:evidence.matrix.length,checks:evidence.checks}));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
