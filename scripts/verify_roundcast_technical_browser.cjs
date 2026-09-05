// T08/T09 acceptance against the running local preview. Never sends a Codex message.
const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..');
const playwright=require(process.env.ROUNDCAST_PLAYWRIGHT||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const registry=JSON.parse(fs.readFileSync(path.join(root,'examples/roundcast_v1_cases.json'),'utf8'));
const url=process.env.ROUNDCAST_URL||'http://127.0.0.1:8765/';
const output=path.join(root,'reports/roundcast_interactive_v1');
const prefix=process.env.ROUNDCAST_EVIDENCE_PREFIX||'t09';
const pairs=[['pre_round','xgboost'],['pre_round','lightgbm'],['post_first_kill','xgboost'],['post_first_kill','lightgbm']];
const key=(stage,algorithm)=>`${algorithm==='xgboost'?'xgb':'lgbm'}_${stage}`;
(async()=>{
  const browser=await playwright.chromium.launch({executablePath:process.env.ROUNDCAST_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const evidence={status:'running',matrix:[],checks:[],pageErrors:[],screenshots:[]};
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1050}});page.setDefaultTimeout(15000);
    page.on('pageerror',e=>evidence.pageErrors.push(e.message));
    const writes=[],outcomeReads=[],responseReads=[],predictions=new Map();
    page.on('request',r=>{if(r.method()==='POST')writes.push(r.url());if(r.url().endsWith('/outcome'))outcomeReads.push(r.url());});
    page.on('response',r=>{if(r.url().endsWith('/api/predict')&&r.status()===200)responseReads.push(r.json().then(body=>predictions.set(body.request_id,body)));});
    const ready=()=>page.waitForFunction(()=>!document.querySelector('#run').disabled);
    const metricsReady=()=>page.waitForFunction(()=>document.querySelector('#metrics-status').dataset.status==='success');
    const view=async name=>{await page.locator(`[data-view="${name}"]`).click();await page.locator(`#${name}-view`).waitFor({state:'visible'});};
    await page.goto(url+'#view=technical&case=A&stage=pre_round&algorithm=xgboost');await ready();await metricsReady();
    assert.equal(await page.locator('#technical-view').isVisible(),true);
    assert.equal(writes.length,0);assert.equal(outcomeReads.length,0);
    assert.equal(await page.locator('#technical-version').textContent(),'—');
    assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    const metrics=await (await page.request.get(url+'api/metrics')).json();assert.equal(metrics.status,'success');
    const models=(await (await page.request.get(url+'api/models')).json()).models;
    assert.equal(metrics.models.length,4);
    for(const row of metrics.models){
      const bytes=fs.readFileSync(path.join(root,row.source.path));
      assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),row.source.sha256);
      let expected;
      if(row.source.path.endsWith('.json')){
        expected=JSON.parse(bytes.toString('utf8'));for(const part of row.source.selector.split('.'))expected=expected[part];
      }else expected=Object.fromEntries(bytes.toString('utf8').trim().split(/\r?\n/).slice(1).map(line=>{const [k,v]=line.split(',');return [k,Number(v)];}));
      for(const [name,value]of Object.entries(expected))assert(Math.abs(value-row.metrics[name])<1e-14);
      assert.equal(row.n_test,row.stage==='pre_round'?4172:4170);
    }
    fs.writeFileSync(path.join(output,`${prefix==='t09'?'t08':prefix}-metrics-evidence.json`),JSON.stringify(metrics,null,2)+'\n');
    evidence.checks.push('four-formal-metric-sources-match-by-sha256-and-all-five-values','technical-deep-link-does-not-predict-or-reveal');
    for(const example of ['A','B','C']){
      await page.selectOption('#case-select',example);await ready();
      assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
      assert.equal(await page.locator('#technical-version').textContent(),'—');
      const before=writes.length;await page.locator('#run-all').click();
      await page.locator('#comparison-status').filter({hasText:'4/4 成功'}).waitFor();await ready();
      await Promise.all(responseReads);
      assert.equal(writes.length-before,4);
      for(const [stage,algorithm]of pairs){
        await page.selectOption('#stage-select',stage);await ready();await page.selectOption('#algorithm-select',algorithm);await ready();
        const model=key(stage,algorithm),formal=metrics.models.find(r=>r.model_id===model);
        assert.equal(await page.locator('#metrics-scope').textContent(),`${formal.n_test.toLocaleString('en-US')} 个测试回合 · 按 series_id 划分`);
        for(const [name,value]of Object.entries(formal.metrics)){
          const cell=page.locator(`#technical-metrics [data-metric="${name}"] dd`);
          assert.equal(Number(await cell.getAttribute('data-value')),value);
          assert.equal(await cell.textContent(),name==='accuracy'?(value*100).toFixed(2)+'%':value.toFixed(6));
        }
        const source=JSON.parse(await page.locator('#technical-source').textContent());
        assert.equal(source.model_id,model);assert.equal(source.request_id,await page.locator('#request-id').textContent());
        assert.equal(source.model_sha256,models.find(row=>row.model_id===model).model_sha256);
        assert.equal(source.calibrator_sha256,models.find(row=>row.model_id===model).calibrator_sha256);
        const actual=predictions.get(source.request_id);assert(actual);assert.equal(actual.model_id,model);assert.equal(actual.example_id,example);
        assert(Math.abs(actual.prediction.ct_win_probability-registry.reference_probabilities[example][model])<=1e-8);
        assert.equal(await page.locator('#technical-version').textContent(),actual.model_version);
        assert.equal(await page.locator('#technical-elapsed').textContent(),`${actual.inference_ms.toFixed(1)} ms`);
        assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
        assert.equal(await page.locator('#ct-probability').textContent(),(registry.reference_probabilities[example][model]*100).toFixed(2)+'%');
        evidence.matrix.push({example_id:example,model_id:model,ct_probability:actual.prediction.ct_win_probability,reference:registry.reference_probabilities[example][model],request_id:source.request_id,n_test:formal.n_test});
      }
      const requestId=await page.locator('#request-id').textContent(),count=writes.length;
      await view('viewer');await view('analyst');await view('technical');
      assert.equal(await page.locator('#request-id').textContent(),requestId);assert.equal(writes.length,count);
      await page.goBack();await page.locator('#analyst-view').waitFor({state:'visible'});
      await page.goForward();await page.locator('#technical-view').waitFor({state:'visible'});
      const outcomeResponse=page.waitForResponse(r=>r.url().endsWith('/outcome'));
      await page.locator('#technical-reveal').click();const outcome=await (await outcomeResponse).json();
      await page.waitForFunction(()=>document.querySelector('#technical-correctness').dataset.correct!=='');
      const p=registry.reference_probabilities[example].lgbm_post_first_kill;
      assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),String((p>=.5?'CT':'T')===outcome.winning_side));
      await view('viewer');assert((await page.locator('#outcome').textContent()).startsWith(outcome.winning_side));await view('technical');
      evidence.matrix[evidence.matrix.length-1].revealed_winner=outcome.winning_side;
      if(example==='B')for(const width of [1440,1280,390]){
        await page.setViewportSize({width,height:1050});await page.evaluate(()=>scrollTo(0,0));
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
        const filename=`${prefix}-technical-${width}.png`;await page.screenshot({path:path.join(output,filename),fullPage:true});evidence.screenshots.push(filename);
      }
      await page.setViewportSize({width:1440,height:1050});
    }
    evidence.checks.push('all-12-combinations-display-matching-metrics-and-runtime','three-views-back-forward-share-results-and-outcome','outcome-correctness-only-after-explicit-reveal','three-widths-without-page-overflow');
    // Controlled metrics outage: genuine inference remains usable, old metrics disappear.
    await page.route('**/api/metrics',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({message:'受控验收：指标暂时不可用'})}));
    await page.locator('#metrics-reload').click();await page.waitForFunction(()=>document.querySelector('#metrics-status').dataset.status==='error');
    for(const cell of await page.locator('#technical-metrics dd').all())assert.equal(await cell.textContent(),'—');
    await page.locator('#run').click();await ready();assert.equal(await page.locator('#technical-status').textContent(),'本次运行成功');
    await page.unroute('**/api/metrics');await page.locator('#metrics-reload').click();await metricsReady();
    // Failed rerun clears correctness and runtime, but not valid formal metrics.
    await page.route('**/api/predict',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({message:'受控验收：本次运行失败'})}));
    await page.locator('#run').click();await ready();assert.equal(await page.locator('#technical-version').textContent(),'—');
    assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    assert.equal(await page.locator('#metrics-status').getAttribute('data-status'),'success');
    await page.unroute('**/api/predict');
    const retainedURL=page.url(),writeCount=writes.length,outcomes=outcomeReads.length;
    await page.reload();await ready();await metricsReady();assert.equal(page.url(),retainedURL);
    assert.equal(writes.length,writeCount);assert.equal(outcomeReads.length,outcomes);
    assert.equal(await page.locator('#technical-version').textContent(),'—');assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    assert(writes.every(item=>item.endsWith('/api/predict')));assert.deepEqual(evidence.pageErrors,[]);
    evidence.checks.push('metrics-outage-clears-stale-values-without-blocking-real-inference','prediction-failure-clears-runtime-and-correctness-not-formal-metrics','refresh-restores-selection-not-results-or-outcome','no-codex-messages','no-page-errors');
    evidence.status='passed';fs.writeFileSync(path.join(output,`${prefix}-browser-evidence.json`),JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify({status:evidence.status,combinations:evidence.matrix.length,checks:evidence.checks}));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
