// T06: real frozen predictions. Mocks are limited to explicitly labelled failure/race checks.
const {spawn} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const playwright = require(process.env.ROUNDCAST_PLAYWRIGHT || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const references = JSON.parse(fs.readFileSync(path.join(root,'examples/roundcast_v1_cases.json'),'utf8')).reference_probabilities;
(async()=>{
  const server=spawn(process.env.ROUNDCAST_PYTHON||path.join(process.env.USERPROFILE,'11/envs/game/python.exe'),['-m','src.csdemo.roundcast_server','--port','0'],{cwd:root,windowsHide:true});
  let browser;
  const evidence={status:'running',predictions:[],checks:[],pageErrors:[],faultChecks:[]};
  try{
    const url=await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(Error('Server startup timeout')),15000);
      server.stdout.on('data',data=>{const match=String(data).match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timer);resolve(match[0]);}});
      server.on('error',reject);
    });
    browser=await playwright.chromium.launch({executablePath:process.env.ROUNDCAST_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1050}});
    page.setDefaultTimeout(12000);
    page.on('pageerror',e=>evidence.pageErrors.push(e.message));
    const requests=[];
    page.on('request',r=>{if(r.url().endsWith('/api/predict'))requests.push(r.postDataJSON());});
    page.on('response',async r=>{if(r.url().endsWith('/api/predict')&&r.ok())evidence.predictions.push(await r.json());});
    await page.goto(url);await page.locator('#status[data-status="idle"]').waitFor();
    assert.equal(await page.locator('#first-kill-panel').isHidden(),true);
    assert.equal(requests.length,0);
    for(const example of ['A','B','C']){
      await page.selectOption('#case-select',example);await page.locator('#status[data-status="idle"]').waitFor();
      assert.equal(await page.locator('#ct-probability').textContent(),'—');
      assert(!(await page.locator('#outcome').textContent()).includes('获胜 ·'));
      const before=requests.length;
      await page.locator('#run-all').click();
      await page.locator('#comparison-status').filter({hasText:'4/4 成功'}).waitFor();
      assert.equal(requests.length-before,4);
      assert.equal(new Set(requests.slice(before).map(s=>s.stage+'/'+s.algorithm)).size,4);
      assert(requests.slice(before).every(s=>s.example_id===example));
      for(const stage of ['pre_round','post_first_kill'])for(const algorithm of ['xgboost','lightgbm']){
        await page.selectOption('#stage-select',stage);await page.selectOption('#algorithm-select',algorithm);
        await page.locator('#status[data-status="success"]').waitFor();
        const key=`${algorithm==='xgboost'?'xgb':'lgbm'}_${stage}`;
        assert.equal(await page.locator('#ct-probability').textContent(),(references[example][key]*100).toFixed(2)+'%');
        assert.equal(await page.locator('#first-kill-panel').isHidden(),stage==='pre_round');
        assert((await page.locator('#validation').textContent()).includes(stage==='pre_round'?'27 项输入':'31 项输入'));
        assert((await page.locator('#chat-context').textContent()).includes(`案例 ${example}`));
        const row=page.locator(`tr[data-model="${key}"]`);
        assert.equal(await row.locator('td').nth(2).textContent(),await page.locator('#ct-probability').textContent());
      }
      await page.locator('#reveal').click();await page.locator('#outcome').filter({hasText:'获胜 · 真实历史赛果'}).waitFor();
    }
    assert.equal(evidence.predictions.length,12);
    assert.equal(new Set(evidence.predictions.map(r=>r.request_id)).size,12);
    for(const r of evidence.predictions)assert(Math.abs(r.prediction.ct_win_probability-references[r.example_id][r.model_id])<=1e-8);
    evidence.checks.push('all-12-real-browser-predictions-match-frozen-reference','case-change-clears-outcome','selected-hero-table-chat-match','stage-controls-hide-future-fields','real-27-31-field-counts');
    const output=path.join(root,'reports/roundcast_interactive_v1');
    for(const width of [1440,1280,390]){
      await page.setViewportSize({width,height:1100});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      await page.screenshot({path:path.join(output,`t06-viewer-${width}.png`),fullPage:true});
    }
    // Deliberately inject one HTTP failure, not a fabricated successful probability.
    await page.setViewportSize({width:1440,height:1050});
    await page.route('**/api/predict',async route=>{
      const s=route.request().postDataJSON();
      if(s.stage==='post_first_kill'&&s.algorithm==='lightgbm')return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({status:'error',message:'受控测试：模型不可用'})});
      return route.continue();
    });
    await page.locator('#run-all').click();
    await page.locator('tr[data-model="lgbm_post_first_kill"] .status[data-status="error"]').waitFor();
    assert.equal(await page.locator('#ct-probability').textContent(),'—');
    assert.equal(await page.locator('tr .status[data-status="success"]').count(),3);
    await page.screenshot({path:path.join(output,'t06-controlled-partial-failure.png'),fullPage:true});
    await page.unroute('**/api/predict');
    const retryCount=requests.length;
    await page.locator('tr[data-model="lgbm_post_first_kill"] .row-run').click();
    await page.locator('#status[data-status="success"]').waitFor();assert.equal(requests.length,retryCount+1);
    evidence.faultChecks.push('one-failure-keeps-three-successes','retry-only-failed-combination');
    let release;
    const gate=new Promise(r=>release=r);
    await page.route('**/api/predict',async route=>{await gate;await route.continue();});
    const oldCount=requests.length;
    await page.locator('#run-all').click();
    await page.locator('#run-all').evaluate(button=>{button.click();button.click();});
    await page.selectOption('#case-select','A');await page.locator('#status[data-status="idle"]').waitFor();
    release();await page.waitForResponse(r=>r.url().endsWith('/api/predict'));
    await page.unroute('**/api/predict');
    await page.locator('#timeline-pre').click();await page.locator('#status[data-status="idle"]').waitFor();
    assert.equal(await page.locator('#first-kill-panel').isHidden(),true);
    assert.equal(await page.locator('#ct-probability').textContent(),'—');
    assert.equal(await page.locator('tr .status[data-status="success"]').count(),0);
    assert.equal(requests.length,oldCount+1);
    evidence.faultChecks.push('duplicate-batch-guard','case-switch-invalidates-late-result-and-stops-old-batch');
    assert.deepEqual(evidence.pageErrors,[]);
    evidence.status='passed';evidence.requestBodies=requests;
    fs.writeFileSync(path.join(output,'t06-browser-evidence.json'),JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify({status:evidence.status,realMatrixCount:12,checks:evidence.checks,faultChecks:evidence.faultChecks}));
  }finally{if(browser)await browser.close();server.kill();}
})().catch(error=>{console.error(error);process.exitCode=1;});
