// T10 controlled browser faults on an owned temporary server, never the user's preview.
const {spawn}=require('node:child_process'),{once}=require('node:events');
const path=require('node:path'),fs=require('node:fs'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),output=path.join(root,'reports/roundcast_interactive_v1');
const playwright=require(process.env.ROUNDCAST_PLAYWRIGHT||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
let server;
async function start(port=0){
  server=spawn(process.env.ROUNDCAST_PYTHON||path.join(process.env.USERPROFILE,'11/envs/game/python.exe'),['-m','src.csdemo.roundcast_server','--port',String(port)],{cwd:root,windowsHide:true});
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Temporary service startup timeout')),15000);
    server.once('error',e=>{clearTimeout(timer);reject(e);});
    server.once('exit',code=>{clearTimeout(timer);reject(Error(`Temporary service exited: ${code}`));});
    server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timer);resolve(match[0]);}});
  });
}
async function stop(){if(server&&server.exitCode===null){const exited=once(server,'exit');server.kill();await exited;}}
(async()=>{
  let browser;const evidence={status:'running',checks:[],pageErrors:[],consoleErrors:[],requestFailures:[]};let phase='normal';
  try{
    const url=await start();browser=await playwright.chromium.launch({executablePath:process.env.ROUNDCAST_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1050}});page.setDefaultTimeout(15000);
    page.on('pageerror',e=>evidence.pageErrors.push(e.message));
    page.on('console',m=>{if(m.type()==='error')evidence.consoleErrors.push({phase,message:m.text()});});
    page.on('requestfailed',r=>evidence.requestFailures.push({phase,path:new URL(r.url()).pathname,error:r.failure()?.errorText}));
    const writes=[];page.on('request',r=>{if(r.method()==='POST')writes.push({path:new URL(r.url()).pathname,body:r.postDataJSON()});});
    const ready=()=>page.waitForFunction(()=>!document.querySelector('#run').disabled);
    const failed=()=>page.waitForFunction(()=>document.querySelector('#status').dataset.status==='error');
    const view=async name=>{await page.locator(`[data-view="${name}"]`).click();await page.locator(`#${name}-view`).waitFor({state:'visible'});};
    await page.goto(url);await ready();await view('technical');
    assert.equal(writes.length,0);
    // Real response with exactly one deliberately removed field, never a fake winning probability.
    phase='malformed-prediction';
    for(const field of ['inference_ms','validation','model_sha256']){
      await page.route('**/api/predict',async route=>{const response=await route.fetch(),body=await response.json();delete body[field];await route.fulfill({response,json:body});});
      await page.locator('#run').click();await failed();
      assert.equal(await page.locator('#technical-version').textContent(),'—');assert.equal(await page.locator('#ct-probability').textContent(),'—');
      assert.equal(await page.locator('#analysis-chart .chart-point').count(),0);await page.unroute('**/api/predict');
    }
    evidence.checks.push('incomplete-success-never-renders-success-probability-or-runtime');
    phase='catalog-reconnect';let release,hit;const gate=new Promise(r=>release=r),started=new Promise(r=>hit=r);
    await page.route('**/api/models',async route=>{const response=await route.fetch(),body=await response.json();hit();await gate;body.models.forEach(m=>m.inference_ready=false);await route.fulfill({response,json:body});});
    await page.locator('#reload').click();await started;assert.equal(await page.locator('#case-select').isDisabled(),true);
    const before=writes.length;await view('analyst');await view('technical');
    await page.locator('#case-select').evaluate(el=>{el.value='B';el.dispatchEvent(new Event('change',{bubbles:true}));});
    assert.equal(await page.locator('#run').isDisabled(),true);release();await failed();
    assert.equal(await page.locator('#case-select').inputValue(),'A');assert.equal(writes.length,before);
    await page.unroute('**/api/models');await page.locator('#reload').click();await ready();
    evidence.checks.push('reconnect-invalidates-old-catalog-and-blocks-case-change-until-ready');
    phase='catalog-failure';
    await page.route('**/api/models',route=>route.fulfill({status:503,json:{message:'受控测试：清单不可用'}}));
    await page.goto(url);await failed();await view('technical');
    await page.locator('#metrics-status[data-status="success"]').waitFor();
    assert.equal(await page.locator('#run').isDisabled(),true);await page.unroute('**/api/models');
    await page.locator('#reload').click();await ready();
    evidence.checks.push('independent-technical-metrics-remain-navigable-when-catalog-fails');
    // Rapid case switch and repeated clicks while a real old-case request is delayed.
    phase='late-response';let releaseRun,runHit;const runGate=new Promise(r=>releaseRun=r),runStarted=new Promise(r=>runHit=r);
    await page.route('**/api/predict',async route=>{runHit();await runGate;await route.continue();});
    const count=writes.length;await page.locator('#run-all').click();await runStarted;
    await page.locator('#run-all').evaluate(el=>{el.click();el.click();});
    await page.selectOption('#case-select','B');await ready();await view('viewer');await view('analyst');
    const oldResponse=page.waitForResponse(r=>r.url().endsWith('/api/predict'));releaseRun();await (await oldResponse).finished();
    await page.unroute('**/api/predict');await view('technical');
    assert.equal(writes.length,count+1);assert.equal(await page.locator('#ct-probability').textContent(),'—');
    assert.equal(await page.locator('#analysis-chart .chart-point').count(),0);assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    evidence.checks.push('duplicate-clicks-and-case-switch-do-not-publish-or-continue-old-batch');
    phase='late-outcome';let revealHit,releaseReveal;const revealStarted=new Promise(r=>revealHit=r),revealGate=new Promise(r=>releaseReveal=r);
    await page.route('**/outcome',async route=>{revealHit();await revealGate;await route.continue();});
    await page.locator('#technical-reveal').click();await revealStarted;
    await page.selectOption('#case-select','C');await ready();
    const oldOutcome=page.waitForResponse(r=>r.url().endsWith('/outcome'));releaseReveal();await (await oldOutcome).finished();
    await page.unroute('**/outcome');
    for(const name of ['viewer','analyst','technical']){
      await view(name);assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
      assert(!(await page.locator('#outcome').textContent()).includes('获胜 ·'));
    }
    evidence.checks.push('late-outcome-from-old-case-does-not-reveal-current-case');
    // Stop only the temporary child we created, then restart on its exact prior port.
    phase='before-outage';await page.locator('#run-all').click();
    await page.locator('#comparison-status').filter({hasText:'4/4 成功'}).waitFor();
    const priorRequest=await page.locator('#request-id').textContent();
    await page.locator('#technical-reveal').click();await page.waitForFunction(()=>document.querySelector('#technical-correctness').dataset.correct!=='');
    phase='actual-service-stopped';await stop();
    await page.locator('#run').click();await failed();
    assert.equal(await page.locator('#ct-probability').textContent(),'—');assert.equal(await page.locator('#technical-version').textContent(),'—');
    assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    await page.locator('#technical-reveal').click();await page.locator('#outcome').filter({hasText:'无法获取赛果'}).waitFor({state:'attached'});
    await page.locator('#metrics-reload').click();await page.locator('#metrics-status[data-status="error"]').waitFor();
    assert((await page.locator('#technical-metrics dd').allTextContents()).every(value=>value==='—'));
    const shot='t10-actual-service-stopped.png';await page.screenshot({path:path.join(output,shot),fullPage:true});
    evidence.outageScreenshot=shot;
    phase='recovered';assert.equal(await start(Number(new URL(url).port)),url);
    await page.locator('#reload').click();await ready();await page.locator('#metrics-reload').click();
    await page.locator('#metrics-status[data-status="success"]').waitFor();
    assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'),'');
    await page.locator('#run').click();await page.locator('#status[data-status="success"]').waitFor({state:'attached'});
    assert.notEqual(await page.locator('#request-id').textContent(),priorRequest);
    evidence.checks.push('actual-process-stop-clears-current-prediction-outcome-and-metrics','same-port-restart-restores-real-inference-with-new-request');
    assert.deepEqual(evidence.pageErrors,[]);assert(writes.every(r=>r.path==='/api/predict'));
    assert.equal(evidence.consoleErrors.filter(r=>['normal','before-outage','recovered'].includes(r.phase)).length,0);
    evidence.status='passed';fs.writeFileSync(path.join(output,'t10-fault-browser-evidence.json'),JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify({status:evidence.status,checks:evidence.checks}));
  }finally{if(browser)await browser.close();await stop();}
})().catch(error=>{console.error(error);process.exitCode=1;});
