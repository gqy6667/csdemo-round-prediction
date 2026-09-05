// T11: replay the delivery guide against an owned server and capture actual UI.
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'reports/roundcast_interactive_v1');
const playwright = require(process.env.ROUNDCAST_PLAYWRIGHT || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const prior = JSON.parse(fs.readFileSync(path.join(output, 't10-api-evidence.json'), 'utf8'));
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex');
const files = Object.keys(prior.hashes_after);
const codeFiles = ['web/roundcast/index.html', 'web/roundcast/app.js', 'web/roundcast/styles.css', 'web/roundcast/chat.js', 'src/csdemo/roundcast_service.py', 'src/csdemo/roundcast_server.py'];
const hashes = names => Object.fromEntries(names.map(file => [file, sha(file)]));
let server;
async function start(port = 0) {
  server = spawn(process.env.ROUNDCAST_PYTHON || path.join(process.env.USERPROFILE, '11/envs/game/python.exe'), ['-m', 'src.csdemo.roundcast_server', '--host', '127.0.0.1', '--port', String(port)], {cwd: root, windowsHide: true});
  return new Promise((resolve, reject) => {
    let stdout = '';
    const timer = setTimeout(() => reject(Error('Service startup timed out')), 15000);
    server.once('error', error => {clearTimeout(timer); reject(error);});
    server.once('exit', code => {clearTimeout(timer); reject(Error(`Service exited: ${code}`));});
    server.stdout.on('data', chunk => {stdout += chunk; const match = stdout.match(/http:\/\/127\.0\.0\.1:\d+/); if (match) {clearTimeout(timer); resolve(match[0]);}});
  });
}
async function stop() {
  if (server && server.exitCode === null) {const exited = once(server, 'exit'); server.kill(); await exited;}
}
(async () => {
  let browser;
  const evidence = {status: 'running', recorded_at: new Date().toISOString(), matrix: [], screenshots: [], pageErrors: [], consoleErrors: [], source_hashes_before: hashes(files), code_hashes_before: hashes(codeFiles)};
  assert.deepEqual(evidence.source_hashes_before, prior.hashes_after);
  try {
    const url = await start();
    browser = await playwright.chromium.launch({executablePath: process.env.ROUNDCAST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => evidence.pageErrors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') evidence.consoleErrors.push(message.text());});
    const writes = [], pending = [], results = new Map();
    page.on('request', request => {if (request.method() === 'POST') writes.push(new URL(request.url()).pathname);});
    page.on('response', response => {if (response.url().endsWith('/api/predict') && response.status() === 200) pending.push(response.json().then(body => results.set(`${body.example_id}/${body.stage}/${body.algorithm}`, body)));});
    const ready = () => page.waitForFunction(() => !document.querySelector('#run').disabled);
    const view = async name => {await page.locator(`[data-view="${name}"]`).click(); await page.locator(`#${name}-view`).waitFor({state: 'visible'});};
    await page.goto(url); await ready();
    assert.equal(await page.locator('#case-select').inputValue(), 'A');
    assert.equal(await page.locator('#ct-probability').textContent(), '—');
    assert.equal(writes.length, 0);
    for (const example of ['A', 'B', 'C']) {
      await page.selectOption('#case-select', example); await ready();
      const before = writes.length;
      await page.locator('#run-all').click();
      await page.locator('#comparison-status').filter({hasText: '4/4 成功'}).waitFor(); await ready(); await Promise.all(pending);
      assert.equal(writes.length - before, 4);
      for (const reference of prior.records.filter(row => row.request.example_id === example)) {
        const {stage, algorithm} = reference.request;
        const result = results.get(`${example}/${stage}/${algorithm}`);
        assert(Math.abs(result.prediction.ct_win_probability - reference.reference) <= 1e-8);
        evidence.matrix.push({example_id: example, stage, algorithm, ct_probability: result.prediction.ct_win_probability, request_id: result.request_id, absolute_error: Math.abs(result.prediction.ct_win_probability - reference.reference), identity: result.identity});
      }
      await page.selectOption('#stage-select', 'post_first_kill'); await ready();
      await page.selectOption('#algorithm-select', 'lightgbm'); await ready();
      const requestId = results.get(`${example}/post_first_kill/lightgbm`).request_id;
      const beforeViews = writes.length;
      for (const name of ['viewer', 'analyst', 'technical']) {
        await view(name);
        assert.equal(await page.locator('#request-id').textContent(), requestId);
        assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'), '');
        if (name === 'technical') await page.waitForFunction(() => document.querySelector('#metrics-status').dataset.status === 'success');
        if (example === 'B') for (const width of [1440, 1280]) {
          await page.setViewportSize({width, height: 1050}); await page.evaluate(() => window.scrollTo(0, 0));
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
          const full = `t11-${name}-${width}.png`, focus = `t11-${name}-${width}-focus.png`;
          await page.screenshot({path: path.join(output, full), fullPage: true});
          await page.screenshot({path: path.join(output, `t11-${name}-${width}-screen.png`)});
          const section = await page.locator(`#${name}-view`).boundingBox();
          await page.screenshot({path: path.join(output, focus), clip: {x: 0, y: 0, width, height: Math.ceil(section.y + section.height + 24)}});
          evidence.screenshots.push({view: name, width, full, focus, example_id: example, stage: 'post_first_kill', algorithm: 'lightgbm', outcome_revealed: false, request_id: requestId});
        }
      }
      assert.equal(writes.length, beforeViews);
      await page.locator('#technical-reveal').click();
      await page.waitForFunction(() => document.querySelector('#technical-correctness').dataset.correct !== '');
      const winner = (await (await page.request.get(`${url}/api/examples/${example}/outcome`)).json()).winning_side;
      for (const name of ['viewer', 'analyst', 'technical']) {await view(name); assert((await page.locator(name === 'viewer' ? '#outcome' : name === 'analyst' ? '#analysis-outcome' : '#technical-outcome-text').textContent()).includes(winner));}
      for (const row of evidence.matrix.filter(row => row.example_id === example)) {row.winner = winner; row.correct = (row.ct_probability >= 0.5 ? 'CT' : 'T') === winner;}
    }
    await stop();
    const restarted = await start(Number(new URL(url).port)); assert.equal(restarted, url);
    await page.reload(); await ready();
    assert.equal(await page.locator('#technical-version').textContent(), '—');
    assert.equal(await page.locator('#technical-correctness').getAttribute('data-correct'), '');
    await page.locator('#run').click(); await page.waitForFunction(() => document.querySelector('#technical-status').dataset.status === 'success'); await Promise.all(pending);
    evidence.restart_request_id = results.get('C/post_first_kill/lightgbm').request_id;
    assert(!evidence.matrix.some(row => row.request_id === evidence.restart_request_id));
    assert(writes.every(route => route === '/api/predict'));
    assert.deepEqual(evidence.pageErrors, []); assert.deepEqual(evidence.consoleErrors, []);
    evidence.source_hashes_after = hashes(files); evidence.code_hashes_after = hashes(codeFiles);
    assert.deepEqual(evidence.source_hashes_after, evidence.source_hashes_before);
    assert.deepEqual(evidence.code_hashes_after, evidence.code_hashes_before);
    evidence.checks = ['fresh-start-and-empty-default', 'three-cases-twelve-real-predictions', 'three-views-share-request-and-hidden-outcome', 'six-desktop-layouts', 'explicit-reveal-matches-all-views', 'same-port-restart-and-fresh-inference', 'no-codex-message', 'source-and-application-hashes-unchanged'];
    evidence.status = 'passed'; fs.writeFileSync(path.join(output, 't11-delivery-evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
    console.log(JSON.stringify({status: evidence.status, cases: 3, predictions: evidence.matrix.length, screenshots: evidence.screenshots.length, checks: evidence.checks}));
  } finally {if (browser) await browser.close(); await stop();}
})().catch(error => {console.error(error); process.exitCode = 1;});
