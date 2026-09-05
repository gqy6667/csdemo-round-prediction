// Local-only integration check; uses an existing Playwright and browser installation.
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const playwright = require(process.env.ROUNDCAST_PLAYWRIGHT || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const python = process.env.ROUNDCAST_PYTHON || path.join(process.env.USERPROFILE, '11/envs/game/python.exe');
const browserPath = process.env.ROUNDCAST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const output = path.join(root, 'reports/roundcast_interactive_v1');

(async () => {
  const server = spawn(python, ['-m', 'src.csdemo.roundcast_server', '--port', '0'], { cwd: root, windowsHide: true });
  let browser;
  const errors = [], predictions = [], requestBodies = [], outcomes = [];
  try {
    const url = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(Error('Server startup timeout')), 15000);
      server.stdout.on('data', data => { const match = String(data).match(/http:\/\/127\.0\.0\.1:\d+/); if (match) { clearTimeout(timeout); resolve(match[0]); } });
      server.once('error', reject); server.once('exit', code => { if (code) reject(Error('Server startup failed')); });
    });
    browser = await playwright.chromium.launch({ executablePath: browserPath, headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    page.setDefaultTimeout(6000);
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
    page.on('request', req => { if (req.url().endsWith('/api/predict')) requestBodies.push(req.postDataJSON()); });
    page.on('response', async res => {
      if (res.url().endsWith('/api/predict') && res.ok()) predictions.push(await res.json());
      if (res.url().endsWith('/outcome') && res.ok()) outcomes.push(await res.json());
    });
    await page.goto(url);
    await page.locator('#status[data-status="idle"]').waitFor();
    assert.equal(await page.locator('#ct-probability').textContent(), '—');
    assert.equal(outcomes.length, 0); assert.equal(predictions.length, 0);
    await page.locator('#run').click();
    await page.locator('#status[data-status="success"]').waitFor();
    assert.equal(await page.locator('#ct-probability').textContent(), '62.92%');
    assert.equal(await page.locator('#t-probability').textContent(), '37.08%');
    assert(Math.abs(predictions[0].prediction.ct_win_probability - 0.6291529536247253) <= 1e-8);
    const barPercent = await page.locator('#ct-fill').evaluate(el => parseFloat(el.style.width));
    const rawPercent = predictions[0].prediction.ct_win_probability * 100;
    // Chromium serializes CSS percentages to fewer digits; model tolerance stays 1e-8.
    assert(Math.abs(barPercent - rawPercent) < 1e-4, `CSS serialized width=${barPercent}, raw API percent=${rawPercent}`);
    const firstID = await page.locator('#request-id').textContent();
    await page.locator('#reveal').click(); await page.locator('#outcome').filter({ hasText: 'CT 获胜' }).waitFor();
    assert.equal(outcomes.length, 1);
    for (const width of [1440, 1280]) {
      await page.setViewportSize({ width, height: 1100 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.screenshot({ path: path.join(output, `t04-viewer-${width}.png`), fullPage: true });
    }
    let release;
    const gate = new Promise(resolve => release = resolve);
    await page.route('**/api/predict', async route => { await gate; await route.continue(); });
    await page.locator('#run').click();
    assert.equal(await page.locator('#run').isDisabled(), true);
    assert.equal(await page.locator('#ct-probability').textContent(), '—');
    await page.locator('#run').evaluate(button => { button.click(); button.click(); });
    release(); await page.locator('#status[data-status="success"]').waitFor();
    assert.equal(requestBodies.length, 2); assert.notEqual(await page.locator('#request-id').textContent(), firstID);
    await page.unroute('**/api/predict');
    const normalErrors = [...errors]; assert.deepEqual(normalErrors, []);
    const closed = once(server, 'exit'); server.kill(); await closed;
    await page.locator('#run').click(); await page.locator('#status[data-status="error"]').waitFor();
    assert.equal(await page.locator('#ct-probability').textContent(), '—');
    await page.locator('#reveal').click(); await page.locator('#outcome').filter({ hasText: '无法获取赛果' }).waitFor();
    await page.screenshot({ path: path.join(output, 't04-disconnected.png'), fullPage: true });
    const evidence = { status: 'passed', requestBodies, predictions, outcomes,
      checks: ['initial-no-prediction-or-outcome', 'real-click-inference', 'bar-and-number-same-response', 'explicit-outcome',
        '1440-and-1280-no-overflow', 'duplicate-click-guard', 'repeat-run-new-request', 'actual-server-stop-no-stale-result', 'outcome-offline-error'],
      normalConsoleErrors: normalErrors, expectedDisconnectConsoleErrors: errors.slice(normalErrors.length) };
    fs.writeFileSync(path.join(output, 't04-browser-evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', checks: evidence.checks, actualCT: predictions[0].prediction.ct_win_probability }));
  } finally { if (browser) await browser.close(); if (server.exitCode === null) server.kill(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
