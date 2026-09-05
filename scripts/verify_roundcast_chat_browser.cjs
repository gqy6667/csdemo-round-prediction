// Explicit opt-in live smoke. The local server must already be running and logged into Codex.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const playwright = require(process.env.ROUNDCAST_PLAYWRIGHT || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
if (!process.argv.includes('--live')) throw Error('Use --live to explicitly send one test question to Codex.');

(async () => {
  const browser = await playwright.chromium.launch({ executablePath: process.env.ROUNDCAST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const evidence = { status: 'running', mode: 'real-codex-no-mocks', checks: [], errors: [] };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    page.on('pageerror', err => evidence.errors.push(err.message));
    const sent = [];
    page.on('request', request => { if (request.url().endsWith('/api/chat')) sent.push(request.postDataJSON()); });
    await page.goto('http://127.0.0.1:8765/');
    await page.locator('#status[data-status="idle"]').waitFor();
    await page.locator('#chat-status[data-status="success"]').waitFor();
    await page.locator('[data-chat-question]').first().click();
    assert.equal(sent.length, 0);
    evidence.checks.push('quick-question-does-not-send');
    await page.locator('#run').click();
    await page.locator('#status[data-status="success"]').waitFor();
    const firstRequest = await page.locator('#request-id').textContent();
    await page.locator('#chat-input').fill('请只用两句话回答：当前地图、回合、算法及 CT/T 胜率是什么？能否据此确定实际赛果？不要使用工具。');
    await page.locator('#chat-send').click();
    await page.locator('#chat-progress').filter({ hasText: '正在思考' }).waitFor();
    assert.equal(await page.locator('#chat-send').isDisabled(), true);
    await page.locator('#chat-send').evaluate(button => { button.click(); button.click(); });
    assert.equal(sent.length, 1);
    assert.equal(sent[0].request_id, firstRequest);
    assert.equal(Object.keys(sent[0]).length, 6);
    evidence.checks.push('single-send-bound-to-trusted-prediction');
    console.log('Real question sent; waiting for Codex while checking independent model run.');
    await page.locator('#run').click();
    await page.locator('#status[data-status="success"]').waitFor();
    assert.notEqual(await page.locator('#request-id').textContent(), firstRequest);
    evidence.checks.push('model-can-rerun-during-chat');
    await page.waitForFunction(() => document.querySelector('.chat-assistant') || document.querySelector('#chat-error').textContent, null, { timeout: 165000 });
    const error = await page.locator('#chat-error').textContent();
    assert.equal(error, '', error);
    const response = await page.locator('.chat-assistant .chat-message-content').textContent();
    assert(response.includes('62.92') && response.includes('37.08'), response);
    assert((await page.locator('.chat-assistant .chat-message-header').textContent()).includes(firstRequest.slice(0, 8)));
    evidence.reply = response;
    evidence.selection = sent[0];
    evidence.checks.push('real-answer-matches-model-probabilities', 'answer-retains-original-context');
    await page.locator('#chat-title').scrollIntoViewIfNeeded();
    const output = path.join(root, 'reports/roundcast_interactive_v1');
    await page.screenshot({ path: path.join(output, 'chat-live-desktop.png'), fullPage: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    }
    await page.screenshot({ path: path.join(output, 'chat-live-mobile.png'), fullPage: true });
    evidence.checks.push('desktop-mobile-no-overflow');
    assert.equal(evidence.errors.length, 0);
    await page.locator('#chat-clear').click();
    assert.equal(await page.locator('#chat-messages article').count(), 0);
    evidence.checks.push('clear-local-transcript');
    evidence.status = 'passed';
    fs.writeFileSync(path.join(output, 'chat-live-browser-evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
    console.log(JSON.stringify(evidence));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
