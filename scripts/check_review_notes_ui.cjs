// Offline browser verification using real saved content and current static assets.
// No live backend, provider requests, or existing database writes.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const input = process.argv[2];
  const output = process.argv[3];
  if (!input || !output) throw new Error('Usage: node script saved-result.json new-output-directory');
  fs.mkdirSync(output); // Never overwrite a previous check.
  const data = JSON.parse(fs.readFileSync(input, 'utf8')).result;
  const originals = data.run.metadata.reader_report.unverified_limitations || [];
  // UI fixture mirrors the publication schema; backend serialization has its own tests.
  data.final_answer.review_notes = [...new Set(originals)].map(text => ({
    text, status: 'UNVERIFIED_MODEL_NOTE', origin: 'withheld_limitation'
  }));
  const attack = '<img src=x onerror="window.__noteExecuted=true">';
  data.final_answer.review_notes.push({text: attack, status: 'UNVERIFIED_MODEL_NOTE'});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({viewport: {width, height: 1000}});
      await page.route('**/*', async route => {
        const pathname = new URL(route.request().url()).pathname;
        const asset = {'/': 'index.html', '/assets/app.js': 'app.js', '/assets/styles.css': 'styles.css'}[pathname];
        if (asset) return route.fulfill({body: fs.readFileSync(path.join(root, 'src/paperpilot_agentic/web_static', asset)),
          contentType: asset.endsWith('.js') ? 'text/javascript' : asset.endsWith('.css') ? 'text/css' : 'text/html'});
        if (pathname === '/api/bootstrap') return route.fulfill({json: {}});
        if (pathname === '/api/agentic/runs') return route.fulfill({json: {runs: []}});
        return route.abort();
      });
      await page.goto('http://paperpilot.test/');
      await page.evaluate(value => {
        state.thread = [value];
        state.current = value;
        state.newConversation = false;
        renderConversation([value]);
        renderRun(value);
      }, data);
      const details = page.locator('.message-agent details').filter({has: page.locator('summary', {hasText: '不作为事实结论'})});
      assert.equal(await page.locator('#empty-state').isVisible(), false);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      assert.equal(await details.count(), 1);
      assert.equal(await details.getAttribute('open'), null);
      assert.match(await details.locator('summary').innerText(), /不作为事实结论/);
      await details.locator('summary').click();
      assert.notEqual(await details.getAttribute('open'), null);
      assert.match(await details.innerText(), /exclusively English/);
      assert.equal(await details.locator('img').count(), 0);
      assert.equal(await page.evaluate(() => window.__noteExecuted === true), false);
      assert.equal(await page.locator('.message-agent .message-body').innerText(), data.final_answer.answer);
      await details.scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(output, `review-notes-${width}.png`), fullPage: true});
      results.push({width, expanded: true, original_answer_unchanged: true, html_executed: false});
      await page.close();
    }
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify({scope: 'offline UI fixture, not live API',
      provider_requests: 0, results}, null, 2), {flag: 'wx'});
    process.stdout.write(JSON.stringify(results));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
