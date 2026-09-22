// Focused UX-01/02 browser regressions against the delivery build.
// Run: mise run build:frontend && node scripts/duplicates_ui_test.mjs
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { chromium } from 'playwright';
import { validateAtelierBuild } from './validate_atelier_build.mjs';

const dist = new URL('../internal/webui/atelier/dist/', import.meta.url).pathname;
await validateAtelierBuild(dist);
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  try {
    const file = path.startsWith('/assets/') ? path.slice(1) : 'index.html';
    res.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css' })[extname(file)]);
    res.end(await readFile(join(dist, file)));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/?view=duplicates`;
const browser = await chromium.launch({ headless: true });
const image = (id) => ({ image_id: id, original_name: `match-${id}.jpg`, storage_path: `images/${id}.jpg`, mime_type: 'image/jpeg', width: 450, height: 424, file_size_bytes: 2048, index_state: 'done' });
try {
  for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage({ viewport });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const requests = [];
    let conflict = false, failConfirm = false, removed = false;
    await page.route('**/api/**', async route => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      let body = {};
      if (path === '/api/duplicates') {
        if (request.method() === 'POST') {
          const data = request.postDataJSON(); requests.push(data);
          if (data.confirm && failConfirm) return route.fulfill({ status: 409, json: { error: 'preview changed; review again' } });
          if (data.confirm) removed = true;
          body = { keeper_id: data.keeper_id, removed_ids: data.group_ids.filter(id => id !== data.keeper_id), affected_collections: 2, file_policy: 'retained', conflicts: conflict ? ['Conflicting manual titles: Cat / Dog. Edit these images to agree before cleanup.'] : [], preview_token: 'reviewed-token' };
        } else body = { groups: removed ? [] : [{ items: [image(1), image(2), image(3)] }], max_distance: 4, scanned: 3, unhashed: 0 };
      } else if (path === '/api/stats') body = { images_total: 3, standalone_images_total: 3, videos_total: 0, jobs: {} };
      else if (path === '/api/images') body = { images: [], total: 0 };
      else if (path === '/api/videos') body = { videos: [], total: 0 };
      else if (path === '/api/collections') body = { collections: [] };
      else if (path === '/api/search/tag-cloud') body = { tags: [] };
      else if (path === '/api/live') return route.abort();
      else if (path === '/api/settings') return route.fulfill({ status: 503, json: { error: 'not exposed' } });
      return route.fulfill({ json: body });
    });
    await page.route('**/media/**', route => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="450" height="424"><rect width="450" height="424" fill="tan"/></svg>' }));
    await page.goto(url);
    await page.locator('[data-duplicate-group]').waitFor();
    assert.equal(await page.locator('option[value="0"]').textContent(), 'Identical visual hash');
    assert.match(await page.locator('[data-duplicates-pane]').innerText(), /dHash.*not byte.*equality/is);
    assert.match(await page.locator('[data-duplicate-group]').innerText(), /Possible matches/);
    assert.equal(await page.getByText('Same resolution', { exact: true }).count(), 3);
    assert.match(await page.locator('[data-duplicate-item="1"]').innerText(), /JPEG.*2.*KiB/);
    // Native radio keeper selection works with keyboard, not only pointer taps.
    const alternate = page.getByRole('radio', { name: 'Keep match-2.jpg', exact: true });
    await alternate.focus(); await page.keyboard.press('Space');
    assert.equal(await alternate.isChecked(), true);
    assert.equal(await page.locator('[data-duplicate-item="2"] [data-keep-badge]').innerText(), 'Keep');
    // Full originals, uncropped two-up at both viewport sizes; flip is optional.
    const compare = page.locator('[data-duplicate-compare]');
    assert.equal(await compare.locator('img').count(), 2);
    const boxes = await compare.locator('figure').evaluateAll(nodes => nodes.map(node => { const b = node.getBoundingClientRect(); return { x:b.x,y:b.y,width:b.width }; }));
    assert.ok(Math.abs(boxes[0].y - boxes[1].y) < 2 && boxes[1].x > boxes[0].x, 'two-up stays side-by-side on phones');
    assert.equal(await compare.locator('img').first().evaluate(node => getComputedStyle(node).objectFit), 'contain');
    await page.getByRole('button', { name: 'Flip comparison' }).click();
    assert.equal(await compare.locator('img').count(), 1);
    const beforeFlip = await compare.locator('img').getAttribute('src');
    await page.getByRole('button', { name: /Show (keeper|comparison)/ }).click();
    assert.notEqual(await compare.locator('img').getAttribute('src'), beforeFlip);
    await page.getByRole('button', { name: 'Two-up comparison' }).click();
    assert.equal(await compare.locator('img').count(), 2);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
    // Conflicts are visible before a destructive confirmation is available.
    conflict = true;
    await page.getByRole('button', { name: 'Review consolidation' }).click();
    await page.locator('[data-duplicate-conflicts]').waitFor();
    assert.match(await page.locator('[data-duplicate-conflicts]').innerText(), /Cat \/ Dog/);
    assert.equal(await page.locator('[data-duplicate-conflicts]').evaluate(node => node === document.activeElement), true, 'conflicts receive focus even when the group was below the fold');
    assert.equal(await page.locator('[data-confirm-accept]').count(), 0);
    assert.equal(requests.some(r => r.confirm), false);
    conflict = false;
    await page.getByRole('button', { name: 'Review consolidation' }).click();
    await page.locator('[data-confirm-dialog]').waitFor();
    const text = await page.locator('[data-confirm-dialog]').innerText();
    assert.match(text, /match-2.jpg/); assert.match(text, /2.*library entries/); assert.match(text, /2.*collections/); assert.match(text, /files.*retained/i); assert.match(text, /not.*free.*disk/i);
    await page.locator('[data-confirm-cancel]').click();
    assert.equal(requests.some(r => r.confirm), false);
    await page.getByRole('button', { name: 'Review consolidation' }).click();
    failConfirm = true;
    await page.locator('[data-confirm-accept]').click();
    await page.locator('[data-duplicates-error]').waitFor();
    assert.equal(await page.locator('[data-duplicate-item]').count(), 3, 'failure must not remove any items');
    failConfirm = false;
    await page.getByRole('button', { name: 'Review consolidation' }).click();
    await page.locator('[data-confirm-accept]').click();
    await page.locator('[data-duplicates-empty]').waitFor();
    const last = requests.at(-1);
    assert.equal(last.keeper_id, 2); assert.equal(last.preview_token, 'reviewed-token'); assert.equal(last.confirm, true); assert.deepEqual(last.group_ids, [1,2,3]);
    assert.deepEqual(errors, []);
    console.log(`duplicates desktop/mobile ${viewport.width}: PASS`);
    await page.close();
  }
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
