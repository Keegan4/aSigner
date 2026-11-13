const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch();
  for (const name of process.argv.slice(2)) {
    const svg = fs.readFileSync(name + '.svg', 'utf8');
    const w = +svg.match(/width="(\d+)"/)[1], h = +svg.match(/height="(\d+)"/)[1];
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 3 });
    await p.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    await p.screenshot({ path: name + '.png', clip: { x: 0, y: 0, width: w, height: h } });
    await p.pdf({ path: name + '.pdf', width: w + 'px', height: (h + 1) + 'px', printBackground: true, pageRanges: '1' });
    await p.close();
  }
  await b.close();
})();
