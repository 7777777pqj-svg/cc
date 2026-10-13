// 把 index.html 的动画逐帧渲染成 MP4。
//
// 用法：node scripts/render-mp4.js [输出.mp4] [--fps 30] [--width 1920] [--audio 配乐.wav] [--gsap 本地gsap.min.js] [--frames 0,3.5,12]
//   需要：playwright（带 Chromium）和系统里的 ffmpeg。
//   --audio 把配乐混进视频（AAC 256k），配乐由 music/make_music.py 生成
//   --gsap  网络访问不到 cdnjs 时，用本地的 GSAP 3.12.5 文件代替
//   --frames 只导出这几个时间点的 PNG 截图（检查画面用），不生成视频
//
// 原理：页面带 #render 打开时不会自动播放，而是暴露 window.__render.seek(t)，
// 把主时间线、宠物的常驻动作和 SVG 动画都定位到同一时刻，所以每一帧都是确定的。

const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');

const args = process.argv.slice(2);
function opt(name, def) {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : def;
}
const out = args[0] && !args[0].startsWith('--') ? args[0] : 'ai-relay.mp4';
const fps = Number(opt('fps', 30));
const width = Number(opt('width', 1920));
const height = Math.round(width * 9 / 16);
const gsapFile = opt('gsap');
const frameList = opt('frames');
const audio = opt('audio');
const page = 'file://' + path.resolve(__dirname, '..', 'index.html') + '#render';

(async () => {
  const browser = await chromium.launch();
  const p = await browser.newPage({ viewport: { width, height } });
  if (gsapFile) {
    await p.route('**/gsap.min.js', r => r.fulfill({ path: path.resolve(gsapFile), contentType: 'application/javascript' }));
  }
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  await p.goto(page, { waitUntil: 'load' });
  await p.waitForFunction(() => window.__render, null, { timeout: 15000 });
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(300);
  const duration = await p.evaluate(() => window.__render.duration);
  const clip = { x: 0, y: 0, width, height };

  if (frameList) {
    for (const t of frameList.split(',').map(Number)) {
      await p.evaluate(v => window.__render.seek(v), t);
      const file = out.replace(/\.mp4$/, '') + '-' + t.toFixed(1) + 's.png';
      await p.screenshot({ path: file, clip });
      console.log('frame', file);
    }
    await browser.close();
    if (errors.length) console.log('page errors:', errors);
    return;
  }

  const total = Math.ceil(duration * fps);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
    ...(audio ? ['-i', path.resolve(audio), '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '256k', '-shortest'] : []),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));

  const t0 = Date.now();
  for (let f = 0; f < total; f++) {
    await p.evaluate(v => window.__render.seek(v), f / fps);
    const buf = await p.screenshot({ type: 'jpeg', quality: 95, clip });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % (fps * 5) === 0) console.log(`frame ${f}/${total}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await done;
  await browser.close();
  if (errors.length) console.log('page errors:', errors);
  console.log(`done: ${out}  ${total} frames, ${duration.toFixed(1)}s @ ${fps}fps, ${width}x${height}`);
})().catch(e => { console.error(e); process.exit(1); });
