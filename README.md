# AI 四棒接力

一段约 45 秒的撞色动画短片，讲我的 AI 工作流：

1. 一个想法的聊天气泡升起
2. Claude、Gemini、Codex、OpenRouter 依次从气泡里跳出来：我来规划、我来调研、我来执行、我来拓展
3. 镜头冲进图标，依次看四只宠物在干什么：Claude Code 的 Clawd（按终端启动画面的方块字符图还原）在写 PLAN.md；Gemini 的星星拿放大镜核对来源；Codex Pets 风格的像素海狸写代码，状态气泡从红钟“卡住了”变成绿勾“完成”；OpenRouter 的章鱼把线插进一排模型。Gemini 和 OpenRouter 没有官方吉祥物，这两只是自己设计的
4. 拉远成四色格子，然后三张订阅账单砸进屏幕，画面变红，钱包吞下账单后爆炸

浏览器打开 `index.html` 即可播放（动画用 cdnjs 上的 GSAP 3.12.5）。播放条可以暂停、拖动进度、全屏；键盘空格暂停，R 从头播放。

账单截图放在 `assets/receipts/`（订单号已打码），没有提交进仓库。缺少截图时，账单会显示成文字卡片。

## 导出 MP4

```sh
npm i playwright@1.56.1   # 需要配套的 Chromium，系统里还要有 ffmpeg
node scripts/render-mp4.js ai-relay.mp4            # 1920x1080，30fps
node scripts/render-mp4.js ai-relay.mp4 --fps 60
node scripts/render-mp4.js shot.mp4 --frames 15,24.5   # 只导出这几秒的截图，检查画面用
```

脚本用 `index.html#render` 打开页面，逐帧调用 `window.__render.seek(t)` 定位后截图，再交给 ffmpeg 编码成 H.264。访问不到 cdnjs 时，加 `--gsap 本地的gsap.min.js`。
