const pptxgen = require("pptxgenjs");

// ============ 色票（沿用 v9 暖調編輯風）============
const NAVY = "26221D";
const NAVY_2 = "332C24";
const ICE = "EDE3D6";
const GOLD = "C15B2A";
const WHITE = "FFFFFF";
const INK = "26221D";
const INK2 = "6B6157";
const MUTED = "9C9184";
const LIGHT_BG = "F6F1E9";
const CARD_BG = "FFFFFF";
const BORDER = "E4DACB";
const BULL = "A83A2C";
const BEAR = "3E6B4F";
const SAND = "C9BBA6";

const TITLE_FONT = "Georgia";
const BODY_FONT = "Microsoft JhengHei";
const FOOT = "整合技術面分析與市場情緒之股票交易資訊平台｜散戶情緒十年檢驗";

let PG = 0;
function newSlide(pres, bg) {
  const s = pres.addSlide();
  s.background = { color: bg || WHITE };
  s._pg = ++PG;
  return s;
}
function kicker(slide, text, color) {
  slide.addShape("rect", { x: 0.6, y: 0.42, w: 0.32, h: 0.06, fill: { color: color || GOLD }, line: { type: "none" } });
  slide.addText(text.toUpperCase(), {
    x: 1.05, y: 0.32, w: 8, h: 0.35, fontFace: BODY_FONT, fontSize: 11, bold: true, color: GOLD,
    charSpacing: 2, isTextBox: true, margin: 0,
  });
}
function pageTitle(slide, text, color) {
  slide.addText(text, {
    x: 0.6, y: 0.72, w: 12.1, h: 0.75, fontFace: TITLE_FONT, fontSize: 24, bold: true, color: color || INK,
    isTextBox: true, margin: 0, valign: "middle",
  });
}
function footer(slide, n, color) {
  slide.addText(FOOT, { x: 0.6, y: 7.1, w: 9, h: 0.3, fontFace: BODY_FONT, fontSize: 9, color: color || MUTED, isTextBox: true, margin: 0 });
  slide.addText(String(n || slide._pg), { x: 12.4, y: 7.1, w: 0.4, h: 0.3, fontFace: BODY_FONT, fontSize: 9, color: color || MUTED, align: "right", isTextBox: true, margin: 0 });
}
function statTile(slide, x, y, w, h, value, label, valueColor, valueSize) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: CARD_BG }, line: { color: BORDER, width: 1 } });
  slide.addText(value, {
    x: x + 0.18, y: y + 0.12, w: w - 0.36, h: h * 0.5, fontFace: TITLE_FONT, fontSize: valueSize || 24, bold: true,
    color: valueColor || INK, isTextBox: true, margin: 0, align: "left", valign: "bottom",
  });
  slide.addText(label, {
    x: x + 0.18, y: y + h * 0.58, w: w - 0.36, h: h * 0.38, fontFace: BODY_FONT, fontSize: 10.5, color: INK2,
    isTextBox: true, margin: 0, align: "left", valign: "top", lineSpacingMultiple: 1.15,
  });
}
function bulletList(slide, items, x, y, w, opts) {
  const lh = (opts && opts.lineHeight) || 0.58;
  const fontSize = (opts && opts.fontSize) || 12;
  const color = (opts && opts.color) || INK;
  items.forEach((it, i) => {
    const iy = y + i * lh;
    slide.addShape("ellipse", { x, y: iy + 0.09, w: 0.11, h: 0.11, fill: { color: (opts && opts.dot) || GOLD }, line: { type: "none" } });
    slide.addText(it, { x: x + 0.32, y: iy, w: w - 0.32, h: lh, fontFace: BODY_FONT, fontSize, color, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
  });
}
function callout(slide, x, y, w, h, text, tone) {
  const bg = tone === "warn" ? "FBEEEA" : tone === "good" ? "EAF1EC" : LIGHT_BG;
  const fg = tone === "warn" ? BULL : tone === "good" ? BEAR : INK;
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.06, fill: { color: bg }, line: { type: "none" } });
  slide.addText(text, { x: x + 0.25, y: y + 0.05, w: w - 0.5, h: h - 0.1, fontFace: BODY_FONT, fontSize: 11.5, color: fg, isTextBox: true, margin: 0, valign: "middle", lineSpacingMultiple: 1.25 });
}
// 淺色表格：表頭深色，內文交替底色
function table(slide, header, rows, x, y, colW, rowH, opts) {
  const fs = (opts && opts.fontSize) || 11;
  const head = header.map((h, i) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: fs + 0.5, align: i === 0 ? "left" : "center", valign: "middle" } }));
  const body = rows.map((r, ri) => r.map((c, i) => {
    let text = c, color = INK, bold = false;
    if (typeof c === "object") { text = c.t; color = c.color || INK; bold = !!c.bold; }
    return { text, options: { color, bold, fontSize: fs, align: i === 0 ? "left" : "center", valign: "middle", fill: { color: ri % 2 === 0 ? WHITE : LIGHT_BG } } };
  }));
  const headH = (opts && opts.headH) || 0.5;
  slide.addTable([head, ...body], {
    x, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH: [headH, ...rows.map(() => rowH)],
    border: { type: "solid", color: BORDER, pt: 1 }, autoPage: false,
  });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: TITLE_FONT, bodyFontFace: BODY_FONT };

// ============================================================ 1 標題
{
  const s = newSlide(pres, NAVY);
  s.addText("期中進度報告", { x: 0.9, y: 1.7, w: 8, h: 0.4, fontFace: BODY_FONT, fontSize: 13, bold: true, color: GOLD, charSpacing: 3, isTextBox: true, margin: 0 });
  s.addText("整合技術面分析與市場情緒之\n股票交易資訊平台", { x: 0.9, y: 2.15, w: 11.4, h: 1.9, fontFace: TITLE_FONT, fontSize: 34, bold: true, color: WHITE, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15 });
  s.addText("以十年台股資料檢驗「情緒」該怎麼放進平台｜2015–2025｜全市場 1,154 檔", { x: 0.9, y: 4.2, w: 11.6, h: 0.45, fontFace: BODY_FONT, fontSize: 15, color: ICE, isTextBox: true, margin: 0 });
  s.addShape("line", { x: 0.9, y: 4.8, w: 2.2, h: 0, line: { color: GOLD, width: 2 } });
  s.addText("結論：情緒、新聞與公告「方向說得對」，但市場反應得比讀者快，看到再進場賺不到；扣掉成本後還有用的，只有技術面加散戶實際交易行為、拉長到一個月的選股。情緒的價值在描述與提醒。", {
    x: 0.9, y: 5.0, w: 10.8, h: 1.0, fontFace: BODY_FONT, fontSize: 13, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.35,
  });
  s.addNotes("先講結論：我們把資料拉到十年、265檔，散戶情緒『能描述行情，但大多預測不了』。只有散戶實際的融資融券行為有穩定的小訊號。");
}

// ============================================================ S 起點：選讀論文與上次的概念驗證（v18 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "Where We Started");
  pageTitle(s, "起點：我選讀的論文，和上次概念驗證留下的後續工作");
  const card = (x, title, sub, lines, color) => {
    s.addShape("roundRect", { x, y: 1.6, w: 5.95, h: 3.75, rectRadius: 0.08, fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 } });
    s.addShape("rect", { x, y: 1.6, w: 5.95, h: 0.08, fill: { color }, line: { type: "none" } });
    s.addText(title, { x: x + 0.3, y: 1.8, w: 5.4, h: 0.45, fontFace: BODY_FONT, fontSize: 15, bold: true, color, isTextBox: true, margin: 0 });
    s.addText(sub, { x: x + 0.3, y: 2.25, w: 5.4, h: 0.35, fontFace: BODY_FONT, fontSize: 10.5, color: INK2, isTextBox: true, margin: 0 });
    bulletList(s, lines, x + 0.3, 2.75, 5.4, { fontSize: 11.5, lineHeight: 0.62 });
  };
  card(0.6, "選讀論文：Jing, Wu & Wang（2021）", "Expert Systems with Applications｜投資人情緒 × 深度學習預測股價", [
    "CNN 判斷東方財富論壇 88 萬則貼文的情緒，加上 16 項技術指標",
    "LSTM 預測隔天收盤價",
    "加入情緒後誤差（MAPE）0.0675 → 0.0449，降約三成",
    "→ 我們平台「技術面＋情緒雙軌整合」的依據",
  ], GOLD);
  card(6.75, "上次的概念驗證（7 月）", "台積電 2330｜PTT 133 篇、1.8 萬則留言｜35 個交易日", [
    "爬蟲 → 情緒詞典 → 技術指標，整條流程跑通",
    "動能指標（RSI、MOM）與情緒同向，相關約 0.2",
    "限制：樣本太少，不能談預測",
    "後續工作：資料拉長、用 LSTM 驗證情緒的預測力",
  ], BEAR);
  callout(s, 0.6, 5.6, 12.1, 1.25, "這次要回答的問題：論文說「加情緒能讓預測更準」——在台股、十年、全市場的資料下還成立嗎？如果不成立，平台上的情緒應該怎麼用？", "plain");
  footer(s);
  s.addNotes("先講起點。我選讀的論文是 Jing、Wu、Wang 2021 年發表在 Expert Systems with Applications 的研究，它用 CNN 判斷論壇貼文的情緒，再加上技術指標，用 LSTM 預測隔天收盤價，加了情緒之後誤差降了大約三成。這是我們平台想要『技術面加情緒雙軌整合』的依據。七月的概念驗證，我們用台積電三十五個交易日把爬蟲、情緒分析和技術指標整條流程跑通，看到動能指標和情緒同方向，但樣本太少，留下的後續工作就是把資料拉長、用模型驗證情緒到底有沒有預測力。這次的報告就是在回答這個問題。");
}

// ============================================================ 2 資料與問題
{
  const s = newSlide(pres, WHITE);
  kicker(s, "Data & Question");
  pageTitle(s, "研究問題與資料：散戶「說」的 vs 散戶「做」的");

  const tiles = [
    ["2,021 萬則", "PTT 股板留言（個股 231 萬則），2015–2025"],
    ["179 萬篇", "CMoney 同學會貼文，2020–2025（平台 2020 前無資料）"],
    ["59.7 萬則", "鉅亨網新聞 18.8 萬＋公司重大訊息 40.9 萬"],
    ["1,154 檔", "TEJ 全市場股價與籌碼（含下市）"],
  ];
  const tw = 2.85, gap = 0.25;
  tiles.forEach((t, i) => statTile(s, 0.6 + i * (tw + gap), 1.7, tw, 1.45, t[0], t[1], GOLD, 22));

  const cols = [
    ["散戶說的（PTT）", ["個股留言 → Breeze 三分類；每日淨情緒 (正−負)/(正+負)", "閒聊串（盤中／盤後閒聊）→ 大盤散戶溫度計與 8 類細分情緒", "PTT 新聞標題 8.3 萬篇 → 好／壞／無"]],
    ["散戶做的（融資融券）", ["融資＝散戶槓桿看多；融券＝看空（台灣文獻常用的散戶情緒指標）", "三大法人只當對照組，不當主軸", "另含美股隔夜報酬（只當對照，不是情緒）"]],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 6.15;
    s.addShape("roundRect", { x, y: 3.4, w: 5.95, h: 2.55, rectRadius: 0.08, fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 } });
    s.addText(c[0], { x: x + 0.3, y: 3.5, w: 5.4, h: 0.45, fontFace: BODY_FONT, fontSize: 14, bold: true, color: GOLD, isTextBox: true, margin: 0 });
    bulletList(s, c[1], x + 0.3, 4.05, 5.4, { fontSize: 11.5, lineHeight: 0.62 });
  });
  callout(s, 0.6, 6.15, 12.1, 0.75, "驗證方式：逐年滾動（測試年 2018–2024，用之前所有年份訓練），只用「當時已經知道」的資訊；統計檢定用按週/月的 bootstrap。股價與籌碼另以 TEJ 資料庫重跑驗證，並擴到全市場 1,144 檔。", "plain");
  footer(s);
  s.addNotes("兩條主線：散戶說什麼（PTT）、散戶做什麼（融資融券）。所有預測都是逐年往前滾，避免偷看未來。");
}

// ============================================================ C 結論四句話（v15 新增）
{
  const s = newSlide(pres, NAVY);
  kicker(s, "Conclusion", GOLD);
  pageTitle(s, "先講結論：四句話", WHITE);
  const pts = [
    ["1", "情緒指標都是真的", "PTT、CMoney、媒體新聞、公司公告、隔夜漲幅：方向都對、統計上都顯著，重要的都在沒看過的股票或年份上重現。"],
    ["2", "但市場比讀者快", "文字類訊號幾天內就反應完，多數在隔天開盤那一下；扣成本並剔除漲跌停買不到的之後，沒有一個能贏大盤。"],
    ["3", "散戶「做」的比「說」的有用", "融資融券在全市場、每一年都有小訊號；拉長到一個月和技術面一起選股，扣成本後選到的股票比一般股票好。"],
    ["4", "情緒的價值在描述與提醒", "恐慌指數在股災時準確升高；沒消息又沒鎖漲停的大漲之後平均回吐；恐慌時技術面較不可靠——適合做成平台提示。"],
  ];
  pts.forEach((c, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.65 + Math.floor(i / 2) * 2.45;
    s.addShape("roundRect", { x, y, w: 5.95, h: 2.25, rectRadius: 0.08, fill: { color: NAVY_2 }, line: { color: "4A4036", width: 1 } });
    s.addText(c[0], { x: x + 0.25, y: y + 0.2, w: 0.6, h: 0.6, fontFace: TITLE_FONT, fontSize: 28, bold: true, color: GOLD, isTextBox: true, margin: 0 });
    s.addText(c[1], { x: x + 0.95, y: y + 0.22, w: 4.8, h: 0.5, fontFace: BODY_FONT, fontSize: 15, bold: true, color: WHITE, isTextBox: true, margin: 0 });
    s.addText(c[2], { x: x + 0.95, y: y + 0.85, w: 4.8, h: 1.3, fontFace: BODY_FONT, fontSize: 12, color: ICE, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });
  });
  footer(s, null, MUTED);
  s.addNotes("先講結論，後面每一段都在證明其中一句。第一，我們測到的情緒指標都是真的，方向對、統計上顯著，而且重要的結果都在沒看過的資料上重現過。第二，但市場比讀者快，等你看到留言、新聞或公告，價格大多已經反應完了，扣掉成本就沒有了。第三，散戶實際做的事比說的話有用，融資融券每一年都有小訊號，拉長到一個月和技術面一起選股，扣掉成本還是比一般股票好。第四，所以情緒的價值在描述和提醒，這就是平台要做的事：告訴使用者現在市場的氣氛，以及哪些情況要小心。");
}

// ============================================================ 6 原論文作法
{
  const s = newSlide(pres, WHITE);
  kicker(s, "Step 1 · Replication");
  pageTitle(s, "第一步：照論文的作法在台股重做——誤差沒有降，連「今天＝明天」都贏不了");

  const tiles = [
    ["13.2%", "LSTM 預測股價 MAPE（技術面基準）", BULL],
    ["1.22%", "天真預測（今天收盤當明天）的 MAPE", BEAR],
    ["0 / 265", "檔股票中，LSTM 贏過天真預測的檔數（0 檔）", INK],
    ["≈ 45%", "漲跌方向準確率（低於亂猜的 50%）", GOLD],
  ];
  tiles.forEach((t, i) => statTile(s, 0.6 + i * 3.1, 1.6, 2.85, 1.5, t[0], t[1], t[2], 26));

  table(s, ["4 組比較（10 seeds）", "MAPE", "與技術面差異"], [
    ["技術面", "13.24%", "—"],
    ["+ PTT", "13.08%", "−0.16pp（p = 0.35）"],
    ["+ 融資融券", "13.26%", "不顯著（方向準確率 −0.15pp，p = 0.043，反而變差）"],
    ["+ PTT + 融資融券", "13.26%", "不顯著"],
  ], 0.6, 3.3, [3.4, 2.0, 6.7], 0.46, { fontSize: 11.5, headH: 0.5 });
  callout(s, 0.6, 5.75, 12.1, 1.15, "論文：加情緒後 MAPE 0.0675 → 0.0449（降約三成）；我們在台股十年、265 檔：13.24% → 13.08%（不顯著）。而且「今天收盤當明天」就有 1.22%——論文沒有和這種天真預測比較。差別：論文測 2 個月、我們測約 1.5 年；論文用股吧「貼文」——換成最接近的 CMoney 貼文（2020–24、262 檔）：10.64% → 10.56%（p=0.44），天真預測 1.33%，一樣沒有效果。", "warn");
  footer(s);
  s.addNotes("第一步就是照我選讀的論文重做：技術指標加論壇情緒、LSTM 預測隔天收盤價、用 MAPE 比較。論文說加情緒後誤差降三成；我們在台股十年、265 檔上重做，模型連『今天收盤當明天』的天真預測都輸，情緒有沒有幫助根本看不出來。所以我們改測漲跌方向。");
}

// ============================================================ 4 PTT → 個股
{
  const s = newSlide(pres, WHITE);
  kicker(s, "A · Retail Says");
  pageTitle(s, "PTT 情緒對隔天個股漲跌：十年、七個測試年都無效");

  s.addText("六種情緒因子（XGBoost，目標＝隔天開盤→收盤，用開盤前可得資訊）", { x: 0.6, y: 1.55, w: 7, h: 0.35, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  table(s, ["情緒因子", "ΔAUC", "顯著年數"], [
    ["F1 淨情緒 (正−負)/(正+負)", "−0.0001", "0 / 7"],
    ["F2 推噓比", "−0.0002", "0 / 7"],
    ["F3 情緒變化", "−0.0004", "0 / 7"],
    ["F4 意見分歧度", "−0.0002", "0 / 7"],
    ["F5 標的文多空（2,549 篇）", "−0.0001", "0 / 7"],
    ["F6 可信度加權（時間點一致）", "−0.0002", "0 / 7"],
    [{ t: "LSTM 版（+PTT）", bold: true }, "+0.0002", "1 / 7"],
  ], 0.6, 1.95, [3.6, 1.4, 1.4], 0.48, { fontSize: 11, headH: 0.45 });

  statTile(s, 7.9, 1.55, 4.8, 1.5, "59.5%", "的留言其實在收盤 13:30 之後才發。舊的「同日訊號」是時間錯置，放對時間就歸零。", BULL, 26);
  statTile(s, 7.9, 3.25, 4.8, 1.5, "1 / 6", "格蘭傑檢驗：情緒→股價的效果只有 股價→情緒 的約 1/6，且經濟上可忽略。", GOLD, 26);
  callout(s, 7.9, 4.95, 4.8, 1.0, "網友是「看了行情才留言」，不是預測行情。", "warn");
  callout(s, 0.6, 6.15, 12.1, 0.7, "六種因子、兩種模型、十年資料都一致：PTT 留言情緒沒有可用的隔天預測力。", "plain");
  footer(s);
  s.addNotes("PTT 六種做法全部無效。舊版同日訊號其實是時間錯置：一半以上留言在收盤後才發，是在反應行情。");
}

// ============================================================ 5b 只用 PTT 情緒（2026-10-02 夜間）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "A · PTT Only");
  pageTitle(s, "只用 PTT 情緒：情緒跟在股價後面，拿來交易扣成本後虧損");
  s.addImage({ path: "../ptt_event_car.png", x: 0.6, y: 1.55, w: 6.9, h: 3.83 });
  s.addText("圖：情緒強正／強負的股票，留言日前後的累積超額報酬（減掉當天全體平均）", { x: 0.6, y: 5.42, w: 6.9, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
  bulletList(s, [
    "規則「情緒>0 就買、<0 就賣」（2015–2024，約 3 萬筆）：猜對約 47%，未扣成本每筆 +4~6 bp，扣成本 32~59 bp 後十年虧光。",
    "只用 PTT 特徵的模型：漲跌方向 AUC 0.50，等於亂猜。",
    "情緒正面的股票：留言前 20 天已漲 5.6%，之後 20 天只再 +0.1% → 情緒落後股價。",
    "唯一殘留：2015–24 年對「隔夜」報酬 +2.6 bp（t=2.2），但 2025 年樣本外反轉為 −9.9 bp（t=−2.6）。",
    "大盤：情緒上升的隔天，2020–24 年 +23 bp，但 2025 年樣本外 −30 bp → 兩者都不穩定。",
  ], 7.8, 1.6, 4.9, { fontSize: 11, lineHeight: 0.98 });
  callout(s, 0.6, 6.25, 12.1, 0.65, "PTT 情緒能描述行情，但不能穩定預測漲跌或報酬；單獨拿來交易，扣掉成本後是虧損。", "warn");
  footer(s);
  s.addNotes("這頁回答『只用 PTT 情緒能不能預測、能不能賺』。規則很簡單：情緒正就買、負就賣，十年下來猜對率不到五成，扣成本後虧光。事件研究圖顯示，情緒正面的股票其實是先漲了才被看好。唯一留下來的是隔夜約 3 bp 的小效果，和台股散戶情緒推動隔夜報酬的文獻一致，但我們另外抓了 2025 年的 PTT 資料做樣本外測試，個股隔夜效果和大盤效果都反轉了，所以這些都不是穩定的訊號。");
}

// ============================================================ 5c 換分類器／換平台／換股價（v12 新增）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "A · Robustness");
  pageTitle(s, "換分類器、換平台、換股價來源：「說的沒有預測力」都不變");

  table(s, ["換掉什麼", "怎麼做", "主要結果（原本 → 換掉後）", "判讀"], [
    [{ t: "情緒分類器", bold: true }, "用約 6,300 則重標留言微調中文 BERT（與重標的一致率 62% → 69%），重標全部 231 萬則",
      "隔夜效果 +4.4 → +3.5 bp；只用 PTT 的 AUC 0.498 → 0.499；加 PTT 的 ΔAUC −0.0001 → −0.0001", { t: "沒有變好", color: BULL, bold: true }],
    [{ t: "討論平台", bold: true }, "CMoney 股市同學會：2020–2025 約 179 萬篇（含開盤前補抓），用 CMoney 自己的標註訓練",
      "情緒 → 隔天報酬 +4.1 bp（t=4.9）；PTT 同期 +8.0 bp（t=2.3）；細看見下一頁", { t: "統計上有、遠低於成本 47~59 bp", color: GOLD, bold: true }],
    [{ t: "股價來源", bold: true }, "Yahoo 還原價 → TEJ 還原價（兩者約 1% 的股票日報酬有出入）",
      "隔夜 +4.35 → +4.27 bp；融資融券 ΔAUC +0.0019 → +0.0022；PTT −0.0001 → −0.0000", { t: "幾乎相同", color: BEAR, bold: true }],
  ], 0.6, 1.6, [1.7, 4.0, 4.2, 2.2], 0.88, { fontSize: 10.5, headH: 0.45 });

  statTile(s, 0.6, 4.95, 3.9, 1.25, "0.20 vs 0.12", "CMoney／PTT 淨情緒與「當天」超額報酬的相關：CMoney 描述力更好。", GOLD, 20);
  statTile(s, 4.7, 4.95, 3.9, 1.25, "+1.2 bp", "收盤買 CMoney 情緒為正（非漲停）的股票、隔天收盤賣的超額報酬。", BULL, 22);
  statTile(s, 8.8, 4.95, 3.9, 1.25, "0.65 → 0.67", "股票日淨情緒與重標的相關（Breeze → 微調 BERT）：留言層級的錯誤在平均時大多互相抵銷。", INK, 20);
  callout(s, 0.6, 6.35, 12.1, 0.55, "三種「會不會是資料問題」的質疑都實際換掉重跑過：分類器更準、平台不同、股價來源不同，結論都一樣。", "plain");
  footer(s);
  s.addNotes("這頁回答三個常見質疑。第一，分類器不準：我們真的微調了一個比較準的分類器、把 231 萬則全部重標再重跑，結果沒有變好，隔夜效果還略小。原因是留言層級的錯誤在算每天淨情緒時大多互相抵銷。第二，是不是 PTT 這個平台的問題：我們重爬了 CMoney 同學會五年 122 萬篇，情緒對隔天報酬有統計顯著的效果，但只有 4 bp，交易成本是 47 到 59 bp。CMoney 資料只有 2020 年以後，熱門股是收盤前最後 20 篇的抽樣。第三，股價資料：改用學校授權的 TEJ 還原股價重跑，數字只差小數點後一位。");
}

// ============================================================ 5c1 CMoney 完整檢驗（v20 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "A · Second platform");
  pageTitle(s, "換一個平台：CMoney 測得到一點，但一樣賺不到錢");

  statTile(s, 0.6, 1.6, 2.86, 1.45, "+0.003", "加 CMoney 後隔天漲跌預測的 AUC 增量（PTT 是 0），三年都為正", GOLD, 24);
  statTile(s, 3.68, 1.6, 2.86, 1.45, "+1.0%／月", "最樂觀組比最悲觀組下月多漲；2025 樣本外 +1.4%（未顯著）", BEAR, 22);
  statTile(s, 6.76, 1.6, 2.86, 1.45, "< 1 bp／天", "換成報酬的增量；當沖成本 32 bp，扣完全虧", BULL, 22);
  statTile(s, 9.84, 1.6, 2.86, 1.45, "10.64→10.56%", "照論文的 LSTM（MAPE）；天真預測 1.33%，0 檔贏過", INK, 20);

  table(s, ["拿 CMoney 情緒選股（月頻、扣成本）", "2020–2024", "2025 樣本外", "判讀"], [
    [{ t: "剔除最悲觀的 20%（市值加權）", bold: true }, "+0.29%／月（t=1.8）", "+0.04%／月", { t: "沒有贏大盤", color: BULL, bold: true }],
    [{ t: "只買最樂觀的 20%", bold: true }, "+0.09%／月", "+0.62%／月（t=0.9）", { t: "不顯著", color: MUTED, bold: true }],
    [{ t: "反過來買最悲觀的 20%", bold: true }, "−0.93%／月（t=−3.7）", "−0.77%／月（12 個月只贏 2 個月）", { t: "散戶的悲觀是對的", color: BEAR, bold: true }],
    [{ t: "放進技術面選股模型（週／月）", bold: true }, "週 −0.02%、月 +0.17%（t=1.9）", "—", { t: "增量不顯著", color: MUTED, bold: true }],
  ], 0.6, 3.3, [4.0, 2.9, 3.0, 2.2], 0.5, { fontSize: 11, headH: 0.42 });

  callout(s, 0.6, 5.88, 12.1, 1.0, "抓資料時發現的坑：熱門股每天只抓收盤前 20 篇，「隔天開盤前的貼文」有沒有被抓到取決於隔天盤中熱不熱鬧，等於偷看了隔天。補抓 1.4 萬個股票日的開盤前貼文後，AUC 增量 +0.0031 → +0.0027，結論仍成立。假說都在看資料前寫好；2025 年資料是寫好假說後才抓的。", "warn");
  footer(s);
  s.addNotes("這頁是 CMoney 的完整檢驗。CMoney 是貼文型的個股討論區，比 PTT 留言更接近論文用的股吧，但平台 2020 年以前沒有資料，所以只能做 2020 到 2025。結果是 CMoney 比 PTT 有用一點：加進漲跌預測模型，AUC 每年都多 0.003 左右，PTT 是 0；散戶越悲觀的股票，下個月表現越差，控制規模和動能之後還在，2025 年沒看過的資料方向也重現，但只有 12 個月，還不夠顯著。可是一換成報酬就很小：每天不到 1 bp，當沖成本 32 bp；拿來選股，剔除最悲觀的、只買最樂觀的，在 2025 都沒有可靠的超額，也沒有贏大盤。最穩定的是反過來買最悲觀的會一直輸，所以散戶的悲觀不是反指標。照論文的 LSTM 也一樣沒有效果。另外抓資料時發現一個坑：熱門股每天只抓收盤前 20 篇，會讓隔天開盤前的情緒偷看到隔天的熱度，我們補抓修正後結論還在。所以 CMoney 適合放進平台當『散戶極度悲觀』的提示。");
}

// ============================================================ 5c2 換預測目標：波動與成交量（v13 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "A · Other Targets");
  pageTitle(s, "換預測目標：隔天的波動與成交量，PTT 也沒有額外的預測力");

  table(s, ["預測目標（隔天）", "基準的預測排序（Rank IC）", "加 PTT 的增量", "加融資融券", "加法人買賣超"], [
    [{ t: "波動（最高／最低的幅度）", bold: true }, "0.725（樣本外 R² 0.49）", { t: "−0.0000（t=−0.3）", color: BULL, bold: true }, "+0.0001（t=2.5）", "+0.0003（t=4.4）"],
    [{ t: "成交量（相對 20 日均量）", bold: true }, "0.562（樣本外 R² 0.35）", { t: "+0.0003（t=1.9，未過門檻）", color: GOLD, bold: true }, "+0.0012（t=7.5）", "+0.0006（t=6.3）"],
  ], 0.6, 1.6, [3.0, 2.9, 2.8, 1.7, 1.7], 0.55, { fontSize: 11, headH: 0.45 });
  s.addText("基準＝技術面＋過去幾天的振幅與量能；XGBoost 逐年滾動 2018–2024，265 檔；兩個主要檢定事先寫定（門檻 p<0.025）。", { x: 0.6, y: 3.2, w: 12.1, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });

  statTile(s, 0.6, 3.65, 3.9, 1.3, "+2.0%／+2.8%", "PTT 留言比平常多一個單位，隔天振幅／成交量的增加（t=12、17）：相關是顯著的。", GOLD, 20);
  statTile(s, 4.7, 3.65, 3.9, 1.3, "7.16% → 7.19%", "但把留言變數加進去，隔天振幅的解釋力幾乎不變（CMoney：7.84% → 8.11%）。", BULL, 20);
  statTile(s, 8.8, 3.65, 3.9, 1.3, "7.5%", "有 PTT 留言的股票日占比：大部分股票、大部分日子沒有留言可用。", INK, 22);

  bulletList(s, [
    "波動和成交量本來就高度持續：今天振幅大、量大，明天通常也是。留言爆量多半發生在已經爆量的那天，帶的是同一個資訊。",
    "留言爆量最多的一組，隔天成交量是最少一組的約 1.4 倍，但那組當天的振幅本來就比較大 → 是延續，不是預先知道。",
  ], 0.6, 5.15, 12.1, { fontSize: 11.5, lineHeight: 0.52 });
  callout(s, 0.6, 6.3, 12.1, 0.6, "與文獻（Antweiler & Frank 2004）一致：留言與波動、成交量顯著相關；但控制過去的波動與量能後，沒有可用的增量。「描述 ≠ 預測」不只限於漲跌方向。", "plain");
  footer(s);
  s.addNotes("有人會問：情緒預測不了漲跌，那預測波動或成交量呢？文獻說留言板對波動的預測力比對報酬強。我們測了。先講基準：波動和成交量本來就很好預測，只用過去的振幅和量能，排序相關就有 0.72 和 0.56。在這個基準上加 PTT，波動的增量是零，成交量有一點點但沒過事先定的門檻。迴歸裡留言量和隔天波動、成交量的相關是很顯著的，這和文獻一致；但解釋力幾乎沒增加，因為留言爆量和當天已經爆量是同一件事。融資融券和法人對成交量有穩定的小增量，但也只在小數點後第三位。所以結論是：描述不等於預測，這點不只在漲跌方向成立，在波動和成交量也成立。融資融券與法人兩欄是探索性的。");
}

// ============================================================ E1 大波動日歸因（v14 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "B · What Moves Prices");
  pageTitle(s, "換個問法：股價大波動的那幾天，到底發生了什麼？");

  s.addText("全市場 5.2 萬個大波動日（當天漲跌超過自己平常波動的 3 倍且 >3%），依序歸到第一個找得到的原因", { x: 0.6, y: 1.5, w: 7.6, h: 0.35, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  const labels = ["公司公告（重大訊息）", "媒體新聞（鉅亨網）", "當天收盤後才有消息", "同產業一起動", "延續前一天", "找不到"];
  s.addChart(pres.charts.BAR, [
    { name: "大漲日", labels, values: [15.8, 5.7, 8.5, 9.0, 7.3, 53.7] },
    { name: "大跌日", labels, values: [23.8, 3.9, 7.7, 6.1, 3.7, 54.8] },
  ], {
    x: 0.5, y: 1.9, w: 7.7, h: 4.2, barDir: "bar", barGrouping: "clustered", chartColors: [BULL, BEAR],
    catAxisLabelFontFace: BODY_FONT, catAxisLabelFontSize: 11, catAxisOrientation: "maxMin",
    valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 60, valAxisLabelFormatCode: "0\"%\"",
    valGridLine: { color: BORDER, size: 0.5 }, showValue: true, dataLabelFontSize: 9.5, dataLabelFormatCode: "0.0\"%\"",
    showLegend: true, legendPos: "b", legendFontFace: BODY_FONT, legendFontSize: 10,
  });
  s.addText("圖為原樣本以外的 889 檔；原 265 檔的比例幾乎相同（找不到：大漲 51%、大跌 56%）。", { x: 0.6, y: 6.1, w: 7.6, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });

  statTile(s, 8.5, 1.55, 4.2, 1.2, "40.9 萬則", "公開資訊觀測站重大訊息，1,154 檔、2015–2025，每則有時分秒", GOLD, 22);
  statTile(s, 8.5, 2.9, 4.2, 1.2, "18.8 萬則", "鉅亨網台股新聞，清理後對到個股 9.4 萬則", GOLD, 22);
  statTile(s, 8.5, 4.25, 4.2, 1.2, "超過一半", "的大波動日，在公告、新聞、同業連動裡都找不到原因", BULL, 22);
  callout(s, 8.5, 5.6, 4.2, 0.8, "有消息的日子，大波動的機率是平常的 2~4 倍；但消息只解釋得了少數的大波動。", "plain");
  callout(s, 0.6, 6.5, 12.1, 0.5, "作法：先找波動最大的日子，再回頭看那天有什麼公告與新聞，而不是先有情緒指標再去預測。", "plain");
  footer(s);
  s.addNotes("這是後半段的新方向。之前都是先算一個情緒指標，再看它能不能預測股價。這裡反過來：先把股價波動最大的日子挑出來，再回頭看那天到底發生了什麼。我們抓了全市場十一年的公司重大訊息四十萬則、鉅亨網新聞十八萬則。結果是：有公告的占一成五到兩成多，有媒體新聞的再加五個百分點左右，當天收盤後才追報的約八個百分點，同產業一起動的約一成，延續前一天的幾個百分點。全部加起來，還是有一半以上的大波動日找不到原因。這個比例有逐年下降，2015 年約六成，2025 年四成多。要說明的限制：新聞只有鉅亨網一家，所以『找不到』也可能只是我們沒抓到；產業分類用的是 TEJ 的子產業。");
}

// ============================================================ E2 公司公告（v14 新增）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "B · Company Announcements");
  pageTitle(s, "公司公告：哪一類會讓股價往哪邊動——265 檔歸納，889 檔驗證");

  table(s, ["公告類別", "原 265 檔（歸納規則）", "另外 889 檔（驗證）", "則數（889 檔）", "判讀"], [
    [{ t: "澄清媒體報導", bold: true }, "+0.52%", { t: "+0.56%（t=11.8）", color: BEAR, bold: true }, "6,957", "被報導的多半是利多傳聞"],
    [{ t: "庫藏股", bold: true }, "+0.43%", { t: "+0.44%（t=4.9）", color: BEAR, bold: true }, "3,985", "公司拿錢買回自家股票"],
    [{ t: "法人說明會（之後 4 天）", bold: true }, "+0.26%", { t: "+0.33%（t=6.4）", color: BEAR, bold: true }, "18,114", "公告要開法說後股價偏強"],
    [{ t: "股利分派／配股", bold: true }, "+0.22%", { t: "+0.16%（t=4.3）", color: BEAR, bold: true }, "16,506", ""],
    [{ t: "訴訟／裁罰／檢調", bold: true }, "−0.46%", { t: "−0.44%（t=−4.5）", color: BULL, bold: true }, "1,228", ""],
    [{ t: "災害／停工／事故", bold: true }, "−0.51%", { t: "−0.67%（t=−3.9）", color: BULL, bold: true }, "462", ""],
    ["減資（彌補虧損）", "−3.53%", "−2.36%（t=−5.4）", "200", "非事先寫定，方向一致"],
    ["財報通過／提報", "+0.06%", "0.00%", "27,507", "常大波動，但方向不定"],
  ], 0.6, 1.6, [3.0, 2.4, 2.6, 1.6, 2.5], 0.44, { fontSize: 10.5, headH: 0.45 });
  s.addText("數字＝公告影響日當天的超額報酬（個股 − 全市場平均）。收盤後才公告的，影響日算下一個交易日（94% 的公告在收盤後發布）。", { x: 0.6, y: 5.65, w: 12.1, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
  callout(s, 0.6, 6.05, 12.1, 0.85, "前六類是事先寫定的假說：規則只看 265 檔歸納，一個字不改套到完全沒看過的 889 檔，六個全部通過，兩段期間方向也一致。這是「解釋股價」而不是「可以交易」：幅度 0.2~0.7%，多數隔天一開盤就反應完。", "good");
  footer(s);
  s.addNotes("這頁是公司自己發的重大訊息。我們先在原本 265 檔上，用公告主旨的關鍵字把公告分成十幾類，看每一類公告之後股價平均怎麼動。然後把六個最清楚的寫成假說，規則完全不改，直接套到另外 889 檔從來沒看過的股票上。結果六個全部通過，數字幾乎一樣：澄清媒體報導、庫藏股、法說會、股利是正的；訴訟裁罰、災害事故是負的。彌補虧損的減資當天平均跌兩到三個百分點，但件數少、不在事先寫定的六個裡面。財報通過那天常常大波動，但漲跌各半，平均是零。限制：類別是用關鍵字規則分的，會有分錯；這是平均反應，而且九成四的公告在收盤後才發，隔天開盤就反應了，所以不能說可以拿來交易。");
}

// ============================================================ E3 媒體新聞（v14 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "B · Media News");
  pageTitle(s, "媒體新聞：語氣的方向很準，但開盤那一下就反應完了");

  table(s, ["開盤前已發布的新聞（標題語氣）", "原 265 檔", "另外 889 檔（驗證）", "其中：隔夜跳空", "其中：開盤→收盤", "之後 4 天"], [
    [{ t: "正面（創新高、調升、接單…）", bold: true }, "+0.43%", { t: "+0.57%（t=15.7）", color: BEAR, bold: true }, "+0.60%", "−0.03%", "+0.13%"],
    [{ t: "負面（下修、虧損、重挫…）", bold: true }, "−0.29%", { t: "−0.58%（t=−9.8）", color: BULL, bold: true }, "−0.53%", "−0.06%", { t: "−0.43%（t=−3.8）", color: BULL, bold: true }],
    [{ t: "正面 − 負面", bold: true }, "+0.68%", { t: "+1.14%（t=16.0）", color: BEAR, bold: true }, "—", "—", "+0.58%（t=4.8）"],
  ], 0.6, 1.6, [3.3, 1.2, 2.4, 1.5, 1.6, 2.1], 0.5, { fontSize: 10.5, headH: 0.5 });
  s.addText("語氣詞表與三個假說都在看報酬之前寫定；889 檔三個全部通過。右三欄為 889 檔。", { x: 0.6, y: 3.7, w: 12.1, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });

  statTile(s, 0.6, 4.1, 3.9, 1.3, "幾乎全在跳空", "正面新聞的 +0.57% 裡，隔夜跳空就有 +0.60%，開盤後是 −0.03%：看到新聞再買已來不及。", BULL, 20);
  statTile(s, 4.7, 4.1, 3.9, 1.3, "股價常先動", "有正面新聞的股票，前一天已先漲 0.5%、前五天漲 1.6%；收盤後才出的新聞有一部分是在追報。", GOLD, 20);
  statTile(s, 8.8, 4.1, 3.9, 1.3, "續跌也做不到", "負面新聞之後四天再跌 0.4%，但剔除鎖跌停的之後，放空只剩 0.15%，扣成本是負的。", INK, 20);
  bulletList(s, [
    "伴隨漲跌最大的新聞類別（889 檔）：訂單／出貨 +0.70%、併購／合作 +0.56%、新產品／擴產 +0.56%、月營收 +0.41%；處置／違約／弊案／事故 −0.81%。",
    "關鍵字：驗得過的多半是在描述股價本身（漲停、跳空、爆量、重挫），是結果不是原因；「減資」「法說」等詞在後半段驗不過。",
  ], 0.6, 5.6, 12.1, { fontSize: 11, lineHeight: 0.5 });
  callout(s, 0.6, 6.6, 12.1, 0.42, "和 PTT 留言的結論一致，只是這次換成記者寫的新聞：方向說得對，但市場比讀者快。", "plain");
  footer(s);
  s.addNotes("這頁是鉅亨網的新聞。我們先寫好一組正面詞和負面詞，只看影響日開盤前就已經發布的新聞，也就是投資人開盤前讀得到的。正面新聞當天平均多半個百分點，負面少半個百分點，兩組股票、兩段期間都一樣，這是很穩的關係。但把當天拆成隔夜跳空和開盤到收盤兩段，會發現幾乎全部發生在開盤那一下，開盤之後是零。所以新聞的方向是對的，可是看到再買已經來不及。另外股價常常比新聞先動：有正面新聞的股票，前一天已經先漲了。負面新聞之後四天看起來還會續跌，中小型股比較明顯；但我們做了實際回測，把當天鎖跌停、空不到的剔除之後，放空的超額報酬只剩千分之一點五，扣掉成本是負的。所以續跌也是鎖住造成的，一樣做不到。限制：只有鉅亨網一家、只看標題、有三分之二的標題用詞表分不出正負。");
}

// ============================================================ 10 為什麼
{
  const s = newSlide(pres, WHITE);
  kicker(s, "B · Why");
  pageTitle(s, "為什麼預測不出來？情緒跟著股價走，不會領先");

  const cards = [
    ["1", "價格先動、情緒後跟", "格蘭傑：股價→情緒 F≈470，情緒→股價僅約 1/6；59.5% 的留言在收盤後才發。"],
    ["2", "隔天漲跌本來就接近隨機", "技術面模型 AUC 也只有約 0.56，任何額外訊號都只能是零點幾個百分點。"],
    ["3", "訊號短命、交易成本高", "融資融券的預測力隔兩天剩約一半（ΔAUC +0.0021→+0.0010）；當沖來回成本 0.32~0.44%。我們也直接檢驗「當沖越多、消化越快」：高低當沖組的衰減沒有差異，這個推測未獲支持。"],
    ["4", "方法有效：對照組測得出來", "融資融券 t = 5.1；恐慌指數在崩盤時準確升高；資料已檢查（還原股價、補缺漏日、留言時間）。"],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.65 + Math.floor(i / 2) * 2.35;
    s.addShape("roundRect", { x, y, w: 5.95, h: 2.15, rectRadius: 0.08, fill: { color: i === 3 ? "EAF1EC" : LIGHT_BG }, line: { color: BORDER, width: 1 } });
    s.addText(c[0], { x: x + 0.25, y: y + 0.2, w: 0.6, h: 0.6, fontFace: TITLE_FONT, fontSize: 28, bold: true, color: GOLD, isTextBox: true, margin: 0 });
    s.addText(c[1], { x: x + 0.95, y: y + 0.22, w: 4.8, h: 0.5, fontFace: BODY_FONT, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
    s.addText(c[2], { x: x + 0.95, y: y + 0.85, w: 4.8, h: 1.2, fontFace: BODY_FONT, fontSize: 12.5, color: INK2, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });
  });
  callout(s, 0.6, 6.25, 12.1, 0.8, "資料上的限制：Breeze 分類器準確率約六成、約 24% 留言與股市無關，會稀釋訊號。我們量過影響：以 Claude 逐則重標 5,194 則留言（400 個股票日）當對照，分類器再準，隔夜效果也只從 +4.5 變成約 +5.9 bp（保守上界 10~13 bp），仍遠低於交易成本 32~59 bp；實際換成微調分類器全部重跑也沒有變好；依規模分三組，最小那組 PTT 也沒有增量。", "plain");
  footer(s);
  s.addNotes("四個原因。重點是第四點：我們用有效的對照組證明方法是可以偵測訊號的。底下的限制要主動講：分類器只有六成準，所以我們另外用 Claude 重標了五千多則留言來量影響。Breeze 的主要毛病是把中立的留言硬判成有方向（它判正的留言只有 47% 真的是正、44% 其實中立），不是把正負搞反。換算下來 Breeze 的雜訊讓效果剩約七到八成（衰減係數 0.76±0.09），就算分類器跟重標一樣準，效果頂多放大約 1.3 倍，還是不到成本的三分之一。限制：重標只有一位標註者、400 個股票日。");
}

// ============================================================ 5 融資融券
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "C · Retail Does");
  pageTitle(s, "散戶「做」的：融資融券有穩定、但很小的訊號");

  table(s, ["加入的訊號", "模型", "ΔAUC", "t 值", "顯著年數（為正年數）"], [
    [{ t: "融資融券", bold: true }, "XGBoost", { t: "+0.0019", color: BEAR, bold: true }, "5.06", "5 / 7（7 / 7）"],
    [{ t: "融資融券", bold: true }, "LSTM", "+0.0007", "0.92", "2 / 7（6 / 7）"],
    ["PTT 留言（對照）", "XGBoost", "−0.0001", "—", "0 / 7"],
    ["PTT 留言（對照）", "LSTM", "+0.0002", "0.62", "1 / 7"],
    ["法人買賣超（對照）", "XGBoost", "+0.0039", "3.91", "4 / 7"],
  ], 0.6, 1.6, [2.3, 1.3, 1.1, 0.8, 1.9], 0.46, { fontSize: 11, headH: 0.5 });

  // 名詞說明框
  s.addShape("roundRect", { x: 8.3, y: 1.6, w: 4.4, h: 2.8, rectRadius: 0.08, fill: { color: CARD_BG }, line: { color: BORDER, width: 1 } });
  s.addText("名詞說明", { x: 8.55, y: 1.68, w: 3.9, h: 0.35, fontFace: BODY_FONT, fontSize: 12.5, bold: true, color: GOLD, isTextBox: true, margin: 0 });
  const gl = (term, desc, last) => [
    { text: term + "　", options: { bold: true, color: INK, fontSize: 11.5 } },
    { text: desc, options: { color: INK2, fontSize: 11.5, breakLine: !last } },
  ];
  s.addText([
    ...gl("融資", "借錢買股票（看多）。增加＝散戶願意借錢加碼看多。"),
    ...gl("融券", "借股票來賣（放空）。增加＝有人押注會跌。"),
    ...gl("券資比", "融券餘額 ÷ 融資餘額，越高代表看空越多。"),
    ...gl("為何當散戶指標", "融資主力是散戶；「做」要花錢，比「說」更可信。", true),
  ], { x: 8.55, y: 2.05, w: 3.95, h: 2.25, fontFace: BODY_FONT, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.3, paraSpaceAfter: 8 });

  statTile(s, 0.6, 4.6, 3.9, 1.5, "券資比高 → 較差", "高券資比的股票，之後 1~3 個月報酬較低（Fama-MacBeth t ≈ −3.5）：放空者方向是對的。", INK, 18);
  statTile(s, 4.7, 4.6, 3.9, 1.5, "0.32~0.44%", "台股當沖來回成本，吃掉訊號：做多前 5% 六折成本後年化約 −24%（回測用含法人版）。", BULL, 22);
  statTile(s, 8.8, 4.6, 3.9, 1.5, "隔兩天剩一半", "融資融券 ΔAUC 從隔天 +0.0021 降到隔兩天 +0.0010，約剩一半。", GOLD, 20);
  callout(s, 0.6, 6.3, 12.1, 0.6, "訊號是真的（七年都為正），但幅度小、成本高，實務上無法穩定獲利。", "plain");
  footer(s);
  s.addNotes("融資融券是唯一穩健的訊號，XGBoost 七年全為正。但 AUC 只多 0.002，扣掉當沖成本就沒了。");
}

// ============================================================ 5d 全市場（v12 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "C · Full Market");
  pageTitle(s, "擴到全市場 1,144 檔：散戶「做」的訊號在沒被挑進樣本的股票也成立");

  statTile(s, 0.6, 1.6, 2.85, 1.3, "1,144 檔", "TEJ 四碼普通股，2015–2024，含 64 檔期間內消失（下市等）", GOLD, 22);
  statTile(s, 3.7, 1.6, 2.85, 1.3, "879 檔", "完全不在原本 265 檔樣本內", GOLD, 22);
  statTile(s, 6.8, 1.6, 2.85, 1.3, "243 萬", "股票日；每年 956~1,083 檔", INK, 22);
  statTile(s, 9.9, 1.6, 2.8, 1.3, "3 / 3", "事先寫定的主要檢定全部通過（Bonferroni）", BEAR, 22);

  table(s, ["檢驗", "原 265 檔", "全市場 1,144 檔", "不在原樣本的 879 檔"], [
    [{ t: "加融資融券的增量 ΔAUC", bold: true }, "+0.0025", { t: "+0.0026（t=7.9，7/7 年為正）", color: BEAR, bold: true }, "+0.0027"],
    [{ t: "加散戶成交比重的增量 ΔAUC", bold: true }, "+0.0031", { t: "+0.0034（t=6.3，7/7 年為正）", color: BEAR, bold: true }, "+0.0035"],
    ["加法人買賣超（對照）", "+0.0041", "+0.0046（t=8.2）", "+0.0048"],
    [{ t: "券資比最高 − 最低組，隔天盤中", bold: true }, "−19.4 bp", { t: "−21.7 bp（t=−11.3）", color: BEAR, bold: true }, "−21.5 bp"],
    [{ t: "月頻：券資比與下月報酬 Rank IC", bold: true }, "−0.061", { t: "−0.047（t=−3.9）", color: BEAR, bold: true }, "−0.042"],
  ], 0.6, 3.15, [4.1, 2.2, 3.4, 2.4], 0.46, { fontSize: 11, headH: 0.45 });
  callout(s, 0.6, 6.1, 12.1, 0.8, "「265 檔是事後挑的、有倖存者偏差」這項限制，對融資融券與散戶成交比重的結論可以解除。PTT 的結論仍只限 265 檔（全市場沒有 PTT 配對）；下市當下的報酬尚未處理。", "good");
  footer(s);
  s.addNotes("原本的限制之一是 265 檔是事後選的。這次用 TEJ 把樣本擴到全市場一千多檔、包含中途下市的股票，模型在全市場重新訓練。融資融券和散戶成交比重的增量，在全市場、以及完全沒被挑進原樣本的 879 檔上都重現，數字幾乎一樣，七個測試年全部為正。券資比的每日排序與月頻結果也一樣。假說和門檻都是跑之前寫定的。要主動講的限制：全市場沒有 PTT 資料，所以 PTT 的結論仍只限 265 檔；股票下市那一刻的報酬沒有算進去；全市場那次訓練用顯示卡，其餘設定相同。265 檔那欄是同一次全市場模型在 265 檔上的表現。");
}

// ============================================================ T 交易頻率（v16 新增）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "C · Holding Period");
  pageTitle(s, "訊號不是沒有，是交易越頻繁、成本吃掉越多");
  table(s, ["持有時間", "做法（全市場、只做多、扣成本、剔除買不到的）", "扣成本後，比一般股票多", "結論"], [
    [{ t: "幾天", bold: true }, "PTT／CMoney 情緒、新聞語氣、公告、庫藏股、有消息的大漲、負面新聞放空", { t: "全部 ≤ 0", color: BULL, bold: true }, "成本吃光"],
    [{ t: "一週", bold: true }, "技術面＋散戶交易行為選股，每週換（已持有的還在前 40% 就續抱）", { t: "+0.09%／週（t=1.8）", color: GOLD, bold: true }, "剩一點、不夠穩"],
    [{ t: "一個月", bold: true }, "同上，每月換一次", { t: "+0.63%／月（t=2.9）", color: BEAR, bold: true }, "唯一站得住"],
  ], 0.6, 1.6, [1.6, 5.9, 2.8, 1.8], 0.62, { fontSize: 11.5, headH: 0.5 });
  statTile(s, 0.6, 4.25, 3.9, 1.35, "≈ 0.5%", "每次進出的成本（手續費六折＋證交稅）；日頻訊號只有 0.2~1%", BULL, 24);
  statTile(s, 4.7, 4.25, 3.9, 1.35, "39% → 18%", "週頻每週換股比例：降低換股後，年化 14% → 19%", GOLD, 24);
  statTile(s, 8.8, 4.25, 3.9, 1.35, "≈ 0", "新聞、公告、隔夜情緒加進週頻或月頻模型的增量", INK, 24);
  callout(s, 0.6, 5.85, 12.1, 1.0, "文字類的情緒（留言、新聞、公告）在每一種時間長度都加不上去——資訊幾天內就反映在價格裡了。拉長時間後還留得住的，是價格與籌碼類：技術面加上散戶實際的交易行為（券資比、融資增減等）。", "plain");
  footer(s);
  s.addNotes("這頁把前面所有結果放在一起看。同樣的訊號，交易越頻繁，成本吃得越多。幾天內進出的做法，扣掉成本全部是零或負的；每週換股，扣掉成本剩一點，但統計上還不夠穩；每個月換一次，才扣得起成本，而且顯著贏過一般股票。另外，不管哪個時間長度，把新聞、公告或隔夜情緒加進去都沒有幫助，因為那些資訊幾天內就反應完了。留得住的是價格和籌碼類的訊號。");
}

// ============================================================ E5 拉長到一個月（v15 新增，延伸）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "C · Monthly");
  pageTitle(s, "拉長到一個月：扣掉成本後才有剩");

  statTile(s, 0.6, 1.55, 2.85, 1.3, "+22.1%", "年化（扣成本後），2018–2025；同一批股票平均 +12.9%", BEAR, 24);
  statTile(s, 3.7, 1.55, 2.85, 1.3, "每月 +0.63%", "贏過一般股票（t=2.9）；沒看過的 889 檔也一樣（+0.66%）", BEAR, 20);
  statTile(s, 6.8, 1.55, 2.85, 1.3, "−14.5%", "最大回落；加權指數約 −26%", INK, 24);
  statTile(s, 9.9, 1.55, 2.8, 1.3, "不保證贏大盤", "相對加權指數不顯著；2024、2025 兩年輸", BULL, 18);

  table(s, ["加進月頻選股模型的資訊", "每月贏過一般股票", "比原本多"], [
    [{ t: "技術面（動能、波動、成交量等 11 項）", bold: true }, "+0.50%", "—"],
    [{ t: "＋散戶交易行為（券資比、融資增減、信用交易、散戶成交比重）", bold: true }, { t: "+0.63%", color: BEAR, bold: true }, { t: "+0.13%（t=3.3）", color: BEAR, bold: true }],
    ["＋媒體新聞＋公司公告（月）", "+0.67%", "+0.03%（不顯著）"],
    ["＋隔夜報酬情緒（月）", "+0.62%", "−0.01%（不顯著）"],
  ], 0.6, 3.1, [6.6, 2.7, 2.8], 0.46, { fontSize: 11, headH: 0.45 });
  bulletList(s, [
    "作法：每月底依模型分數買全市場前 20%（約 160 檔），持有一個月；成交太冷清的不買、下市當月記賠 50%、每月換股約一半並扣成本。規則與通過標準事先寫定。",
    "為什麼這次付得起成本：一個月才換一次股，成本每月約 0.2%；前面日頻的訊號只有 0.2~1%，卻每幾天就要付一次 0.5%。",
  ], 0.6, 5.5, 12.1, { fontSize: 10.5, lineHeight: 0.5 });
  callout(s, 0.6, 6.5, 12.1, 0.5, "限制：要靠模型把十幾個特徵組合才有效（只用三個好懂的訊號不行）；2025 年只和一般股票打平；文字情緒拉長到一個月也沒有剩。", "warn");
  footer(s);
  s.addNotes("這頁回答一個很自然的問題：到底有沒有什麼扣掉成本還有用？前面都失敗，是因為訊號只有零點幾個百分點、幾天就消失，卻每次都要付半個百分點的成本。把持有時間拉長到一個月，成本就付得起了。每個月底用技術面加上散戶的交易行為給全市場打分數，買前兩成，抱一個月。八年下來扣掉成本年化百分之二十二，同一批股票的平均是十三，在沒看過的八百多檔上也一樣。散戶交易行為每個月多貢獻千分之一點三，而且顯著；新聞、公告、隔夜情緒加進去都沒有再多。要誠實講三件事：第一，跟加權指數比統計上不顯著，2024、2025 兩年是輸的，因為那兩年指數靠台積電這類權值股；第二，要靠模型組合十幾個特徵，只用三個好懂的訊號是不行的；第三，2025 這個完全沒用過的年份只跟一般股票打平。所以這是『有選股能力』，不是『穩贏大盤』。");
}

// ============================================================ E4 有消息 vs 沒消息／庫藏股（v14 新增）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "C · After the Move");
  pageTitle(s, "大漲之後：找不到消息的會回吐，有消息的不會");

  table(s, ["大漲日收盤買、持有 5 天的超額報酬（未扣成本）", "有消息", "沒消息", "樣本"], [
    [{ t: "沒收漲停（買得到）— 2015–2025，另外 889 檔", bold: true }, "−0.02%", { t: "−0.60%（t=−8.2；11 年全為負）", color: BULL, bold: true }, "4,019／14,136 個"],
    [{ t: "沒收漲停（買得到）— 2015–2025，原 265 檔", bold: true }, "+0.09%", { t: "−0.67%（t=−4.5；11 年全為負）", color: BULL, bold: true }, "1,276／4,869 個"],
    [{ t: "沒收漲停 — 2001–2013（沒看過的年份，不分有無消息）", bold: true }, { t: "不分消息：−0.48%", color: BULL, bold: true }, { t: "t=−6.7；13 年裡 12 年為負", color: BULL, bold: true }, "13,966 個"],
    ["含收漲停（多數買不到）— 889 檔", "+1.12%", "+0.30%", "漲停占 37%"],
  ], 0.6, 1.6, [5.6, 2.6, 2.6, 1.3], 0.5, { fontSize: 10.5, headH: 0.45 });
  s.addText("「有消息」＝當天收盤前已有公司公告或媒體新聞；「沒消息」＝兩者都沒有。2001–2013 沒有新聞資料，只能驗不分消息的版本，假說事先寫定。", { x: 0.6, y: 4.2, w: 12.1, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });

  statTile(s, 0.6, 4.6, 3.9, 1.35, "續漲是假象", "含漲停時「有消息的大漲之後 5 天 +1.1%」，幾乎全來自隔天繼續鎖漲停、買不到的股票。", GOLD, 20);
  statTile(s, 4.7, 4.6, 3.9, 1.35, "庫藏股 +1.23%", "公告後第 2~5 天（2001–2013 驗證，t=5.9；2015–25 為 +0.65%）。扣成本後贏不了大盤。", BEAR, 20);
  statTile(s, 8.8, 4.6, 3.9, 1.35, "恐慌時掉三成", "技術面訊號在市場恐慌時的準確度（IC 0.151 → 0.106）；但控制大盤漲跌後只剩邊緣顯著。", INK, 20);
  callout(s, 0.6, 6.15, 12.1, 0.75, "平台可以直接用的提示：「今天大漲、沒鎖漲停、又找不到消息 → 過去二十四年，之後五天平均回吐約半個百分點」。這是提醒不要追高，不是放空策略（未扣成本、放空有限制）。", "good");
  footer(s);
  s.addNotes("找到原因之後，下一個問題是：大漲之後會怎麼走？文獻上有個說法，有新聞的股價變動會延續，沒新聞的會反轉。我們一開始含漲停一起看，有消息的大漲之後五天還多一個百分點，看起來是延續。但做實際回測、把當天收在漲停買不到的剔除之後，有消息的變成零，沒消息的是負的千分之六，而且兩組股票、十一年裡每一年都是負的。所以原本看到的續漲，是隔天繼續鎖漲停造成的，根本買不到。我們再用 2001 到 2013 年完全沒看過的資料驗證，那段期間沒有新聞，只能驗不分消息的版本，假說事先寫好，結果通過：十三年裡十二年為負。要誠實說的是：2015 年以後『沒消息會回吐』這個切法不是事先寫定的，事先寫的版本含漲停、沒有通過；我們是在回測時才發現要剔除漲停。但它在兩組股票每一年方向都一樣，而且不分消息的版本在舊資料上事先寫定後通過。中間那格是庫藏股：公告後第二到第五天續漲，在沒看過的 2001 到 2013 年驗證通過，但扣掉成本後贏不了大盤。右邊那格是情緒當開關：技術面訊號在恐慌時比較不準，但控制大盤近期漲跌後，情緒的效果大半被吸收。");
}

// ============================================================ 9 細分情緒
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "D · Emotion Index");
  pageTitle(s, "細分情緒指數：描述力很好，預測力弱");

  s.addText("描述力：重大事件當下，對應的情緒指數（z 分數）準確升高", { x: 0.6, y: 1.5, w: 8, h: 0.35, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  const t = [
    ["恐慌 +2.06", "2024-08-05 股災", BULL],
    ["恐慌 +1.90", "2020-03 疫情崩盤", BULL],
    ["恐慌 +1.28", "2015-08 股災", BULL],
    ["貪婪 +1.07", "2024 AI 行情高點", GOLD],
  ];
  t.forEach((x, i) => statTile(s, 0.6 + i * 3.1, 1.95, 2.85, 1.45, x[0], x[1], x[2], 22));

  s.addText("預測力：幾乎就是隨機", { x: 0.6, y: 3.7, w: 8, h: 0.35, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  bulletList(s, [
    "8 類情緒：多／空／貪／慌／追（FOMO）／絕／諷（反串）／中；用 36.2 萬則閒聊留言的每日平均機率做指數。",
    "38 個預測檢定中約 5 個 |t| ≈ 2（純巧合預期約 1.9 個），最大 |t| = 2.2，型態彼此不一致。",
    "另外：大盤散戶溫度計與過去 5 日行情相關 +0.49（跟著行情走），預測未來大盤 |t| < 1.5。",
  ], 0.6, 4.15, 12.1, { fontSize: 11.5, lineHeight: 0.66 });
  callout(s, 0.6, 6.25, 12.1, 0.65, "適合放進平台當「散戶情緒儀表板」，並註明：描述當下，不代表預測未來。", "good");
  footer(s);
  s.addNotes("八類情緒在崩盤時準確升高，很適合做儀表板；但拿來預測沒有穩定效果。");
}

// ============================================================ P 平台的三種功能（v16 新增）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "D · Platform");
  pageTitle(s, "平台：把驗證過的結果做成三個模組");
  const cols = [
    ["情緒指數模組", "描述現在的市場氣氛", ["散戶溫度計、恐慌／貪婪等 8 類情緒指數（股災時準確升高）", "個股淨情緒；標出「散戶極度悲觀」（之後一個月平均較差）", "隔夜情緒：最近是不是被散戶在開盤追高"], BEAR],
    ["技術面＋籌碼看板", "唯一扣成本後還有用的", ["融資融券、散戶成交比重（全市場每一年都有小訊號）", "月頻選股分數：扣成本後比一般股票每月多 0.63%", "註明：不保證贏大盤（2024、2025 輸加權指數）"], GOLD],
    ["事件提示", "提醒，不是預測", ["公告類別的歷史方向（889 檔驗證：澄清＋、訴訟／事故−）", "今天大漲、沒鎖漲停、找不到消息 → 過去之後五天平均回吐", "市場恐慌時，技術面訊號可靠度下降約三成"], BULL],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 4.1;
    s.addShape("roundRect", { x, y: 1.6, w: 3.9, h: 4.35, rectRadius: 0.08, fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 } });
    s.addShape("rect", { x, y: 1.6, w: 3.9, h: 0.08, fill: { color: c[3] }, line: { type: "none" } });
    s.addText(c[0], { x: x + 0.25, y: 1.8, w: 3.4, h: 0.45, fontFace: BODY_FONT, fontSize: 16, bold: true, color: c[3], isTextBox: true, margin: 0 });
    s.addText(c[1], { x: x + 0.25, y: 2.25, w: 3.4, h: 0.35, fontFace: BODY_FONT, fontSize: 11, color: INK2, isTextBox: true, margin: 0 });
    bulletList(s, c[2], x + 0.25, 2.8, 3.45, { fontSize: 11, lineHeight: 1.0 });
  });
  callout(s, 0.6, 6.15, 12.1, 0.75, "每個功能背後都有十年、全市場、事先寫定假說的檢驗；平台上一律標示「這是描述與提醒，不是漲跌預測」。", "good");
  footer(s);
  s.addNotes("研究的結論決定了平台要怎麼做。第一塊是情緒儀表板，因為情緒最強的地方是描述，例如恐慌指數在股災時準確升高。第二塊是籌碼和選股訊號，這是唯一扣掉成本後還有用的，但我們會清楚標示它不保證贏大盤。第三塊是事件提示，例如這類公告過去通常往哪個方向、沒消息的大漲之後通常回吐、恐慌時技術面比較不可靠。每一個功能背後都有檢驗，而且都標示是描述和提醒，不是預測。這也是我們下一步要做的成品。");
}

// ============================================================ 12 待討論
{
  const s = newSlide(pres, WHITE);
  kicker(s, "Discussion");
  pageTitle(s, "待討論事項與下一步");

  const items = [
    ["結論框架可以接受嗎？", "「情緒方向對、但市場比讀者快；能解釋是什麼在動股價；扣成本後有用的是散戶交易行為＋技術面」，是否符合專題成果的期待？"],
    ["平台網頁的重點", "情緒儀表板（散戶溫度計、恐慌/貪婪指數、隔夜情緒）＋融資融券訊號＋事件提示（公告類別、沒消息的大漲、恐慌時訊號可靠度）；註明是描述與提醒、不是預測。"],
    ["還要補做哪些檢驗？", "已補：換分類器、CMoney、TEJ 股價、全市場、波動與成交量、公告與新聞事件分析、月頻選股、隔夜情緒。我們判斷預測面已收斂；可選：注意股與處置股（需重匯資料）、每週散戶持股人數。"],
    ["月頻選股要放多重？", "目前放在 C 段第 17、18 頁。績效主要來自技術面，放太重會像量化選股、偏離「情緒」主題。"],
  ];
  items.forEach((it, i) => {
    const y = 1.7 + i * 1.28;
    s.addShape("roundRect", { x: 0.6, y, w: 12.1, h: 1.08, rectRadius: 0.06, fill: { color: CARD_BG }, line: { color: BORDER, width: 1 } });
    s.addText(`Q${i + 1}`, { x: 0.85, y, w: 0.8, h: 1.08, fontFace: TITLE_FONT, fontSize: 20, bold: true, color: GOLD, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(it[0], { x: 1.75, y: y + 0.08, w: 3.6, h: 0.92, fontFace: BODY_FONT, fontSize: 12.5, bold: true, color: INK, isTextBox: true, margin: 0, valign: "middle", lineSpacingMultiple: 1.15 });
    s.addText(it[1], { x: 5.5, y: y + 0.05, w: 7.0, h: 0.98, fontFace: BODY_FONT, fontSize: 11, color: INK2, isTextBox: true, margin: 0, valign: "middle", lineSpacingMultiple: 1.25 });
  });
  footer(s);
  s.addNotes("請老師決定方向：結論框架、平台重點、還要不要補做。");
}

// ============================================================ X 附錄分隔（v16 新增）
{
  const s = newSlide(pres, NAVY);
  s.addText("附錄", { x: 0.9, y: 2.6, w: 8, h: 0.9, fontFace: TITLE_FONT, fontSize: 40, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText("模型說明、大盤層級的檢驗、散戶持股分組、文獻對照——被問到時再翻。", { x: 0.9, y: 3.6, w: 11, h: 0.6, fontFace: BODY_FONT, fontSize: 15, color: ICE, isTextBox: true, margin: 0 });
  s.addShape("line", { x: 0.9, y: 4.4, w: 2.2, h: 0, line: { color: GOLD, width: 2 } });
}

// ============================================================ 3 三種預測模型
{
  const s = newSlide(pres, WHITE);
  kicker(s, "附錄 · Models");
  pageTitle(s, "三種預測模型：結論一致，不是某個模型的偶然");

  const box = (x, y, w, h, text, fill, fg, fs, bold) => {
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: BORDER, width: 0.75 } });
    s.addText(text, { x, y, w, h, fontFace: BODY_FONT, fontSize: fs || 9, color: fg || INK, bold: !!bold, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  };
  const line = (x1, y1, x2, y2, arrow) => {
    const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
    const flipH = (x2 < x1) !== (y2 < y1) ? true : false;   // 斜線方向
    s.addShape("line", { x, y, w, h, flipH, line: { color: MUTED, width: 1, endArrowType: arrow ? "triangle" : undefined } });
  };
  const cards = [
    { x: 0.6, t: "樹模型（XGBoost）", d: "像一連串的是非題，把幾百棵小樹疊起來，每棵修正前一棵的錯。" },
    { x: 4.7, t: "神經網路（LSTM）", d: "把過去 20 天一天一天讀進去，邊讀邊更新「記憶」，最後輸出預測。" },
    { x: 8.8, t: "邏輯迴歸（基準）", d: "最簡單：每個指標乘上權重加總，換成上漲機率。" },
  ];
  cards.forEach(c => {
    s.addShape("roundRect", { x: c.x, y: 1.55, w: 3.9, h: 2.8, rectRadius: 0.08, fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 } });
    s.addText(c.t, { x: c.x + 0.25, y: 1.62, w: 3.4, h: 0.4, fontFace: BODY_FONT, fontSize: 14, bold: true, color: GOLD, isTextBox: true, margin: 0 });
    s.addText(c.d, { x: c.x + 0.25, y: 3.65, w: 3.45, h: 0.65, fontFace: BODY_FONT, fontSize: 10.5, color: INK2, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
  });

  // 圖解 1：決策樹
  {
    const cx = 0.6 + 1.95, ry = 2.1;
    box(cx - 0.65, ry, 1.3, 0.3, "融資增加？", WHITE, INK, 9);
    box(cx - 1.45, ry + 0.55, 1.1, 0.3, "RSI 偏高？", WHITE, INK, 9);
    box(cx + 0.35, ry + 0.55, 1.1, 0.3, "昨日大漲？", WHITE, INK, 9);
    line(cx, ry + 0.3, cx - 0.9, ry + 0.55); line(cx, ry + 0.3, cx + 0.9, ry + 0.55);
    [[cx - 1.75, "跌", BEAR], [cx - 1.0, "漲", BULL], [cx + 0.2, "漲", BULL], [cx + 0.95, "跌", BEAR]].forEach(([lx, t, col]) => {
      box(lx, ry + 1.1, 0.55, 0.28, t, WHITE, col, 9, true);
    });
    line(cx - 0.9, ry + 0.85, cx - 1.475, ry + 1.1); line(cx - 0.9, ry + 0.85, cx - 0.725, ry + 1.1);
    line(cx + 0.9, ry + 0.85, cx + 0.475, ry + 1.1); line(cx + 0.9, ry + 0.85, cx + 1.225, ry + 1.1);
  }
  // 圖解 2：LSTM
  {
    const x0 = 4.7 + 0.3, y0 = 2.35;
    ["第1天", "第2天", "…", "第20天"].forEach((t, i) => box(x0 + i * 0.85, y0, 0.65, 0.4, t, WHITE, INK, 9));
    for (let i = 0; i < 3; i++) line(x0 + i * 0.85 + 0.65, y0 + 0.2, x0 + (i + 1) * 0.85, y0 + 0.2, true);
    box(x0 + 3 * 0.85 - 0.15, y0 + 0.75, 0.95, 0.4, "預測 漲／跌", "FBEEEA", BULL, 9, true);
    line(x0 + 3 * 0.85 + 0.325, y0 + 0.4, x0 + 3 * 0.85 + 0.325, y0 + 0.75, true);
    s.addText("每天更新「記憶」", { x: x0, y: y0 + 0.8, w: 2.2, h: 0.3, fontFace: BODY_FONT, fontSize: 9, color: MUTED, isTextBox: true, margin: 0 });
  }
  // 圖解 3：邏輯迴歸
  {
    const x0 = 8.8 + 0.2, y0 = 2.35;
    box(x0, y0, 1.0, 0.4, "技術指標 ×w₁", WHITE, INK, 9);
    box(x0 + 1.15, y0, 1.0, 0.4, "融資融券 ×w₂", WHITE, INK, 9);
    box(x0 + 2.3, y0, 1.0, 0.4, "PTT ×w₃", WHITE, INK, 9);
    s.addText("＋", { x: x0 + 1.0, y: y0, w: 0.15, h: 0.4, fontFace: BODY_FONT, fontSize: 11, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText("＋", { x: x0 + 2.15, y: y0, w: 0.15, h: 0.4, fontFace: BODY_FONT, fontSize: 11, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    line(x0 + 1.65, y0 + 0.4, x0 + 1.65, y0 + 0.75, true);
    box(x0 + 1.0, y0 + 0.75, 1.3, 0.4, "上漲機率 0~1", "FBEEEA", BULL, 9, true);
  }

  table(s, ["模型", "技術面 AUC", "加融資融券的增量 ΔAUC（t；顯著年數）", "加 PTT 的增量（t）", "特色"], [
    [{ t: "邏輯迴歸", bold: true }, "0.548", "+0.0006（4.40；2/7）", "+0.0000（0.16）", "最簡單、最快"],
    [{ t: "XGBoost", bold: true }, "0.556", { t: "+0.0019（5.06；5/7）", color: BEAR, bold: true }, "−0.0001（−1.26）", "增量最穩定、可解釋"],
    [{ t: "LSTM", bold: true }, { t: "0.566", color: BEAR, bold: true }, "+0.0007（0.92；2/7）", "+0.0002（0.62）", "技術面基準最高、較慢"],
  ], 0.6, 4.55, [1.9, 1.5, 3.9, 2.3, 2.5], 0.42, { fontSize: 10.5, headH: 0.45 });
  callout(s, 0.6, 6.4, 12.1, 0.55, "三種模型方向一致：PTT 約為 0、融資融券為正。複雜模型的基準略高，但訊號本身很小，不是換模型就能解決。", "plain");
  footer(s);
  s.addNotes("三種模型是『由簡到繁』：邏輯迴歸最簡單，XGBoost 是很多棵決策樹，LSTM 是會記憶的神經網路。結論一致，代表不是某個模型的偶然。差別是：LSTM 技術面基準最高，XGBoost 的融資融券增量最穩定。");
}

// ============================================================ 4 一頁結論（含圖）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "附錄 · Summary");
  pageTitle(s, "一頁結論：描述得出來，多數預測不出來");

  s.addText("各種情緒訊號對「隔天盤中漲跌」的預測增量（ΔAUC，十年逐年滾動）", { x: 0.6, y: 1.5, w: 7.2, h: 0.4, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  const labels = ["PTT 淨情緒（XGBoost）", "PTT 淨情緒（LSTM）", "新聞情緒", "融資融券（LSTM）", "融資融券（XGBoost）", "散戶成交比重（XGBoost）", "法人（對照組）"];
  const values = [-0.0001, 0.0002, -0.0001, 0.0007, 0.0019, 0.0026, 0.0039];
  s.addChart(pres.charts.BAR, [{ name: "ΔAUC", labels, values }], {
    x: 0.5, y: 1.9, w: 7.4, h: 4.6, barDir: "bar", chartColors: [MUTED, MUTED, MUTED, SAND, GOLD, GOLD, NAVY_2],
    catAxisLabelFontFace: BODY_FONT, catAxisLabelFontSize: 11, catAxisOrientation: "maxMin", catAxisLabelPos: "low",
    valAxisLabelFontSize: 10, valAxisMinVal: -0.0012, valAxisMaxVal: 0.005, valAxisLabelFormatCode: "0.000",
    valGridLine: { color: BORDER, size: 0.5 }, showValue: true, dataLabelFontSize: 10, dataLabelFormatCode: "+0.0000;-0.0000",
    dataLabelPosition: "outEnd", showLegend: false,
  });

  const boxes = [
    ["沒有預測力", "PTT 留言情緒（6 種因子）、新聞情緒、文獻常見的 LSTM 預測股價作法", BULL],
    ["有，但很小", "融資融券、散戶成交比重（隔天盤中；全市場 1,144 檔也成立）、盤中開盤情緒（反指標）；扣成本後難獲利", GOLD],
    ["描述力很好", "細分情緒指數（恐慌／貪婪）、大盤散戶溫度計：崩盤時準確升高", BEAR],
  ];
  boxes.forEach((b, i) => {
    const y = 1.6 + i * 1.7;
    s.addShape("roundRect", { x: 8.2, y, w: 4.5, h: 1.5, rectRadius: 0.08, fill: { color: CARD_BG }, line: { color: BORDER, width: 1 } });
    s.addShape("rect", { x: 8.2, y: y + 0.15, w: 0.08, h: 1.2, fill: { color: b[2] }, line: { type: "none" } });
    s.addText(b[0], { x: 8.5, y: y + 0.1, w: 4.0, h: 0.42, fontFace: BODY_FONT, fontSize: 14, bold: true, color: b[2], isTextBox: true, margin: 0 });
    s.addText(b[1], { x: 8.5, y: y + 0.55, w: 4.05, h: 0.9, fontFace: BODY_FONT, fontSize: 11, color: INK2, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.25 });
  });
  s.addText("ΔAUC＝加入情緒後，比只用技術面多出的 AUC（0.5 是亂猜）。", { x: 0.6, y: 6.6, w: 7.3, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, color: MUTED, isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("這張是全場重點。PTT 完全沒有；融資融券有但很小；細分情緒能描述、不能預測。");
}

// ============================================================ 5e 散戶成交比重＋依散戶持股分組（v12 新增）
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "附錄 · Who Trades, Who Holds");
  pageTitle(s, "散戶多的股票：情緒沒有比較有用，融資融券訊號卻比較強");

  s.addText("新指標：散戶成交比重（＝100% − 法人成交比重，每天盤後可得；265 檔）", { x: 0.6, y: 1.5, w: 12, h: 0.32, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  statTile(s, 0.6, 1.88, 2.85, 1.3, "+0.0026", "加進模型的 ΔAUC（t=4.4，6/7 年為正），與融資融券 +0.0022 同級", BEAR, 22);
  statTile(s, 3.7, 1.88, 2.85, 1.3, "+0.0052", "與融資融券合用（7/7 年顯著）：兩者資訊不重疊", BEAR, 22);
  statTile(s, 6.8, 1.88, 2.85, 1.3, "−3.2 bp", "比重異常高的股票隔天盤中較差（t=−3.0），但隔夜 +3.5 bp 抵銷", GOLD, 22);
  statTile(s, 9.9, 1.88, 2.8, 1.3, "月頻 ≈ 0", "對下個月報酬 IC −0.005（t=−0.6），沒有預測力", BULL, 22);

  s.addText("依「散戶持股比例」（持股未滿 400 張者合計比率，用前一年的值）每年分三組", { x: 0.6, y: 3.4, w: 12, h: 0.32, fontFace: BODY_FONT, fontSize: 11.5, bold: true, color: INK, isTextBox: true, margin: 0 });
  table(s, ["各組的效果", "散戶少", "中", "散戶多", "多 − 少"], [
    ["PTT 情緒 → 隔天報酬（bp，2015–24）", "+2.5", "+11.1", "+1.3", { t: "−0.3（t=−0.1）", color: MUTED }],
    ["CMoney 情緒 → 隔天報酬（bp，2020–24）", "+5.0", "+5.5", "+2.6", { t: "−2.9（t=−1.5）", color: MUTED }],
    [{ t: "券資比 → 隔天盤中（每日 Rank IC）", bold: true }, "−0.049", "−0.065", "−0.068", { t: "−0.019（t=−4.0）", color: BEAR, bold: true }],
    [{ t: "融資增減 → 隔天盤中（每日 Rank IC）", bold: true }, "−0.001", "−0.013", "−0.023", { t: "−0.022（t=−5.1）", color: BEAR, bold: true }],
  ], 0.6, 3.78, [4.9, 1.6, 1.6, 1.6, 2.4], 0.42, { fontSize: 11, headH: 0.42 });
  callout(s, 0.6, 6.15, 12.1, 0.75, "與主結論一致：散戶「說」的即使在散戶最多的股票也沒有比較準；散戶「做」的在散戶多的股票更明顯。訊號仍只有千分之幾的 AUC，遠低於成本。", "plain");
  footer(s);
  s.addNotes("這頁是兩個新檢驗。第一，散戶成交比重：TEJ 有每天法人成交占比，用 100 減掉就是非法人的占比。它在日頻有和融資融券差不多大的訊號，而且兩個一起用是加總的，代表資訊不重疊。但單獨排序只有 3 bp，而且盤中少的被隔夜多的抵銷，不能交易；月頻完全沒有。要注意它是『非法人』，包含大戶，不等於純散戶。第二，如果情緒真的反映散戶，那散戶持股多的股票，情緒應該更有用。結果不是：PTT 和 CMoney 在散戶多的組都沒有比較強，CMoney 方向還相反。反而是融資融券在散戶多的組明顯更強。持股級距是年資料、不分自然人與法人，這是限制。融資增減那一列是探索性的，不在事先寫定的主要檢定內。");
}

// ============================================================ 7 盤中反指標
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "附錄 · Intraday");
  pageTitle(s, "盤中：開盤情緒轉變是「反指標」，預測收盤前半小時大盤");

  statTile(s, 0.6, 1.6, 3.9, 1.5, "−1.0 bp / 1σ", "開盤情緒轉變 → 收盤前半小時大盤報酬（t = −2.41）", BULL, 24);
  statTile(s, 4.7, 1.6, 3.9, 1.5, "前後兩段都成立", "2015–19：t = −1.98；2020–24：t = −2.07（幅度相近）", GOLD, 18);
  statTile(s, 8.8, 1.6, 3.9, 1.5, "≈ 1 bp", "台指期來回成本，遠低於股票當沖成本", BEAR, 26);

  bulletList(s, [
    "方法（仿 Renault 2017）：ΔS ＝ 開盤半小時淨情緒 − 前一晚淨情緒；預測 13:00→13:30 的加權指數報酬，並控制先前報酬。",
    "資料：每分鐘加權指數（2015–2024，2,439 日）＋開盤半小時 36.9 萬則閒聊留言（約 97 則表態/日）。",
    "方向與美國相反：美國是同向，台股是反指標——開盤情緒轉樂觀，收盤前反而略回檔。",
    "訊號只出現在收盤前半小時；預測 09:30→收盤整段則不顯著。",
  ], 0.6, 3.4, 12.0, { fontSize: 12, lineHeight: 0.66 });
  callout(s, 0.6, 6.15, 12.1, 0.75, "限制：1σ 訊號只有約 1bp（收盤前半小時標準差約 21bp），幅度很小；適合當平台的參考，不宜宣稱可獲利。", "warn");
  footer(s);
  s.addNotes("這是『說』這一側唯一顯著的預測訊號，而且前後兩個時期都成立；但幅度只有約 1bp。");
}

// ============================================================ 8 盤後/隔夜（文獻重測）
{
  const s = newSlide(pres, WHITE);
  kicker(s, "附錄 · Replication");
  pageTitle(s, "對照文獻：盤後／隔夜情緒預測隔天大盤（重測 2025 論文）");

  s.addText("邏輯迴歸，逐年滾動 2018–2024（1,705 個交易日）；情緒只用「開盤前已發」的留言，美股用開盤前已收盤的最近一日報酬", { x: 0.6, y: 1.5, w: 12.1, h: 0.5, fontFace: BODY_FONT, fontSize: 11, color: INK2, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25 });
  table(s, ["預測目標（AUC）", "只有技術面", "＋美股隔夜報酬", "＋美股＋PTT 盤後/隔夜情緒", "情緒的增量"], [
    [{ t: "收盤→收盤（論文設定）", bold: true }, "0.50（準確率 53%）", "0.72（67%）", "0.73（69%）", { t: "+0.017（6/7 年為正）", color: BEAR, bold: true }],
    [{ t: "前收→開盤（隔夜跳空）", bold: true }, "0.48", "0.85（78%）", "0.85（78%）", "+0.006"],
    [{ t: "開盤→收盤（盤中）", bold: true }, "0.54", "0.56", "0.56", { t: "+0.007（不顯著）", color: MUTED }],
  ], 0.6, 2.1, [3.3, 2.3, 2.1, 2.5, 1.9], 0.6, { fontSize: 11, headH: 0.55 });

  bulletList(s, [
    "不控制美股時，情緒看起來很強（AUC 0.63、準確率 61%）。但深夜留言只有約 6% 提到美股，情緒較像是當晚整體氣氛的反映；控制美股後只剩約 +0.01~0.017 的小增量。",
    "預測力集中在 21:30 到隔天 09:00 的留言（前晚幾乎無用）；增量來自「沒提到美股／期貨」的留言（ΔAUC +0.010），提到的留言樣本太少（約 5 則/日）、無顯著增量。",
    "論文報告準確率 87.5% → 90.6%（我們推測測試集僅約 32 天，多猜對 1 天＝+3.1pp）。我們的 32 天區塊準確率標準差達 9~10pp，這種差距在雜訊範圍內。",
  ], 0.6, 4.65, 12.1, { fontSize: 11.5, lineHeight: 0.58 });
  callout(s, 0.6, 6.38, 12.1, 0.64, "再控制台指夜盤（2019–2024）：情緒的增量從約 +0.012 縮到約 +0.003~0.006（準確率增量≤0.6pp，區間含 0），夜盤本身 AUC 0.77；歐股與新聞尚未控制。", "warn");
  footer(s);
  s.addNotes("我們照文獻的設定重測。沒控制美股時情緒看起來很強；我們檢查過深夜留言，只有約 6% 提到美股，所以不是在轉述美股，較像整體氣氛，機制無法由內容證明。控制美股之後只剩很小的增量。論文的 90% 疑似小測試集的運氣。");
}

// ============================================================ 11 文獻對照
{
  const s = newSlide(pres, LIGHT_BG);
  kicker(s, "附錄 · Literature");
  pageTitle(s, "文獻對照：我們的結果與既有研究一致或補充");

  table(s, ["文獻", "主要發現", "與我們的關係"], [
    ["Jing, Wu & Wang 2021\nExpert Systems with Applications", "投資人情緒＋LSTM 預測股價", "我選讀的論文；台股十年資料重做測不到情緒增量"],
    ["Antweiler & Frank 2004\nJournal of Finance", "留言板對報酬顯著但經濟上很小；對波動有預測力", "一致：PTT 爆量與波動有關，但增量極小"],
    ["Renault 2017\nJournal of Banking & Finance", "開盤半小時情緒 → 最後半小時大盤（美國同向）", "我們在台股是反指標（t = −2.41）"],
    ["Discover Computing 2025\nTaiwan FinBERT + CNN-BiLSTM-SA", "PTT 盤後閒聊 → 大盤隔日漲跌 87.5% → 90.6%", "重測：控制美股後只剩小增量；測試集疑似很小"],
    ["Bollen et al. 2011 ／ Econ Journal Watch 重新檢視", "Twitter 情緒預測道瓊（準確率 86.7%）；後有文章重新檢視該研究", "亮眼的早期結果需要重測驗證，我們的做法即為此"],
  ], 0.6, 1.6, [4.1, 4.2, 3.8], 0.78, { fontSize: 10.5, headH: 0.5 });
  callout(s, 0.6, 6.3, 12.1, 0.6, "定位：以十年、265 檔、2,021 萬則留言的嚴格滾動驗證，檢驗既有文獻的說法。", "plain");
  footer(s);
  s.addNotes("文獻大致分兩派：早期報正面結果、後來重測常常不成立。我們的結果比較接近第二派。");
}

pres.writeFile({ fileName: "final_report_v20.pptx" }).then(() => console.log("done"));
