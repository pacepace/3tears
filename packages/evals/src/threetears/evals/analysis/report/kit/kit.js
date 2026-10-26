// Minimal chart kit: pixel-true SVG rendered at container width, shared tooltip, sortable tables, and one
// declarative entry point: every div.chart[data-kind][data-chart] is drawn from its JSON (drawCharts).
const NS="http://www.w3.org/2000/svg";
function S(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const k in attrs){if(k==="text")e.textContent=attrs[k];else e.setAttribute(k,attrs[k]);}if(parent)parent.appendChild(e);return e;}
function H(tag,attrs={},html){const e=document.createElement(tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(html!=null)e.innerHTML=html;return e;}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function lin(d0,d1,r0,r1){const f=x=>r0+(x-d0)/(d1-d0)*(r1-r0);f.inv=y=>d0+(y-r0)/(r1-r0)*(d1-d0);f.d=[d0,d1];return f;}
function logs(d0,d1,r0,r1){const a=Math.log10(d0),b=Math.log10(d1);const f=x=>r0+(Math.log10(x)-a)/(b-a)*(r1-r0);f.d=[d0,d1];return f;}
function ticks(a,b,n=5){if(n>1){let k=n,t=ticksRaw(a,b,k);while(t.length>n+1&&k>1){k--;t=ticksRaw(a,b,k);}return t;}return ticksRaw(a,b,n);}
function ticksRaw(a,b,n=5){const step=Math.pow(10,Math.floor(Math.log10((b-a)/n)));const err=(b-a)/n/step;const m=err>=7.5?10:err>=3.5?5:err>=1.5?2:1;const s=m*step;const out=[];for(let v=Math.ceil(a/s)*s;v<=b+1e-9;v+=s)out.push(+v.toFixed(10));return out;}
// nice ticks whose labels are all distinct: a format that rounds (an integer count) would otherwise print
// "1" at 0.5 and at 1, so the step widens until each label names one position
function kFmtTicks(a,b,n,f){for(let k=n;k>1;k--){const t=ticks(a,b,k),L=t.map(f);if(new Set(L).size===L.length)return t;}return ticks(a,b,2);}
// explicit tick values, thinned left to right so no two labels touch at the drawn width
function kThinTicks(vals,x,f){const out=[];let last=-1e9;for(const v of vals){const half=f(v).length*3.2;if(x(v)-half<last+6)continue;out.push(v);last=x(v)+half;}return out;}
const kClamp=(v,a,b)=>Math.max(Math.min(a,b),Math.min(Math.max(a,b),v));
const fmt={
  pct:(v,d=1)=>v.toFixed(d)+"%",
  usd:v=>v>=100?"$"+Math.round(v).toLocaleString():v>=10?"$"+v.toFixed(1):v>=1?"$"+v.toFixed(2):"$"+v.toFixed(3),
  s:v=>v.toFixed(v<10?1:0)+"s", ms:v=>Math.round(v)+" ms", x:v=>v.toFixed(1)+"×"
};
// named formats for chart data: "name" or "name:digits", with a leading "+" to sign positive values.
// Without digits a value prints as many decimals as it needs, up to two (usd and ms keep their own rounding).
const kShort=(v,d)=>d==null?String(+(+v).toFixed(2)):(+v).toFixed(d);
const K_FMTS={
  num:kShort, int:v=>Math.round(v).toLocaleString("en-US"), pct:(v,d)=>kShort(v,d)+"%", pts:(v,d)=>kShort(v,d)+" pts",
  usd:(v,d)=>d==null?fmt.usd(v):"$"+v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}),
  usdk:v=>Math.abs(v)>=1000?"$"+(v/1000).toFixed(1)+"k":"$"+Math.round(v).toLocaleString("en-US"),
  ms:(v,d)=>(d==null?String(Math.round(v)):v.toFixed(d))+" ms", s:(v,d)=>kShort(v,d)+" s",
  dur:v=>v>=1000?kShort(v/1000)+"s":kShort(v)+"ms", x:(v,d)=>v.toFixed(d==null?1:d)+"×", k:(v,d)=>(v/1000).toFixed(d==null?1:d)+"k"
};
function namedFmt(spec){const m=typeof spec==="string"&&/^(\+?)([a-z]+)(?::(\d))?$/.exec(spec);
  if(!m||!Object.hasOwn(K_FMTS,m[2]))throw new Error(`fmt ${JSON.stringify(spec)} is not one of ${Object.keys(K_FMTS).join(", ")} (optionally "+name" or "name:digits")`);
  const f=K_FMTS[m[2]],d=m[3]==null?null:+m[3],signed=m[1]==="+";return v=>{const t=f(v,d);return signed&&v>0?"+"+t:t;};}
// chart data is untrusted text: escape everything that lands in markup, and accept colors only as token names
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
function kTok(c,fallback){if(c==null)return fallback;if(typeof c!=="string"||!/^--[a-z][a-z0-9-]*$/.test(c))throw new Error(`color ${JSON.stringify(c)} is not a token name such as "--s1"`);return c;}
// a number that reaches a style or a size: finite, or refused, so data can never write CSS of its own
function kNum(v,dflt,what){if(v==null)return dflt;if(typeof v!=="number"||!Number.isFinite(v))throw new Error(`${what} ${JSON.stringify(v)} is not a finite number`);return v;}
// the kit reads a null field as one not set, as the package's models do: nulls are dropped from every object
function kNoNulls(v){if(Array.isArray(v))return v.map(kNoNulls);if(v&&typeof v==="object")return Object.fromEntries(Object.entries(v).filter(([,x])=>x!=null).map(([k,x])=>[k,kNoNulls(x)]));return v;}
const K_STATUS=["good","warn","bad","neutral"];
function kStatus(k){if(!K_STATUS.includes(k))throw new Error(`status ${JSON.stringify(k)} is not one of ${K_STATUS.join(", ")}`);return k;}
// a tooltip is a list of text lines; the first is set in bold
function tipHTML(t){if(t==null)return null;return (Array.isArray(t)?t:[t]).map((l,i)=>i?esc(l):`<b>${esc(l)}</b>`).join("<br>");}
// text width in the chart font, measured; charts redraw once web fonts load, so widths match what is drawn
let kCanvas;function textW(t,size=12,weight=400){kCanvas=kCanvas||document.createElement("canvas").getContext("2d");
  kCanvas.font=`${weight} ${size}px ${css("--f-body")||"sans-serif"}`;return kCanvas.measureText(String(t)).width;}
// tooltip
const tip=H("div",{id:"tip",role:"tooltip"});document.addEventListener("DOMContentLoaded",()=>document.body.appendChild(tip));
function showTip(html,x,y){tip.innerHTML=html;tip.classList.add("on");const w=tip.offsetWidth,h=tip.offsetHeight;
  let L=x+14,T=y+14;if(L+w>innerWidth-8)L=x-w-14;if(T+h>innerHeight-8)T=y-h-14;tip.style.left=Math.max(8,L)+"px";tip.style.top=Math.max(8,T)+"px";}
function hideTip(){tip.classList.remove("on");}
document.addEventListener("pointermove",e=>{const t=e.target.closest&&e.target.closest("[data-tip]");if(t)showTip(t.getAttribute("data-tip"),e.clientX,e.clientY);else hideTip();});
document.addEventListener("focusin",e=>{const t=e.target.closest&&e.target.closest("[data-tip]");if(t){const r=t.getBoundingClientRect();showTip(t.getAttribute("data-tip"),r.right,r.top);}});
document.addEventListener("focusout",hideTip);
// responsive render: fn(el,width) called on resize and once fonts load; the returned redraw() carries its observer as .ro
function chart(el,fn){const go=()=>{const w=Math.max(180,Math.floor(el.clientWidth));if(el._w===w)return;el._w=w;el.innerHTML="";fn(el,w);};
  go();const ro=new ResizeObserver(go);ro.observe(el);const redraw=()=>{el._w=0;go();};redraw.ro=ro;kRedraws.add(redraw);return redraw;}
const kRedraws=new Set();if(document.fonts)document.fonts.ready.then(()=>{for(const r of kRedraws)r();});
function svgIn(el,w,h){return S("svg",{width:w,height:h,viewBox:`0 0 ${w} ${h}`,role:"img"},el);}
// hit target helper: transparent rect larger than mark; html is already-escaped tooltip markup
function hit(g,x,y,w,h,html){const r=S("rect",{x,y,width:w,height:h,fill:"transparent","data-tip":html,tabindex:0},g);return r;}
// sortable tables: th.sort[data-k], rows rendered by fn(sortedRows)
function sortable(tableEl,rows,render,defKey,defDir=-1){let key=defKey,dir=defDir;
  const ths=tableEl.querySelectorAll("th.sort");
  const draw=()=>{ths.forEach(th=>{th.removeAttribute("aria-sort");if(th.dataset.k===key)th.setAttribute("aria-sort",dir<0?"descending":"ascending");});
    const s=[...rows].sort((a,b)=>{const A=a[key],B=b[key];return (A>B?1:A<B?-1:0)*dir;});render(s);};
  ths.forEach(th=>{th.tabIndex=0;const f=()=>{if(key===th.dataset.k)dir=-dir;else{key=th.dataset.k;dir=-1;}draw();};th.onclick=f;th.onkeydown=e=>{if(e.key==="Enter")f();};});draw();}
function pill(kind,text){const ic={good:"✓",warn:"!",bad:"✕",neutral:"–"}[kind];return `<span class="pill ${kind}"><span aria-hidden="true">${ic}</span>${text}</span>`;}
// colors are applied via style="fill:var(--token)" so theme changes need no re-render
// label placement: try right, left, above, below, then nudges up and down; avoids other labels and obstacle boxes [x0,y0,x1,y1].
// A label's box is its measured 12px text plus a little air, so two placed labels never touch; above and below clear
// the label's own mark (radius it.r, default 5.5). An obstacle may carry a fifth number, its cost (default 10): a
// reference line costs 1, so crossing it beats colliding with a label. A spot costs more the further its box is from
// its own mark, and a spot nearer another mark than its own costs 3, so a label strays from its mark only to avoid a
// collision and never sits beside another arm's mark. marks: [[x,y]] of every mark drawn (default: the items').
function placeLabels(items,bounds,obstacles=[],marks){const W=(t,strong)=>textW(t,12,strong?600:400)+3;const placed=[...obstacles];
  const pts=marks||items.map(it=>[it.px,it.py]);const near=(b,x,y)=>Math.hypot(Math.max(b[0]-x,0,x-b[2]),Math.max(b[1]-y,0,y-b[3]));
  for(const it of items){const w=W(it.text,it.strong),g=(it.r==null?5.5:it.r)+2,R=it.px+g+1.5,L=it.px-g-1.5,up=it.py-g-4.5,dn=it.py+g+13;
    const opts=[[R,it.py+4,"start"],[L,it.py+4,"end"],[it.px,up,"middle"],[it.px,dn,"middle"]];
    for(const d of [-6,6,12,-12,24,-24,36,-36])opts.push([R,it.py+4+d,"start"],[L,it.py+4+d,"end"]);
    opts.push([it.px,up-14,"middle"],[it.px,dn+14,"middle"]);
    const box=o=>{const x0=o[2]==="end"?o[0]-w:o[2]==="middle"?o[0]-w/2:o[0];return [x0,o[1]-12.5,x0+w,o[1]+4];};
    const score=b=>{let s=b[0]<bounds[0]||b[2]>bounds[2]||b[1]<bounds[1]||b[3]>bounds[3]?100:0;
      for(const p of placed)if(b[0]<p[2]&&b[2]>p[0]&&b[1]<p[3]&&b[3]>p[1])s+=p[4]==null?10:p[4];
      const own=near(b,it.px,it.py);s+=own*.06;
      if(pts.some(([x,y])=>(x!==it.px||y!==it.py)&&near(b,x,y)<Math.max(own,4)))s+=3;return s;};
    let pick=opts[0],bs=1e9;opts.forEach((o,i)=>{const sc=score(box(o))+i*.08;if(sc<bs){bs=sc;pick=o;}});
    it.x=pick[0];it.y=pick[1];it.anchor=pick[2];placed.push(box(pick));}
  return items;}
// marks: one shape per arm where color alone would carry identity; hollow marks a failure or an absent setting
function mark(g,shape,x,y,r,color,hollow){const st=hollow?`fill:var(--surface);stroke:var(${color});stroke-width:2`:`fill:var(${color});stroke:var(--surface);stroke-width:2`;
  if(shape==="square")return S("rect",{x:x-r*.88,y:y-r*.88,width:r*1.76,height:r*1.76,rx:1.5,style:st},g);
  if(shape==="triangle")return S("path",{d:`M${x},${y-r*1.1}L${x+r*1.05},${y+r*.8}L${x-r*1.05},${y+r*.8}Z`,style:st},g);
  if(shape==="diamond")return S("path",{d:`M${x},${y-r*1.15}L${x+r*1.1},${y}L${x},${y+r*1.15}L${x-r*1.1},${y}Z`,style:st},g);
  return S("circle",{cx:x,cy:y,r,style:st},g);}
const K_SHAPES=["circle","square","triangle","diamond"];
function kShape(s){if(s==null)return "circle";if(!K_SHAPES.includes(s))throw new Error(`shape ${JSON.stringify(s)} is not one of ${K_SHAPES.join(", ")}`);return s;}
// patterns: a fill told apart by its texture, not its color alone (one status hue, several causes). The box is
// washed in its token and the texture drawn over it in the token as plain lines and dots, clipped to the box by
// geometry, so a pattern references nothing outside its own marks.
const K_PATTERNS=["hatch","cross","dots","rows","cols"];
function kPatternName(name){if(!K_PATTERNS.includes(name))throw new Error(`pattern ${JSON.stringify(name)} is not one of ${K_PATTERNS.join(", ")}`);return name;}
function kTexture(g,name,x,y,w,h,color,opacity=1){kPatternName(name);const t=S("g",{style:`opacity:${opacity}`},g),P=6;
  S("rect",{x,y,width:w,height:h,rx:2,style:`fill:color-mix(in oklab, var(${color}) 22%, var(--surface));stroke:var(${color});stroke-width:1`},t);
  if(w<=0||h<=0)return t;const d=[];
  // a diagonal from (sx,sy) along (1,dir) for h, cut to the box's x range
  const diag=(sx,sy,dir)=>{const a=Math.max(0,x-sx),b=Math.min(h,x+w-sx);if(b>a)d.push(`M${sx+a},${sy+dir*a}L${sx+b},${sy+dir*b}`);};
  if(name==="hatch"||name==="cross")for(let k=-h;k<w;k+=P)diag(x+k,y+h,-1);
  if(name==="cross")for(let k=-h;k<w;k+=P)diag(x+k,y,1);
  if(name==="rows")for(let v=y+P/2;v<y+h;v+=P)d.push(`M${x},${v}H${x+w}`);
  if(name==="cols")for(let v=x+P/2;v<x+w;v+=P)d.push(`M${v},${y}V${y+h}`);
  if(d.length)S("path",{d:d.join(""),style:`fill:none;stroke:var(${color});stroke-width:${name==="cross"?1:1.4}`},t);
  if(name==="dots")for(let i=x+P/2;i<x+w;i+=P)for(let j=y+P/2;j<y+h;j+=P)S("circle",{cx:i,cy:j,r:1.4,style:`fill:var(${color})`},t);
  return t;}

// ---------- arm naming ----------
// arms: [{id, role:'control'|'challenger', alias?, spec:{model, ...}}]; attrs: [{k, fmt(v)}] in display order
// short = alias, else only the attributes that differ from control (model trimmed of the shared prefix); tag = A, B, C…
// (AA after Z), never P, which is the control's alone
const kCandidateTag=n=>{let t="";for(n++;n;n=Math.floor((n-1)/26))t=String.fromCharCode(65+(n-1)%26)+t;return t;};
function armLabels(arms,attrs,controlName="Production"){
  const ctl=arms.find(a=>a.role==="control");const fmtA=(k,v)=>{const a=attrs.find(x=>x.k===k);return a&&a.fmt?a.fmt(v):String(v);};
  const trimModel=(m,base)=>{if(!base||m===base)return m;let i=0;while(i<m.length&&i<base.length&&m[i]===base[i])i++;
    if(!(i===base.length&&/[-_ .]/.test(m[i]||"")))while(i>0&&!/[-_ .]/.test(m[i-1]))i--;const rest=m.slice(i).replace(/^[-_ .]+/,"");return i>=4&&rest?rest:m;};
  let n=0;
  for(const a of arms){a.long=attrs.map(x=>fmtA(x.k,a.spec[x.k])).join(" · ");
    if(a===ctl){a.short=a.alias||controlName;a.tag="P";continue;}
    do a.tag=kCandidateTag(n++);while(a.tag==="P");
    if(a.alias){a.short=a.alias;continue;}
    const diffs=attrs.filter(x=>a.spec[x.k]!==ctl.spec[x.k]);
    a.short=diffs.map(x=>x.k==="model"?trimModel(a.spec.model,ctl.spec.model):(x.sfmt?x.sfmt(a.spec[x.k]):fmtA(x.k,a.spec[x.k]))).join(" · ")||a.long;}
  // collisions: append the next attribute that separates them
  const seen={};for(const a of arms){(seen[a.short]=seen[a.short]||[]).push(a);}
  for(const k in seen)if(seen[k].length>1)for(const a of seen[k]){const extra=attrs.find(x=>seen[k].some(b=>b!==a&&b.spec[x.k]!==a.spec[x.k]));if(extra)a.short+=" · "+fmtA(extra.k,a.spec[extra.k]);}
  return arms;}
// a tag's letter on its arm's color: on-fill, except on the control's ink-gray, which is light in the dark theme
const kOnTag=c=>c==="--ink-2"?"--surface":"--on-fill";
function armChip(a,color){return `<span class="armchip"><span class="tag" style="background:var(${color});color:var(${kOnTag(color)})">${a.tag}</span>${a.short}</span>`;}
function kArmChip(tag,label,color){const s=H("span",{class:"armchip"});const t=H("span",{class:"tag"}),c=kTok(color,"--ink-2");t.style.background=`var(${c})`;t.style.color=`var(${kOnTag(c)})`;t.textContent=tag;s.append(t,document.createTextNode(label));return s;}

// ---------- horizontal bars (optionally diverging, stacked, grouped, with whiskers and a reference line) ----------
// rows:[{label, sl?, value | segments:[{value,color}], lo?, hi?, color, strong?, muted?, note?, tip?, group?}]; lo without hi
// is an interval open above
// opt:{domain:[a,b], fmt, tickFmt?, ticks?, ref?:{value,label}, origin?, originLabel?, level?, labelW?, rowH?}
function barsH(el,w,rows,opt){const rowH=opt.rowH||28,narrow=w<480,tagW=rows.some(r=>r.tag!=null)?22:0,lw=opt.labelW||(narrow?Math.min(130,w*.36):Math.min(220,w*.3));
  const m={l:lw,r:narrow?52:70,t:opt.ref||opt.level||opt.originLabel?20:6,b:22};
  const grouped=rows.some(r=>r.group!=null);let gaps=0;rows.forEach((r,i)=>{if(i&&r.group!==rows[i-1].group)gaps++;});
  const h=m.t+m.b+rowH*rows.length+gaps*14+(grouped?14:0);const s=svgIn(el,w,h);const [d0,d1]=opt.domain;const x=opt.log?logs(d0,d1,m.l,w-m.r):lin(d0,d1,m.l,w-m.r);
  const tf=opt.tickFmt||opt.fmt;const tw=Math.max(...[d0,d1,(d0+d1)/2].map(v=>tf(v).length))*6.4+14;
  const tv=opt.ticks?kThinTicks(opt.ticks,x,tf):kFmtTicks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/tw)),tf);
  for(const v of tv){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-6,"text-anchor":"middle",class:"tick",text:tf(v)},s);}
  const o=opt.origin||0;const z=x(kClamp(o,d0,d1));S("line",{x1:z,x2:z,y1:m.t-2,y2:h-m.b,class:"axis"},s);
  if(opt.originLabel)S("text",{x:z+4,y:m.t-8,class:"tick",text:opt.originLabel},s);
  let y0=m.t+(grouped?14:0);
  rows.forEach((r,i)=>{if(grouped&&(i===0||r.group!==rows[i-1].group)){if(i)y0+=14;S("text",{x:0,y:y0-4,class:"lbl-strong",text:r.group==null?"":r.group},s);}
    const cy=y0+rowH/2;y0+=rowH;const g=S("g",{},s);const val=r.segments?r.segments.reduce((a,sg)=>a+sg.value,0):r.value;const v=kClamp(val,d0,d1);
    // the label, else its short label, else (where the row has a tag) the tag alone, else the short or long label cut
    const maxC=Math.floor((m.l-14-tagW)/6.6);let lab=r.label;if(lab.length>maxC)lab=r.sl&&r.sl.length<=maxC?r.sl:r.tag!=null?"":(r.sl||lab).slice(0,maxC-1)+"…";
    S("text",{x:m.l-10-tagW,y:cy+4,"text-anchor":"end",class:r.strong?"lbl-strong":"lbl",text:lab},g);
    if(r.tag!=null){const tc=r.color||"--ink-2";S("rect",{x:m.l-26,y:cy-8,width:16,height:16,rx:3,style:`fill:var(${tc})`},g);
      S("text",{x:m.l-18,y:cy+3.5,"text-anchor":"middle",class:"tag",style:`fill:var(${kOnTag(tc)});font-size:10px;font-weight:600`,text:r.tag},g);}
    if(r.segments){let acc=o;r.segments.forEach((sg,j)=>{const a=x(kClamp(acc,d0,d1)),b=x(kClamp(acc+sg.value,d0,d1));const c=sg.color||r.color||"--s1";
        const sx=Math.min(a,b)+(j?1:0),sw=Math.max(0,Math.abs(b-a)-(j?1:0)),sy=cy-(rowH*.28),sh=rowH*.56;
        if(sg.pattern!=null)kTexture(g,sg.pattern,sx,sy,sw,sh,c,r.muted?.45:1);
        else S("rect",{x:sx,y:sy,width:sw,height:sh,rx:2,style:`fill:var(${c});opacity:${r.muted?.45:1}`},g);acc+=sg.value;});}
    else{const x0=Math.min(z,x(v)),bw=Math.max(1.5,Math.abs(x(v)-z));
      S("rect",{x:x0,y:cy-(rowH*.28),width:bw,height:rowH*.56,rx:3,style:`fill:var(${r.color||"--s1"});opacity:${r.muted?.45:1}`},g);}
    // a low end with no high end is an interval open above: its whisker runs on to the axis's end, capped only at
    // the low end and opening into a chevron there, never read as a point
    const open=r.lo!=null&&r.hi==null,hi=open?d1:r.hi;
    if(r.lo!=null){S("line",{x1:x(Math.max(d0,r.lo)),x2:x(Math.min(d1,hi)),y1:cy,y2:cy,style:"stroke:var(--ink);stroke-width:1.25"},g);
      for(const e of open?[r.lo]:[r.lo,r.hi])S("line",{x1:x(kClamp(e,d0,d1)),x2:x(kClamp(e,d0,d1)),y1:cy-4,y2:cy+4,style:"stroke:var(--ink);stroke-width:1.25"},g);
      if(open){const ex=x(d1);S("path",{d:`M${ex-5},${cy-4}L${ex},${cy}L${ex-5},${cy+4}`,style:"fill:none;stroke:var(--ink);stroke-width:1.25"},g);}}
    const right=val>=o;const edge=r.lo!=null?(right?x(Math.min(d1,hi)):x(Math.max(d0,r.lo))):x(v);
    let nt=(r.note!=null?r.note:opt.fmt(val));if(right&&edge+6+nt.length*6.4>w)nt=opt.fmt(val);let nw=nt.length*6.4;const flip=!right&&edge-6-nw<m.l+4;
    const past=Math.max(z,x(Math.min(d1,hi!=null?hi:o)));
    if(flip&&past+6+nw>w){nt=opt.fmt(val);nw=nt.length*6.4;}
    S("text",{x:right?edge+6:flip?past+6:edge-6,y:cy+4,"text-anchor":right||flip?"start":"end",class:r.strong?"lbl-strong":"tick",text:nt},g);
    hit(g,0,cy-rowH/2,w,rowH,tipHTML(r.tip||[r.label,opt.fmt(val)+(r.lo==null?"":open?` (from ${opt.fmt(r.lo)} up, no upper bound)`:` (${opt.fmt(r.lo)} to ${opt.fmt(r.hi)})`)]));});
  if(opt.ref){const rx=x(opt.ref.value);S("line",{x1:rx,x2:rx,y1:m.t-6,y2:h-m.b,class:"floor"},s);if(opt.ref.label)S("text",{x:rx+4,y:m.t-8,class:"tick",text:opt.ref.label},s);}
  if(opt.level)S("text",{x:w,y:m.t-8,"text-anchor":"end",class:"tick",text:`${opt.level} intervals`},s);
  return s;}

// ---------- line chart over ordered categories ----------
// cats:[label], series:[{name, color, values:[...], lo?:[], hi?:[], strong?, dash?, endLabel?}], opt:{domain, fmt, xTitle?, h?, level?}
function lineChart(el,w,cats,series,opt){const narrow=w<480;const h=opt.h||(narrow?260:300);const m={l:44,r:narrow?92:130,t:opt.level?24:14,b:opt.xTitle?44:28};const s=svgIn(el,w,h);
  const x=i=>m.l+(cats.length===1?0:i/(cats.length-1))*(w-m.l-m.r),[d0,d1]=opt.domain,y=lin(d0,d1,h-m.b,m.t);
  for(const v of kFmtTicks(d0,d1,5,opt.fmt)){S("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"},s);if(v>d0)S("text",{x:m.l-8,y:y(v)+4,"text-anchor":"end",class:"tick",text:opt.fmt(v)},s);}
  cats.forEach((c,i)=>S("text",{x:x(i),y:h-m.b+16,"text-anchor":"middle",class:"tick",text:c},s));
  if(opt.xTitle)S("text",{x:(m.l+w-m.r)/2,y:h-6,"text-anchor":"middle",class:"tick",text:opt.xTitle},s);
  if(opt.level)S("text",{x:w-m.r,y:12,"text-anchor":"end",class:"tick",text:`${opt.level} intervals`},s);
  const ends=[];const obs=[];
  for(const se of series){if(se.lo)S("path",{d:"M"+se.hi.map((v,i)=>`${x(i)},${y(v)}`).join("L")+"L"+se.lo.map((v,i)=>[i,v]).reverse().map(([i,v])=>`${x(i)},${y(v)}`).join("L")+"Z",style:`fill:var(${se.color});opacity:.12`},s);}
  for(const se of series){S("path",{d:"M"+se.values.map((v,i)=>`${x(i)},${y(v)}`).join("L"),style:`fill:none;stroke:var(${se.color});stroke-width:${se.strong?2.5:2};${se.dash?"stroke-dasharray:5 4;":""}`},s);
    se.values.forEach((v,i)=>{const g=S("g",{},s);S("circle",{cx:x(i),cy:y(v),r:se.strong?4.5:3.5,style:`fill:var(${se.color});stroke:var(--surface);stroke-width:2`},g);obs.push([x(i)-5,y(v)-5,x(i)+5,y(v)+5]);
      hit(g,x(i)-10,y(v)-10,20,20,tipHTML([`${se.name} · ${cats[i]}`,`${opt.fmt(v)}${se.lo?` (${opt.fmt(se.lo[i])}–${opt.fmt(se.hi[i])})`:""}`]));});
    ends.push({px:x(cats.length-1),py:y(se.values.at(-1)),text:se.endLabel||se.name,strong:se.strong});}
  // end labels stacked to avoid collisions, to the right of the last point
  ends.sort((a,b)=>a.py-b.py);let last=-1e9;for(const e of ends){e.y=Math.max(e.py+4,last+16);last=e.y;}
  const maxE=Math.floor((m.r-12)/6.6);for(const e of ends)S("text",{x:e.px+10,y:e.y,class:e.strong?"lbl-strong":"lbl",text:e.text.length>maxE?e.text.slice(0,maxE-1)+"…":e.text},s);
  return s;}

// ---------- forest plot: change against a control, with interval and an optional margin ----------
// rows:[{label, value, lo, hi, color, strong?, note?, group?, tip?}] opt:{domain:[a,b], fmt, ref?:{value,label} (margin band from value up), level?}
function forest(el,w,rows,opt){const narrow=w<480,rowH=opt.rowH||30,need=Math.max(...rows.map(r=>r.label.length))*6.6+18,lw=opt.labelW||Math.min(need,narrow?w*.42:Math.min(230,w*.32));
  const m={l:lw,r:narrow?60:96,t:22,b:24};let gaps=0;rows.forEach((r,i)=>{if(i&&r.group!==rows[i-1].group)gaps++;});
  const h=m.t+m.b+rowH*rows.length+gaps*14+(rows.some(r=>r.group)?14:0);const s=svgIn(el,w,h);const [d0,d1]=opt.domain;const x=lin(d0,d1,m.l,w-m.r);
  if(opt.ref){const mx=x(Math.max(d0,opt.ref.value));S("rect",{x:mx,y:m.t,width:x(d1)-mx,height:h-m.t-m.b,style:"fill:var(--good-wash);opacity:.6"},s);
    S("line",{x1:mx,x2:mx,y1:m.t-4,y2:h-m.b,class:"floor"},s);if(opt.ref.label)S("text",{x:mx+4,y:narrow&&opt.level?h-m.b-6:m.t-8,class:"tick",text:opt.ref.label},s);}
  for(const v of kFmtTicks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/(Math.max(opt.fmt(d0).length,opt.fmt(d1).length)*6.4+16))),opt.fmt)){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-6,"text-anchor":"middle",class:"tick",text:opt.fmt(v)},s);}
  S("line",{x1:x(0),x2:x(0),y1:m.t-4,y2:h-m.b,class:"axis",style:"stroke-width:1.5"},s);
  if(opt.level)S("text",{x:w-m.r,y:m.t-8,"text-anchor":"end",class:"tick",text:`${opt.level} intervals`},s);
  let y0=m.t+(rows.some(r=>r.group)?14:0);
  rows.forEach((r,i)=>{if(r.group&&(i===0||r.group!==rows[i-1].group)){if(i)y0+=14;S("text",{x:0,y:y0-4,class:"lbl-strong",text:r.group},s);}
    const cy=y0+rowH/2;y0+=rowH;const g=S("g",{},s);const cl=v=>x(Math.max(d0,Math.min(d1,v)));
    const maxC=Math.floor((m.l-14)/6.6);const lab=r.label.length<=maxC?r.label:(r.sl&&r.sl.length<=maxC)?r.sl:r.label.slice(0,maxC-1)+"…";
    S("text",{x:m.l-10,y:cy+4,"text-anchor":"end",class:r.strong?"lbl-strong":"lbl",text:lab},g);
    S("line",{x1:cl(r.lo),x2:cl(r.hi),y1:cy,y2:cy,style:`stroke:var(${r.color||"--s1"});stroke-width:3;stroke-linecap:round`},g);
    S("rect",{x:cl(r.value)-5,y:cy-5,width:10,height:10,rx:2,style:`fill:var(${r.color||"--s1"});stroke:var(--surface);stroke-width:2`},g);
    let nt=r.note!=null?r.note:opt.fmt(r.value);if(nt.length*6.4>m.r-10)nt=opt.fmt(r.value);S("text",{x:w-m.r+8,y:cy+4,class:r.strong?"lbl-strong":"tick",text:nt},g);
    hit(g,0,cy-rowH/2,w,rowH,tipHTML(r.tip||[r.label,`${opt.fmt(r.value)} [${opt.fmt(r.lo)}, ${opt.fmt(r.hi)}]`]));});
  return s;}

// ---------- declarative renderers: each takes the chart's JSON (kit/API.md "Chart data") ----------
function kNeed(d,keys){for(const k of keys)if(d[k]==null)throw new Error(`"${k}" is required`);}
function kList(d,k){if(!Array.isArray(d[k]))throw new Error(`"${k}" must be a list`);return d[k];}
// GENERATOR §5.2: an interval-bearing chart prints its level, so a chart drawing intervals without one is refused
function kNeedLevel(d,rows){if(!d.level&&rows.some(r=>r.lo!=null||r.hi!=null))throw new Error(`rows carry intervals, so "level" is required`);}
const kLabelW=(d,w)=>(w<480?d.label_width_narrow:d.label_width)||null;
const kIsEm=(d,id)=>d.emphasis!=null&&id!=null&&id===d.emphasis;

function kBars(el,w,d){kNeed(d,["domain","fmt"]);const rows=kList(d,"rows");kNeedLevel(d,rows);const f=namedFmt(d.fmt);
  const ref=d.reference;const pref=ref&&ref.kind==="no_preference";
  barsH(el,w,rows.map(r=>({...r,sl:r.short_label,color:kTok(r.color,"--s1"),strong:r.strong||kIsEm(d,r.id),
      segments:r.segments&&r.segments.map(sg=>({value:sg.value,color:kTok(sg.color,null),pattern:sg.pattern}))})),
    {domain:d.domain,fmt:f,tickFmt:d.tick_fmt?namedFmt(d.tick_fmt):null,ticks:d.ticks,ref:ref&&!pref?ref:null,origin:pref?ref.value:0,originLabel:pref?ref.label:null,
     level:d.level,labelW:kLabelW(d,w),rowH:d.row_height,log:d.log});}

function kForest(el,w,d){kNeed(d,["domain","fmt","level"]);const rows=kList(d,"rows");
  forest(el,w,rows.map(r=>({...r,sl:r.short_label,color:kTok(r.color,"--s1"),strong:r.strong||kIsEm(d,r.id)})),
    {domain:d.domain,fmt:namedFmt(d.fmt),ref:d.reference||null,level:d.level,labelW:kLabelW(d,w),rowH:d.row_height});}

function kLines(el,w,d){kNeed(d,["domain","fmt"]);const series=kList(d,"series");const cats=kList(d,"categories");
  if(!d.level&&series.some(se=>se.lo||se.hi))throw new Error(`series carry bands, so "level" is required`);
  lineChart(el,w,cats,series.map(se=>({...se,color:kTok(se.color,"--s1"),endLabel:se.end_label})),{domain:d.domain,fmt:namedFmt(d.fmt),xTitle:d.x_title,h:d.height,level:d.level});}

// scatter: two measures that trade off. gate=true shades where both references pass (gate_scatter).
function kScatter(el,w,d,gate){kNeed(d,["x_domain","y_domain","x_fmt","y_fmt"]);const pts=kList(d,"points");
  if(gate&&(!d.reference||!d.x_reference))throw new Error(`gate_scatter needs "reference" (y) and "x_reference" (x)`);
  const xf=namedFmt(d.x_fmt),yf=namedFmt(d.y_fmt),narrow=w<560;const [x0,x1]=d.x_domain,[y0,y1]=d.y_domain;
  const yt=d.y_ticks||kFmtTicks(y0,y1,5,yf);const h=d.height||Math.round(Math.min(380,Math.max(300,w*.45)));
  const m={l:Math.max(30,Math.max(...yt.map(v=>yf(v).length))*6.6+16),r:18,t:16,b:40};const s=svgIn(el,w,h);
  const x=d.x_log?logs(x0,x1,m.l,w-m.r):lin(x0,x1,m.l,w-m.r),y=lin(y0,y1,h-m.b,m.t);const obs=[];
  const passSide=(r,lo,hi,dflt)=>((r.kind||dflt)==="budget"?[lo,r.value]:[r.value,hi]);
  if(gate){const [ax,bx]=passSide(d.x_reference,x0,x1,"budget"),[ay,by]=passSide(d.reference,y0,y1,"floor");
    S("rect",{x:x(ax),y:y(by),width:x(bx)-x(ax),height:y(ay)-y(by),style:"fill:var(--good-wash)"},s);
    if(d.region_label){S("text",{x:x(ax)+8,y:y(by)+16,class:"lbl",style:"fill:var(--good);font-weight:600",text:d.region_label},s);obs.push([x(ax)+8,y(by)+6,x(ax)+8+d.region_label.length*6.6,y(by)+19]);}}
  if(d.price_line){const p=d.price_line.value,yAt=v=>-(v/Math.max(1e-9,p)),lx=Math.min(x1,p*-y0);
    S("path",{d:`M${x(0)},${y(0)}L${x(x1)},${y(0)}L${x(x1)},${y(Math.max(y0,yAt(x1)))}L${x(lx)},${y(Math.max(y0,yAt(lx)))}Z`,style:"fill:var(--good-wash)"},s);}
  for(const v of yt){S("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"},s);if(v>y0)S("text",{x:m.l-8,y:y(v)+4,"text-anchor":"end",class:"tick",text:yf(v)},s);}
  for(const v of kThinTicks(d.x_ticks||kFmtTicks(x0,x1,5,xf),x,xf)){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-m.b+16,"text-anchor":"middle",class:"tick",text:xf(v)},s);}
  if(d.x_title)S("text",{x:(m.l+w-m.r)/2,y:h-6,"text-anchor":"middle",class:"tick",text:d.x_title},s);
  if(d.y_title)S("text",{x:m.l+6,y:h-m.b-8,class:"tick",text:d.y_title},s);
  if(d.x_reference){const rx=x(d.x_reference.value);S("line",{x1:rx,x2:rx,y1:m.t,y2:h-m.b,class:"floor"},s);obs.push([rx-2,m.t,rx+2,h-m.b,1]);
    if(d.x_reference.label){S("text",{x:rx+4,y:m.t+12,class:"tick",text:d.x_reference.label},s);obs.push([rx+4,m.t+2,rx+4+d.x_reference.label.length*6.4,m.t+15]);}}
  if(d.reference){const ry=y(d.reference.value);S("line",{x1:m.l,x2:w-m.r,y1:ry,y2:ry,class:"floor"},s);obs.push([m.l,ry-2,w-m.r,ry+2,1]);
    if(d.reference.label){const lw=d.reference.label.length*6.4;S("text",{x:w-m.r,y:ry+13,"text-anchor":"end",class:"tick",text:d.reference.label},s);obs.push([w-m.r-lw,ry+1.5,w-m.r,ry+16.5]);}}
  if(d.price_line){const p=d.price_line.value,yAt=v=>-(v/Math.max(1e-9,p)),lx=Math.min(x1,p*-y0);
    S("line",{x1:x(0),y1:y(0),x2:x(lx),y2:y(Math.max(y0,yAt(lx))),style:"stroke:var(--good);stroke-width:1.5;stroke-dasharray:5 4"},s);
    if(d.price_line.label){const mx=lx*.55,t=d.price_line.label,tx=x(mx)+8,ty=y(Math.max(y0,yAt(mx)))+16;S("text",{x:tx,y:ty,class:"tick",style:"fill:var(--good)",text:t},s);obs.push([tx,ty-10,tx+t.length*6.4,ty+3]);}
    if(d.price_line.region_label){const t=d.price_line.region_label;S("text",{x:w-m.r-8,y:m.t+16,"text-anchor":"end",class:"lbl",style:"fill:var(--good);font-weight:600",text:t},s);obs.push([w-m.r-8-t.length*6.6,m.t+6,w-m.r-8,m.t+19]);}}
  const at=p=>[x(kClamp(p.x,x0,x1)),y(kClamp(p.y,y0+(y1-y0)*.025,y1))];const byId=Object.fromEntries(pts.filter(p=>p.id!=null).map(p=>[p.id,p]));
  if(d.frontier){let best=-Infinity;const fr=[];for(const p of [...pts].sort((a,b)=>a.x-b.x))if(p.y>best){fr.push(p);best=p.y;}
    S("path",{d:"M"+fr.map(p=>at(p).join(",")).join("L"),style:"fill:none;stroke:var(--base);stroke-width:1.5"},s);}
  for(const pa of d.paths||[]){const ps=kList(pa,"ids").map(id=>{if(!byId[id])throw new Error(`path names point "${id}", which is not in "points"`);return byId[id];});
    S("path",{d:"M"+ps.map(p=>at(p).join(",")).join("L"),style:`fill:none;stroke:var(${kTok(pa.color,"--base")});stroke-width:${kNum(pa.width,2,"path width")};opacity:${kNum(pa.opacity,1,"path opacity")}`},s);}
  for(const p of pts){const [px,py]=at(p),r=kNum(p.size,5.5,"point size"),g=S("g",{},s);const c=kTok(p.color,"--s1");const mk=mark(g,kShape(p.shape),px,py,r,c,p.hollow);if(p.muted)mk.style.opacity=.45;
    if(kIsEm(d,p.id))S("circle",{cx:px,cy:py,r:r+4.5,style:"fill:none;stroke:var(--ink);stroke-width:1.5"},g);obs.push([px-r-2,py-r-2,px+r+2,py+r+2]);
    hit(g,px-12,py-12,24,24,tipHTML(p.tip||[p.label||p.id||"",`${xf(p.x)} · ${yf(p.y)}`]));}
  // the emphasized point is labelled first, then left to right, so it gets the best spot
  const labs=pts.map(p=>({p,text:narrow?p.narrow_label:p.label})).filter(l=>l.text).map(l=>{const [px,py]=at(l.p);return {...l,px,py,r:kNum(l.p.size,5.5,"point size")};})
    .map(l=>({...l,strong:kIsEm(d,l.p.id)})).sort((a,b)=>(b.strong?1:0)-(a.strong?1:0)||a.px-b.px);
  placeLabels(labs,[m.l,0,w,h-m.b],obs,pts.map(at)).forEach(l=>S("text",{x:l.x,y:l.y,"text-anchor":l.anchor,class:kIsEm(d,l.p.id)?"lbl-strong":"lbl",text:l.text},s));
  return s;}

// one rate per candidate against a floor: dot, interval, floor dashed
function kDotInterval(el,w,d){kNeed(d,["domain","fmt"]);const rows=kList(d,"rows");kNeedLevel(d,rows);const f=namedFmt(d.fmt);const [d0,d1]=d.domain;
  const rowH=d.row_height||28,need_=Math.max(...rows.map(r=>r.label.length))*6.6+18,m={l:kLabelW(d,w)||Math.min(need_,w*.42),r:14,t:22,b:22},maxC=Math.floor((m.l-12)/6.6);
  const h=m.t+m.b+rowH*rows.length;const s=svgIn(el,w,h);const x=lin(d0,d1,m.l,w-m.r);
  const tw=Math.max(f(d0).length,f(d1).length)*6.4+16;
  for(const v of d.ticks?kThinTicks(d.ticks,x,f):kFmtTicks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/tw)),f)){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-6,"text-anchor":"middle",class:"tick",text:f(v)},s);}
  rows.forEach((r,i)=>{const cy=m.t+rowH*i+rowH/2,c=kTok(r.color,"--s1"),g=S("g",{},s);
    S("text",{x:m.l-8,y:cy+4,"text-anchor":"end",class:r.strong||kIsEm(d,r.id)?"lbl-strong":"lbl",text:r.label.length<=maxC?r.label:r.label.slice(0,maxC-1)+"…"},g);
    if(r.lo!=null)S("line",{x1:x(kClamp(r.lo,d0,d1)),x2:x(kClamp(r.hi,d0,d1)),y1:cy,y2:cy,style:`stroke:var(${c});stroke-width:3;stroke-linecap:round;opacity:.35`},g);
    mark(g,kShape(r.shape),x(kClamp(r.value,d0,d1)),cy,5,c,r.hollow);
    hit(g,m.l,cy-rowH/2,w-m.l-m.r,rowH,tipHTML(r.tip||[r.label,f(r.value)+(r.lo!=null&&r.hi!=null?` (${f(r.lo)} to ${f(r.hi)})`:"")]));});
  if(d.reference){const rx=x(d.reference.value);S("line",{x1:rx,x2:rx,y1:m.t-4,y2:h-m.b,class:"floor"},s);if(d.reference.label)S("text",{x:rx+4,y:m.t-8,class:"tick",text:d.reference.label},s);}
  if(d.level)S("text",{x:w-m.r,y:m.t-8,"text-anchor":"end",class:"tick",text:`${d.level} intervals`},s);
  return s;}

// latency distributions: one bar per row from "from" to "to" (p50 to p95), grouped, with a budget line
function kRangeBars(el,w,d){kNeed(d,["domain","fmt"]);const groups=kList(d,"groups");const f=namedFmt(d.fmt);const [d0,d1]=d.domain;
  const band=g=>kList(g,"rows").length*13+20,m={l:Math.min(190,w*.36),r:16,t:16,b:26};const h=m.t+m.b+groups.reduce((a,g)=>a+band(g),0);const s=svgIn(el,w,h);const x=lin(d0,d1,m.l,w-m.r);
  const tw=Math.max(f(d0).length,f(d1).length)*6.4+16;
  for(const v of d.ticks?kThinTicks(d.ticks,x,f):kFmtTicks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/tw)),f)){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-8,"text-anchor":"middle",class:"tick",text:f(v)},s);}
  let top=m.t;for(const gr of groups){const y0=top+10;top+=band(gr);
    S("text",{x:m.l-12,y:y0+gr.rows.length*13/2+3,"text-anchor":"end",class:"lbl-strong",text:w<480&&gr.short_label?gr.short_label:gr.label},s);
    gr.rows.forEach((r,i)=>{const cy=y0+i*13+4,c=kTok(r.color,"--s1"),g=S("g",{},s);
      S("line",{x1:x(kClamp(r.from,d0,d1)),x2:x(kClamp(r.to,d0,d1)),y1:cy,y2:cy,style:`stroke:var(${c});stroke-width:6;stroke-linecap:round`},g);
      mark(g,kShape(r.shape),x(kClamp(r.to,d0,d1)),cy,4,c,r.hollow);
      hit(g,m.l,cy-6.5,w-m.l-m.r,13,tipHTML(r.tip||[`${r.label} · ${gr.label}`,`${f(r.from)} to ${f(r.to)}`]));});}
  if(d.reference){const rx=x(d.reference.value);S("line",{x1:rx,x2:rx,y1:m.t-8,y2:h-m.b,class:"floor"},s);if(d.reference.label)S("text",{x:rx+4,y:m.t-2,class:"tick",text:d.reference.label},s);}
  return s;}

// HTML renderers: drawn once, sized by CSS
function kTxt(tag,cls,t){const e=H(tag,cls?{class:cls}:{});e.textContent=t==null?"":t;return e;}
function kPill(kind,text){const s=H("span",{},pill(kStatus(kind),esc(text)));return s.firstChild;}
function kWithTip(e,t){if(t!=null){e.setAttribute("data-tip",tipHTML(t));e.tabIndex=0;}return e;}
const kGridCols=n=>`auto repeat(${n},minmax(0,1fr))`;

// pass/fail per candidate × task: pill, the binding reasons, cost; the cheapest passing cell highlighted
function kGateMatrix(el,d){const cols=kList(d,"columns"),rows=kList(d,"rows");const cf=d.cost_fmt?namedFmt(d.cost_fmt):null;
  const wrap=H("div",{class:"tbl"}),t=H("table",{class:"gate"}),hr=H("tr");hr.appendChild(kTxt("th","",d.row_header||""));for(const c of cols)hr.appendChild(kTxt("th","",c.label));
  const th=H("thead");th.appendChild(hr);const tb=H("tbody");
  for(const r of rows){const tr=H("tr"),td=H("td",{class:"task"});td.append(kTxt("b","",r.label));if(r.detail)td.append(kTxt("span","",r.detail));if(r.flag)td.append(kTxt("span","xc",r.flag));tr.appendChild(td);
    const cells=kList(r,"cells");if(cells.length!==cols.length)throw new Error(`row "${r.label}" has ${cells.length} cells for ${cols.length} columns`);
    for(const c of cells){const cd=H("td",{class:c.pick?"cell pick":"cell"});if(c.label)cd.appendChild(kPill(c.status,c.label));
      if(c.reasons&&c.reasons.length)cd.appendChild(kTxt("span","why",c.reasons.join("; ")));
      if(c.cost!=null){if(!cf)throw new Error(`cells carry "cost", so "cost_fmt" is required`);cd.appendChild(kTxt("span","cost",cf(c.cost)+(d.cost_unit?" "+d.cost_unit:"")));}
      if(c.pick&&d.pick_label)cd.appendChild(kTxt("span","pickmark",d.pick_label));tr.appendChild(cd);}
    tb.appendChild(tr);}
  t.append(th,tb);wrap.appendChild(t);el.appendChild(wrap);}

// two crossed factors: a tile per cell with the metric, a bar, cost and latency, and a status pill
function kLattice(el,d){kNeed(d,["fmt"]);const cols=kList(d,"columns"),rows=kList(d,"rows");const f=namedFmt(d.fmt),cf=d.cost_fmt?namedFmt(d.cost_fmt):null,lf=d.latency_fmt?namedFmt(d.latency_fmt):null;
  const bd=d.bar_domain||null;const box=H("div",{class:"lattice"});box.style.gridTemplateColumns=kGridCols(cols.length);box.appendChild(kTxt("div","hd",""));
  for(const c of cols)box.appendChild(kTxt("div","hd",c.label));
  for(const r of rows){const rh=H("div",{class:"rh"});if(r.color){const sw=H("span",{class:"sw"});sw.style.background=`var(${kTok(r.color)})`;rh.appendChild(sw);}rh.append(document.createTextNode(r.label));box.appendChild(rh);
    const cells=kList(r,"cells");if(cells.length!==cols.length)throw new Error(`row "${r.label}" has ${cells.length} cells for ${cols.length} columns`);
    for(const c of cells){const cell=kWithTip(H("div",{class:kIsEm(d,c.id)?"cell rec":"cell"}),c.tip);cell.appendChild(kTxt("span","acc",f(c.value)));
      if(bd){const bar=H("div",{class:"bar"}),i=H("i");i.style.width=kClamp((c.value-bd[0])/(bd[1]-bd[0])*100,0,100)+"%";bar.appendChild(i);cell.appendChild(bar);}
      const meta=[c.cost!=null&&cf?cf(c.cost)+(d.cost_unit?" "+d.cost_unit:""):null,c.latency!=null&&lf?lf(c.latency):null].filter(Boolean);
      if(meta.length)cell.appendChild(kTxt("span","meta",meta.join(" · ")));if(c.status)cell.appendChild(kPill(c.status,c.status_label||""));box.appendChild(cell);}}
  el.appendChild(box);}

// grids of cells (heatmap, difference heatmap, confusion) share the .mx layout
function kMxGrid(cols,rows,cell){const box=H("div",{class:"mx"});box.style.gridTemplateColumns=kGridCols(cols.length);box.appendChild(H("div"));
  for(const c of cols)box.appendChild(kTxt("div","h",c));
  rows.forEach((r,i)=>{box.appendChild(kTxt("div","rl",r.label));const cells=kList(r,"cells");if(cells.length!==cols.length)throw new Error(`row "${r.label}" has ${cells.length} cells for ${cols.length} columns`);
    cells.forEach((c,j)=>box.appendChild(cell(c,i,j)));});return box;}
function kMxCell(bg,fg,main,sub,t){const e=kWithTip(H("div",{class:"c"}),t);e.style.background=bg;e.style.color=fg;e.appendChild(kTxt("b","",main));if(sub!=null)e.appendChild(kTxt("span","",sub));return e;}

// target matrix of cells: sequential shading in six steps over the domain, n per cell, outline below the floor
function kHeatmap(el,d){kNeed(d,["domain","fmt"]);const f=namedFmt(d.fmt),nf=d.n_fmt?namedFmt(d.n_fmt):null,[d0,d1]=d.domain;const cols=kList(d,"columns").map(c=>c.label);
  el.appendChild(kMxGrid(cols,kList(d,"rows"),(c,i,j)=>{const k=kClamp(Math.floor((c.value-d0)/(d1-d0)*6),0,5);
    const e=kMxCell(`var(--seq-${k})`,`var(--seq-t${k})`,f(c.value),c.n!=null&&nf?nf(c.n):null,c.tip||[`${d.rows[i].label} · ${cols[j]}`,f(c.value)+(c.n!=null?` · n = ${c.n.toLocaleString("en-US")}`:"")]);
    if(d.reference&&c.value<d.reference.value)e.classList.add("low");return e;}));}

// difference between two configs: diverging shading, gray within ±neutral, full color at ±saturate
function kDiffHeatmap(el,d){kNeed(d,["fmt"]);const f=namedFmt(d.fmt),df=d.detail_fmt?namedFmt(d.detail_fmt):null,nu=d.neutral==null?.5:d.neutral,sat=d.saturate||8;const cols=kList(d,"columns").map(c=>c.label);
  el.appendChild(kMxGrid(cols,kList(d,"rows"),(c,i,j)=>{const v=c.value,mag=Math.min(1,Math.abs(v)/sat),flat=Math.abs(v)<nu;
    const bg=flat?"var(--div-mid)":`color-mix(in oklab, var(${v>0?"--div-pos":"--div-neg"}) ${Math.round(18+mag*72)}%, var(--surface))`;
    return kMxCell(bg,mag>.55&&!flat?"var(--on-fill)":"var(--ink)",f(v),c.detail!=null&&df?df(c.detail):null,c.tip||[`${d.rows[i].label} · ${cols[j]}`,f(v)]);}));}

// errors between classes, rows normalized to 100: the diagonal tinted, off-diagonal cells warn and bad by share
function kConfusion(el,d){const labels=kList(d,"labels"),M=kList(d,"matrix");const f=namedFmt(d.fmt||"pct:1"),warnAt=d.warn_at==null?1:d.warn_at,badAt=d.bad_at==null?3:d.bad_at;
  if(M.length!==labels.length)throw new Error(`"matrix" has ${M.length} rows for ${labels.length} labels`);
  const rows=M.map((r,i)=>{if(!Array.isArray(r))throw new Error(`"matrix" row ${i} must be a list`);const tot=r.reduce((a,v)=>a+v,0);return {label:labels[i],cells:d.normalized===false?r.map(v=>tot?v/tot*100:0):r};});
  el.appendChild(kMxGrid(labels,rows,(v,i,j)=>{const bg=i===j?"var(--seq-1)":v>=badAt?"var(--bad-wash)":v>=warnAt?"var(--warn-wash)":"var(--sunk)";
    return kMxCell(bg,"var(--ink)",f(v),null,[`${labels[i]} → ${labels[j]}`,`${f(v)} of the row`]);}));}

// what failures look like: cards with a meta line, then the question, fields, judge reason, excerpt and a note
function kExamples(el,d){const cards=kList(d,"cards");const box=H("div",{class:"ex"});if(d.card_min_rem!=null)box.style.gridTemplateColumns=`repeat(auto-fit,minmax(min(100%,${kNum(d.card_min_rem,0,"card_min_rem")}rem),1fr))`;
  for(const c of cards){const card=H("div",{class:"excard"}),top=H("div",{class:"top"});
    for(const it of c.meta||[])top.appendChild(it.status?kPill(it.status,it.label):it.tag?kArmChip(it.tag,it.label,it.color):kTxt("span",it.mono?"mono":"",it.text));card.appendChild(top);
    if(c.question!=null)card.appendChild(kTxt("div","q",c.question));
    if(c.fields&&c.fields.length){const dl=H("dl");for(const fl of c.fields){dl.appendChild(kTxt("dt","",fl.term));dl.appendChild(kTxt("dd",fl.tone==="bad"?"bad":"",fl.text));}card.appendChild(dl);}
    if(c.reason!=null)card.appendChild(kTxt("div","reason",c.reason));if(c.excerpt!=null)card.appendChild(kTxt("pre","",c.excerpt));if(c.note!=null)card.appendChild(kTxt("div","note",c.note));
    box.appendChild(card);}
  el.appendChild(box);}

// legend: [{label, color?, mark?: dot|square|line|circle|triangle|diamond, hollow?, opacity?, pattern?}]
function kLegend(items){if(!Array.isArray(items))throw new Error(`"legend" must be a list`);const box=H("div",{class:"legend"});
  for(const it of items){const sp=H("span"),c=kTok(it.color,"--ink-2"),mk=it.mark||"dot";
    if(it.pattern!=null){const sv=S("svg",{width:12,height:12,style:"vertical-align:-1px;margin-right:.4rem"});kTexture(sv,it.pattern,.5,.5,11,11,c);sp.appendChild(sv);}
    else if(mk==="dot"||mk==="square"){const i=H("i",mk==="square"?{class:"sq"}:{});i.style.background=it.hollow?"var(--surface)":`var(${c})`;if(it.hollow)i.style.boxShadow=`inset 0 0 0 2px var(${c})`;sp.appendChild(i);}
    else if(mk==="line"){const sv=S("svg",{width:22,height:10,style:"vertical-align:0;margin-right:.35rem"});S("line",{x1:1,x2:21,y1:5,y2:5,style:`stroke:var(${c});stroke-width:2;opacity:${kNum(it.opacity,1,"legend opacity")}`},sv);sp.appendChild(sv);}
    else{const sv=S("svg",{width:14,height:14,style:"vertical-align:-2px;margin-right:.35rem"});mark(sv,kShape(mk),7,7,5,c,it.hollow);sp.appendChild(sv);}
    sp.append(document.createTextNode(it.label));box.appendChild(sp);}
  return box;}

// ---------- dispatcher ----------
const K_KINDS={bars:{draw:kBars},diverging_bars:{draw:kBars},forest:{draw:kForest},lines:{draw:kLines},
  scatter_frontier:{draw:(el,w,d)=>kScatter(el,w,d,false)},gate_scatter:{draw:(el,w,d)=>kScatter(el,w,d,true)},
  dot_interval:{draw:kDotInterval},range_bars:{draw:kRangeBars},
  gate_matrix:{html:kGateMatrix},lattice:{html:kLattice},heatmap:{html:kHeatmap},diff_heatmap:{html:kDiffHeatmap},confusion:{html:kConfusion},examples:{html:kExamples}};
const K_NOT_DRAWN={table:"a table is written as an HTML table (sortable() adds sorting); the kit does not draw it"};
const K_PANEL_TEXT=["title","subtitle","tag","color"];
function kUnmount(st){for(const r of st.ro){r.ro.disconnect();kRedraws.delete(r);}st.ro=[];}
function kMount(host){const st=host._kit;kUnmount(st);host.innerHTML="";const kind=host.getAttribute("data-kind");
  try{if(!Object.hasOwn(K_KINDS,kind))throw new Error(Object.hasOwn(K_NOT_DRAWN,kind)?K_NOT_DRAWN[kind]:`unknown kind; the kit draws ${Object.keys(K_KINDS).join(", ")}`);const R=K_KINDS[kind];
    let d;try{d=JSON.parse(host.getAttribute("data-chart"));}catch(e){throw new Error(`data-chart is not JSON (${e.message})`);}
    if(!d||typeof d!=="object"||Array.isArray(d))throw new Error("data-chart must be a JSON object");d=kNoNulls(d);
    if(d.legend)host.appendChild(kLegend(d.legend));
    const one=(box,data)=>{if(R.html)R.html(box,data);else st.ro.push(chart(box,(e,w)=>R.draw(e,w,data)));};
    if(d.panels==null){const box=H("div");host.appendChild(box);one(box,d);return;}
    // small multiples: each panel is the chart's own fields overlaid with the panel's
    kList(d,"panels");const {panels,legend,panel_min_rem,...shared}=d;const grid=H("div",{class:"mult"});
    if(panel_min_rem!=null)grid.style.gridTemplateColumns=`repeat(auto-fill,minmax(min(100%,${kNum(panel_min_rem,0,"panel_min_rem")}rem),1fr))`;host.appendChild(grid);
    for(const p of panels){const fig=H("div",{class:"fig"});grid.appendChild(fig);const h4=H("h4");if(p.tag)h4.appendChild(kArmChip(p.tag,p.title||"",p.color));else h4.textContent=p.title||"";fig.appendChild(h4);
      if(p.subtitle)fig.appendChild(kTxt("div","sub",p.subtitle));
      if(Object.keys(p).some(k=>!K_PANEL_TEXT.includes(k))){const box=H("div",{class:"chart"});fig.appendChild(box);one(box,{...shared,...p});}}}
  catch(e){kUnmount(st);host.innerHTML="";console.error(`kit: chart "${kind}"${host.id?" #"+host.id:""}: ${e.message}`);}}
// draws every div.chart[data-kind][data-chart] under root, once; a later change to data-chart redraws that chart
function drawCharts(root=document){const sel="div.chart[data-kind][data-chart]";const els=[...root.querySelectorAll(sel)];if(root.matches&&root.matches(sel))els.unshift(root);
  for(const el of els){if(el._kit)continue;el._kit={ro:[]};new MutationObserver(()=>kMount(el)).observe(el,{attributes:true,attributeFilter:["data-chart","data-kind"]});kMount(el);}}
// a page script that builds chart data from its own input sets it with setChart(); the JSON is pure data
function setChart(el,data){el.setAttribute("data-chart",JSON.stringify(data));}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>drawCharts());else setTimeout(()=>drawCharts());
