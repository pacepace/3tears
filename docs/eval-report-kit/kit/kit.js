// Minimal chart kit: pixel-true SVG rendered at container width, shared tooltip, sortable tables.
const NS="http://www.w3.org/2000/svg";
function S(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const k in attrs){if(k==="text")e.textContent=attrs[k];else e.setAttribute(k,attrs[k]);}if(parent)parent.appendChild(e);return e;}
function H(tag,attrs={},html){const e=document.createElement(tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(html!=null)e.innerHTML=html;return e;}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function lin(d0,d1,r0,r1){const f=x=>r0+(x-d0)/(d1-d0)*(r1-r0);f.inv=y=>d0+(y-r0)/(r1-r0)*(d1-d0);f.d=[d0,d1];return f;}
function logs(d0,d1,r0,r1){const a=Math.log10(d0),b=Math.log10(d1);const f=x=>r0+(Math.log10(x)-a)/(b-a)*(r1-r0);f.d=[d0,d1];return f;}
function ticks(a,b,n=5){if(n>1){let k=n,t=ticksRaw(a,b,k);while(t.length>n+1&&k>1){k--;t=ticksRaw(a,b,k);}return t;}return ticksRaw(a,b,n);}
function ticksRaw(a,b,n=5){const step=Math.pow(10,Math.floor(Math.log10((b-a)/n)));const err=(b-a)/n/step;const m=err>=7.5?10:err>=3.5?5:err>=1.5?2:1;const s=m*step;const out=[];for(let v=Math.ceil(a/s)*s;v<=b+1e-9;v+=s)out.push(+v.toFixed(10));return out;}
const fmt={
  pct:(v,d=1)=>v.toFixed(d)+"%",
  usd:v=>v>=100?"$"+Math.round(v).toLocaleString():v>=10?"$"+v.toFixed(1):v>=1?"$"+v.toFixed(2):"$"+v.toFixed(3),
  s:v=>v.toFixed(v<10?1:0)+"s", ms:v=>Math.round(v)+" ms", x:v=>v.toFixed(1)+"×"
};
// tooltip
const tip=H("div",{id:"tip",role:"tooltip"});document.addEventListener("DOMContentLoaded",()=>document.body.appendChild(tip));
function showTip(html,x,y){tip.innerHTML=html;tip.classList.add("on");const w=tip.offsetWidth,h=tip.offsetHeight;
  let L=x+14,T=y+14;if(L+w>innerWidth-8)L=x-w-14;if(T+h>innerHeight-8)T=y-h-14;tip.style.left=Math.max(8,L)+"px";tip.style.top=Math.max(8,T)+"px";}
function hideTip(){tip.classList.remove("on");}
document.addEventListener("pointermove",e=>{const t=e.target.closest&&e.target.closest("[data-tip]");if(t)showTip(t.getAttribute("data-tip"),e.clientX,e.clientY);else hideTip();});
document.addEventListener("focusin",e=>{const t=e.target.closest&&e.target.closest("[data-tip]");if(t){const r=t.getBoundingClientRect();showTip(t.getAttribute("data-tip"),r.right,r.top);}});
document.addEventListener("focusout",hideTip);
// responsive render: fn(el,width) called on resize
const charts=[];
function chart(el,fn){const go=()=>{const w=Math.max(180,Math.floor(el.clientWidth));if(el._w===w)return;el._w=w;el.innerHTML="";fn(el,w);};new ResizeObserver(go).observe(el);go();return ()=>{el._w=0;go();};}
function svgIn(el,w,h){return S("svg",{width:w,height:h,viewBox:`0 0 ${w} ${h}`,role:"img"},el);}
// hit target helper: transparent rect larger than mark
function hit(g,x,y,w,h,html){const r=S("rect",{x,y,width:w,height:h,fill:"transparent","data-tip":html,tabindex:0},g);return r;}
// sortable tables: th.sort[data-k], rows rendered by fn(sortedRows)
function sortable(tableEl,rows,render,defKey,defDir=-1){let key=defKey,dir=defDir;
  const ths=tableEl.querySelectorAll("th.sort");
  const draw=()=>{ths.forEach(th=>{th.removeAttribute("aria-sort");if(th.dataset.k===key)th.setAttribute("aria-sort",dir<0?"descending":"ascending");});
    const s=[...rows].sort((a,b)=>{const A=a[key],B=b[key];return (A>B?1:A<B?-1:0)*dir;});render(s);};
  ths.forEach(th=>{th.tabIndex=0;const f=()=>{if(key===th.dataset.k)dir=-dir;else{key=th.dataset.k;dir=-1;}draw();};th.onclick=f;th.onkeydown=e=>{if(e.key==="Enter")f();};});draw();}
function pill(kind,text){const ic={good:"✓",warn:"!",bad:"✕",neutral:"–"}[kind];return `<span class="pill ${kind}"><span aria-hidden="true">${ic}</span>${text}</span>`;}
// colors are applied via style="fill:var(--token)" so theme changes need no re-render
// label placement: try right, left, above, below, then nudges; avoids other labels and obstacle boxes [x0,y0,x1,y1]
function placeLabels(items,bounds,obstacles=[]){const W=t=>t.length*6.6+2;const placed=[...obstacles];
  const hitAny=b=>b[0]<bounds[0]||b[2]>bounds[2]||b[1]<bounds[1]||b[3]>bounds[3]||placed.some(p=>b[0]<p[2]&&b[2]>p[0]&&b[1]<p[3]&&b[3]>p[1]);
  for(const it of items){const w=W(it.text);const opts=[[it.px+9,it.py+4,"start"],[it.px-9,it.py+4,"end"],[it.px,it.py-10,"middle"],[it.px,it.py+18,"middle"]];
    for(const d of [12,-12,24,-24])opts.push([it.px+9,it.py+4+d,"start"],[it.px-9,it.py+4+d,"end"]);
    const score=b=>(b[0]<bounds[0]||b[2]>bounds[2]||b[1]<bounds[1]||b[3]>bounds[3]?100:0)+placed.filter(p=>b[0]<p[2]&&b[2]>p[0]&&b[1]<p[3]&&b[3]>p[1]).length;
    let pick=opts[0],bs=1e9;for(const o of opts){const x0=o[2]==="end"?o[0]-w:o[2]==="middle"?o[0]-w/2:o[0];const b=[x0,o[1]-10,x0+w,o[1]+3];const sc=score(b);if(sc<bs){bs=sc;pick=o;}if(sc===0)break;}
    it.x=pick[0];it.y=pick[1];it.anchor=pick[2];const x0=it.anchor==="end"?it.x-w:it.anchor==="middle"?it.x-w/2:it.x;placed.push([x0,it.y-10,x0+w,it.y+3]);}
  return items;}

// ---------- arm naming ----------
// arms: [{id, role:'control'|'challenger', alias?, spec:{model, ...}}]; attrs: [{k, fmt(v)}] in display order
// short = alias, else only the attributes that differ from control (model trimmed of the shared prefix); tag = A, B, C…
function armLabels(arms,attrs,controlName="Production"){
  const ctl=arms.find(a=>a.role==="control");const fmtA=(k,v)=>{const a=attrs.find(x=>x.k===k);return a&&a.fmt?a.fmt(v):String(v);};
  const trimModel=(m,base)=>{if(!base||m===base)return m;let i=0;while(i<m.length&&i<base.length&&m[i]===base[i])i++;
    if(!(i===base.length&&/[-_ .]/.test(m[i]||"")))while(i>0&&!/[-_ .]/.test(m[i-1]))i--;const rest=m.slice(i).replace(/^[-_ .]+/,"");return i>=4&&rest?rest:m;};
  let n=0;
  for(const a of arms){a.long=attrs.map(x=>fmtA(x.k,a.spec[x.k])).join(" · ");
    if(a===ctl){a.short=a.alias||controlName;a.tag="P";continue;}
    a.tag=String.fromCharCode(65+n++);
    if(a.alias){a.short=a.alias;continue;}
    const diffs=attrs.filter(x=>a.spec[x.k]!==ctl.spec[x.k]);
    a.short=diffs.map(x=>x.k==="model"?trimModel(a.spec.model,ctl.spec.model):(x.sfmt?x.sfmt(a.spec[x.k]):fmtA(x.k,a.spec[x.k]))).join(" · ")||a.long;}
  // collisions: append the next attribute that separates them
  const seen={};for(const a of arms){(seen[a.short]=seen[a.short]||[]).push(a);}
  for(const k in seen)if(seen[k].length>1)for(const a of seen[k]){const extra=attrs.find(x=>seen[k].some(b=>b!==a&&b.spec[x.k]!==a.spec[x.k]));if(extra)a.short+=" · "+fmtA(extra.k,a.spec[extra.k]);}
  return arms;}
function armChip(a,color){return `<span class="armchip"><span class="tag" style="background:var(${color})">${a.tag}</span>${a.short}</span>`;}

// ---------- horizontal bars (optionally diverging, with whiskers and a reference line) ----------
// rows:[{label, value, lo?, hi?, color, strong?, muted?, note?, tip?}] opt:{domain:[a,b], fmt, ref?:{value,label}, labelW?, rowH?}
function barsH(el,w,rows,opt){const rowH=opt.rowH||28,narrow=w<480,lw=opt.labelW||(narrow?Math.min(130,w*.36):Math.min(220,w*.3));const m={l:lw,r:narrow?52:70,t:opt.ref?20:6,b:22};
  const h=m.t+m.b+rowH*rows.length;const s=svgIn(el,w,h);const [d0,d1]=opt.domain;const x=lin(d0,d1,m.l,w-m.r);
  const tf=opt.tickFmt||opt.fmt;const tw=Math.max(...[d0,d1,(d0+d1)/2].map(v=>tf(v).length))*6.4+14;
  for(const v of ticks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/tw)))){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-6,"text-anchor":"middle",class:"tick",text:tf(v)},s);}
  const z=x(Math.max(d0,Math.min(d1,0)));S("line",{x1:z,x2:z,y1:m.t-2,y2:h-m.b,class:"axis"},s);
  rows.forEach((r,i)=>{const cy=m.t+rowH*i+rowH/2;const g=S("g",{},s);const v=Math.max(d0,Math.min(d1,r.value));
    const maxC=Math.floor((m.l-14)/6.6);let lab=r.label;if(lab.length>maxC)lab=(r.sl&&r.sl.length<=maxC)?r.sl:lab.slice(0,maxC-1)+"…";
    S("text",{x:m.l-10,y:cy+4,"text-anchor":"end",class:r.strong?"lbl-strong":"lbl",text:lab},g);
    const x0=Math.min(z,x(v)),bw=Math.max(1.5,Math.abs(x(v)-z));
    S("rect",{x:x0,y:cy-(rowH*.28),width:bw,height:rowH*.56,rx:3,style:`fill:var(${r.color||"--s1"});opacity:${r.muted?.45:1}`},g);
    if(r.lo!=null){S("line",{x1:x(Math.max(d0,r.lo)),x2:x(Math.min(d1,r.hi)),y1:cy,y2:cy,style:"stroke:var(--ink);stroke-width:1.25"},g);
      for(const e of [r.lo,r.hi])S("line",{x1:x(Math.max(d0,Math.min(d1,e))),x2:x(Math.max(d0,Math.min(d1,e))),y1:cy-4,y2:cy+4,style:"stroke:var(--ink);stroke-width:1.25"},g);}
    const right=r.value>=0;const edge=r.lo!=null?(right?x(Math.min(d1,r.hi)):x(Math.max(d0,r.lo))):x(v);
    let nt=(r.note!=null?r.note:opt.fmt(r.value));if(right&&edge+6+nt.length*6.4>w)nt=opt.fmt(r.value);let nw=nt.length*6.4;const flip=!right&&edge-6-nw<m.l+4;
    if(flip&&Math.max(z,x(Math.min(d1,r.hi!=null?r.hi:0)))+6+nw>w){nt=opt.fmt(r.value);nw=nt.length*6.4;}
    S("text",{x:right?edge+6:flip?Math.max(z,x(Math.min(d1,r.hi!=null?r.hi:0)))+6:edge-6,y:cy+4,"text-anchor":right||flip?"start":"end",class:r.strong?"lbl-strong":"tick",text:nt},g);
    hit(g,0,cy-rowH/2,w,rowH,r.tip||`<b>${r.label}</b><br>${opt.fmt(r.value)}`);});
  if(opt.ref){const rx=x(opt.ref.value);S("line",{x1:rx,x2:rx,y1:m.t-6,y2:h-m.b,class:"floor"},s);S("text",{x:rx+4,y:m.t-8,class:"tick",text:opt.ref.label},s);}
  return s;}

// ---------- line chart over ordered categories ----------
// cats:[label], series:[{name, color, values:[...], lo?:[], hi?:[], strong?, dash?}], opt:{domain, fmt, xTitle?, yTitle?, h?}
function lineChart(el,w,cats,series,opt){const narrow=w<480;const h=opt.h||(narrow?260:300);const m={l:44,r:narrow?92:130,t:14,b:opt.xTitle?44:28};const s=svgIn(el,w,h);
  const x=i=>m.l+(cats.length===1?0:i/(cats.length-1))*(w-m.l-m.r),[d0,d1]=opt.domain,y=lin(d0,d1,h-m.b,m.t);
  for(const v of ticks(d0,d1,5)){S("line",{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:"grid"},s);if(v>d0)S("text",{x:m.l-8,y:y(v)+4,"text-anchor":"end",class:"tick",text:opt.fmt(v)},s);}
  cats.forEach((c,i)=>S("text",{x:x(i),y:h-m.b+16,"text-anchor":"middle",class:"tick",text:c},s));
  if(opt.xTitle)S("text",{x:(m.l+w-m.r)/2,y:h-6,"text-anchor":"middle",class:"tick",text:opt.xTitle},s);
  const ends=[];const obs=[];
  for(const se of series){if(se.lo)S("path",{d:"M"+se.hi.map((v,i)=>`${x(i)},${y(v)}`).join("L")+"L"+se.lo.map((v,i)=>[i,v]).reverse().map(([i,v])=>`${x(i)},${y(v)}`).join("L")+"Z",style:`fill:var(${se.color});opacity:.12`},s);}
  for(const se of series){S("path",{d:"M"+se.values.map((v,i)=>`${x(i)},${y(v)}`).join("L"),style:`fill:none;stroke:var(${se.color});stroke-width:${se.strong?2.5:2};${se.dash?"stroke-dasharray:5 4;":""}`},s);
    se.values.forEach((v,i)=>{const g=S("g",{},s);S("circle",{cx:x(i),cy:y(v),r:se.strong?4.5:3.5,style:`fill:var(${se.color});stroke:var(--surface);stroke-width:2`},g);obs.push([x(i)-5,y(v)-5,x(i)+5,y(v)+5]);
      hit(g,x(i)-10,y(v)-10,20,20,`<b>${se.name}</b> · ${cats[i]}<br>${opt.fmt(v)}${se.lo?` (${opt.fmt(se.lo[i])}–${opt.fmt(se.hi[i])})`:""}`);});
    ends.push({px:x(cats.length-1),py:y(se.values.at(-1)),text:opt.endLabel?opt.endLabel(se):se.name,strong:se.strong});}
  // end labels stacked to avoid collisions, to the right of the last point
  ends.sort((a,b)=>a.py-b.py);let last=-1e9;for(const e of ends){e.y=Math.max(e.py+4,last+16);last=e.y;}
  const maxE=Math.floor((m.r-12)/6.6);for(const e of ends)S("text",{x:e.px+10,y:e.y,class:e.strong?"lbl-strong":"lbl",text:e.text.length>maxE?e.text.slice(0,maxE-1)+"…":e.text},s);
  return s;}

// ---------- forest plot: change against a control, with interval and an optional margin ----------
// rows:[{label, value, lo, hi, color, strong?, note?, group?, tip?}] opt:{domain:[a,b], fmt, margin?:number (drawn at -margin), marginLabel?, level?:"95%"}
function forest(el,w,rows,opt){const narrow=w<480,rowH=opt.rowH||30,need=Math.max(...rows.map(r=>r.label.length))*6.6+18,lw=opt.labelW||Math.min(need,narrow?w*.42:Math.min(230,w*.32));
  const m={l:lw,r:narrow?60:96,t:22,b:24};let gaps=0;rows.forEach((r,i)=>{if(i&&r.group!==rows[i-1].group)gaps++;});
  const h=m.t+m.b+rowH*rows.length+gaps*14+(rows.some(r=>r.group)?14:0);const s=svgIn(el,w,h);const [d0,d1]=opt.domain;const x=lin(d0,d1,m.l,w-m.r);
  if(opt.margin!=null){const mx=x(Math.max(d0,-opt.margin));S("rect",{x:mx,y:m.t,width:x(d1)-mx,height:h-m.t-m.b,style:"fill:var(--good-wash);opacity:.6"},s);
    S("line",{x1:mx,x2:mx,y1:m.t-4,y2:h-m.b,class:"floor"},s);S("text",{x:mx+4,y:narrow&&opt.level?h-m.b-6:m.t-8,class:"tick",text:opt.marginLabel||`margin −${opt.margin}`},s);}
  for(const v of ticks(d0,d1,Math.max(2,Math.floor((w-m.l-m.r)/(Math.max(opt.fmt(d0).length,opt.fmt(d1).length)*6.4+16))))){S("line",{x1:x(v),x2:x(v),y1:m.t,y2:h-m.b,class:"grid"},s);S("text",{x:x(v),y:h-6,"text-anchor":"middle",class:"tick",text:opt.fmt(v)},s);}
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
    hit(g,0,cy-rowH/2,w,rowH,r.tip||`<b>${r.label}</b><br>${opt.fmt(r.value)} [${opt.fmt(r.lo)}, ${opt.fmt(r.hi)}]`);});
  return s;}
