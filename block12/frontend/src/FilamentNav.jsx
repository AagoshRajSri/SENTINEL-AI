import React, { useEffect } from 'react';
import './FilamentNav.css';

export default function FilamentNav() {
  useEffect(() => {
    // The exact script provided by the user
    const root = document.documentElement;
    root.classList.add('js');

    const $ = s => document.querySelector(s);
    const bar = $('.bar'), inner = $('.inner'), brand = $('.brand'), nav = $('.nav'),
          cv = $('#thread'), main = $('main') || document.body, ctx = cv.getContext('2d');
    const links = [...document.querySelectorAll('.rail a')];
    // Map links to sections, fallback to body if not found (during HMR/unmount)
    const secs = links.map(a => document.getElementById(a.getAttribute('href').slice(1)) || document.body);
    const n = links.length;
    const rm = matchMedia('(prefers-reduced-motion: reduce)');

    const STEP = 6;   // px between string nodes
    const SUB = 6;    // ruler subdivisions per nav slot
    const clamp = (v, a = 0, b = 1) => v < a ? a : v > b ? b : v;
    const lerp = (a, b, t) => a + (b - a) * t;
    const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
    const easeOut = t => 1 - Math.pow(1 - t, 3);

    let W = 0, H = 0, dpr = 1, BH = 64, baseY = 63.5, N = 2, dx = 1, sig = 60;
    let ripY = new Float32Array(2), ripV = new Float32Array(2), off = new Float32Array(2);
    let railL = 0, railR = 0, slot = 1, ix = [], tops = [], navTotal = 64, prog = 0, cur = -1;

    const P = { active: false, x: 0, y: 0 };               // pointer, bar-relative
    const S = { x: 0, reach: 0, amp: 0, bead: 0, ready: false }; // smoothed state
    let scrub = null, suppress = false, running = false, idle = 0, last = 0, acc = 0;
    const t0 = performance.now();

    /* ---------- layout ---------- */
    function layout() {
      if (!bar) return;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const br = bar.getBoundingClientRect();
      W = Math.round(br.width); BH = bar.offsetHeight; H = BH + 12; baseY = BH - .5;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      cv.style.width = W + 'px'; cv.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      N = Math.ceil(W / STEP) + 1; dx = W / (N - 1);
      ripY = new Float32Array(N); ripV = new Float32Array(N); off = new Float32Array(N);
      sig = W < 640 ? 44 : 60;

      const ir = inner.getBoundingClientRect();
      const pad = parseFloat(getComputedStyle(inner).paddingRight) || 16;
      railR = ir.right - br.left - pad;
      const brandR = brand.getBoundingClientRect().right - br.left;
      railL = Math.max(brandR + (W < 640 ? 12 : 32), railR - 440);
      slot = (railR - railL) / n;
      ix = links.map((_, k) => railL + slot * (k + .5));
      links.forEach(a => { a.style.width = slot + 'px'; });

      measure();
      if (!S.ready) { S.bead = prog; S.ready = true; }
      wake();
    }

    function measure() {
      const y = window.scrollY;
      tops = secs.map(s => (s ? s.getBoundingClientRect().top + y : 0));
      navTotal = nav ? nav.offsetHeight : 64;
      readProgress();
    }

    function readProgress() {
      if (!tops.length) return;
      const ref = window.scrollY + navTotal + .5;
      let i = 0;
      for (let k = 0; k < n; k++) if (tops[k] <= ref) i = k;
      let f = 0;
      if (i < n - 1 && tops[i+1] > tops[i]) f = clamp((ref - tops[i]) / (tops[i + 1] - tops[i]));
      prog = i + f;
      if (window.innerHeight + window.scrollY >= root.scrollHeight - 2) prog = n - 1;
      const k = Math.round(prog);
      if (k !== cur) {
        cur = k;
        links.forEach((a, j) => j === k ? a.setAttribute('aria-current', 'location') : a.removeAttribute('aria-current'));
      }
    }

    const xOfProg = p => {
      const i = Math.min(n - 1, Math.floor(p)), f = p - i;
      return lerp(ix[i], ix[Math.min(n - 1, i + 1)], f);
    };
    const offAt = x => {
      const f = clamp(x / dx, 0, N - 1.0001), i = f | 0, t = f - i;
      return off[i] * (1 - t) + off[i + 1] * t;
    };
    const nearest = x => {
      let b = 0, d = 1e9;
      for (let k = 0; k < n; k++) { const e = Math.abs(x - ix[k]); if (e < d) { d = e; b = k; } }
      return b;
    };

    /* ---------- string physics (ripples only; hover lean is smoothed separately) ---------- */
    function step() {
      const c = .3, k = .004, d = .02;
      for (let i = 1; i < N - 1; i++) {
        const a = c * (ripY[i - 1] + ripY[i + 1] - 2 * ripY[i]) - k * ripY[i];
        ripV[i] = (ripV[i] + a) * (1 - d);
      }
      for (let i = 1; i < N - 1; i++) ripY[i] += ripV[i];
    }
    function pluck(x, m) {
      for (let i = 1; i < N - 1; i++) { const d = i * dx - x; ripV[i] -= m * Math.exp(-d * d / 288); }
    }

    /* ---------- render loop ---------- */
    function wake() {
      idle = 0;
      if (!running) { running = true; last = performance.now(); requestAnimationFrame(frame); }
    }

    let isUnmounted = false;

    function frame(now) {
      if (isUnmounted) return;
      const dt = Math.min(.05, (now - last) / 1000 || .016); last = now;
      const reduce = rm.matches;
      const kf = reduce ? 1 : 1 - Math.exp(-dt * 12);
      const kr = reduce ? 1 : 1 - Math.exp(-dt * 9);

      // pointer -> targets
      let px = P.x, amp = 0, reach = 0;
      if (P.active) {
        const dy = P.y - baseY, ad = Math.abs(dy);
        reach = 1 - smooth(60, 140, ad);
        const bi = nearest(px), bd = Math.abs(px - ix[bi]);
        if (bd < 26) px = lerp(px, ix[bi], smooth(0, 1, 1 - bd / 26) * .85); // magnetic snap
        amp = reduce ? 0 : (dy < 0 ? -1 : 1) * Math.min(ad * .6, dy < 0 ? 14 : 6) * reach;
      }
      if (S.reach < .02) S.x = px; else S.x += (px - S.x) * kf;
      S.reach += (reach - S.reach) * kr;
      S.amp += (amp - S.amp) * kf;

      // bead follows scroll progress; its motion drags the string
      const prevBead = S.bead;
      S.bead += (prog - S.bead) * kf;
      const bxp = xOfProg(S.bead);
      if (!reduce) {
        const sp = Math.abs(S.bead - prevBead) * slot;
        if (sp > .05) {
          const m = Math.min(sp, 20) * .03;
          const lo = Math.max(1, ((bxp - 40) / dx) | 0), hi = Math.min(N - 2, Math.ceil((bxp + 40) / dx));
          for (let i = lo; i <= hi; i++) { const d = i * dx - bxp; ripV[i] -= m * Math.exp(-d * d / 200); }
        }
        acc += dt; let s = 0;
        while (acc >= 1 / 120 && s < 12) { step(); acc -= 1 / 120; s++; }
        if (s >= 12) acc = 0;
      }

      // string shape = smoothed lean toward pointer + ripples
      const inv = 1 / (2 * sig * sig);
      for (let i = 0; i < N; i++) { const d = i * dx - S.x; off[i] = S.amp * Math.exp(-d * d * inv) + ripY[i]; }

      // draw
      const it = reduce ? 1 : clamp((now - t0) / 1400);
      ctx.clearRect(0, 0, W, H);
      const path = new Path2D();
      path.moveTo(0, baseY + off[0]);
      for (let i = 1; i < N - 1; i++) {
        const x = i * dx, y = baseY + off[i], nx = (i + 1) * dx, ny = baseY + off[i + 1];
        path.quadraticCurveTo(x, y, (x + nx) / 2, (y + ny) / 2);
      }
      path.lineTo(W, baseY + off[N - 1]);

      ctx.lineWidth = 1; ctx.lineJoin = 'round';
      ctx.save();
      if (it < 1) { ctx.beginPath(); ctx.rect(0, 0, W * easeOut(clamp(it / .55)), H); ctx.clip(); }
      ctx.strokeStyle = '#262626'; ctx.stroke(path);
      if (bxp > ix[0]) {
        ctx.save(); ctx.beginPath(); ctx.rect(ix[0], 0, bxp - ix[0], H); ctx.clip();
        ctx.strokeStyle = '#ededed'; ctx.stroke(path); ctx.restore();
      }
      ctx.restore();

      // ruler ticks
      const total = n * SUB;
      for (let j = 0; j <= total; j++) {
        const ti = smooth(0, 1, (it - .3 - (j / total) * .35) / .3);
        if (ti <= 0) continue;
        const x = railL + j * slot / SUB, major = j % SUB === SUB / 2;
        const w = Math.exp(-((x - S.x) ** 2) / (2 * 54 * 54)) * S.reach;
        const pbk = major ? Math.max(0, 1 - Math.abs(S.bead - (j - SUB / 2) / SUB)) : 0;
        const h = (major ? 8 : 3) + (major ? 8 : 9) * w + 3 * pbk;
        const a = major ? .42 + .58 * Math.max(w, pbk) : .2 + .6 * w;
        const xr = Math.round(x) + .5, by = baseY + offAt(x);
        ctx.strokeStyle = 'rgba(237,237,237,' + (a * ti).toFixed(3) + ')';
        ctx.beginPath(); ctx.moveTo(xr, by); ctx.lineTo(xr, by - h * ti); ctx.stroke();
      }

      // bead
      ctx.globalAlpha = smooth(.5, .8, it);
      ctx.fillStyle = '#ededed';
      ctx.beginPath(); ctx.arc(bxp, baseY + offAt(bxp), 3, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;

      // labels ride the string
      for (let k = 0; k < n; k++) {
        const x = ix[k];
        const wh = Math.exp(-((x - S.x) ** 2) / (2 * 40 * 40)) * S.reach;
        const p = Math.max(wh, Math.max(0, 1 - Math.abs(S.bead - k)));
        const a = links[k];
        if (a) {
          a.style.setProperty('--p', p.toFixed(3));
          a.style.transform = 'translate3d(' + x.toFixed(2) + 'px,' + offAt(x).toFixed(2) + 'px,0) translateX(-50%)';
        }
      }

      // settle detection
      let ripE = 0;
      for (let i = 0; i < N; i++) ripE += Math.abs(ripY[i]) + Math.abs(ripV[i]);
      const settled = it >= 1 && ripE < .3 &&
        Math.abs(amp - S.amp) < .01 && Math.abs(px - S.x) < .05 &&
        Math.abs(reach - S.reach) < .005 && Math.abs(prog - S.bead) < .002;
      if (settled) {
        if (++idle > 8) { ripY.fill(0); ripV.fill(0); running = false; return; }
      } else idle = 0;
      requestAnimationFrame(frame);
    }

    /* ---------- input ---------- */
    function setPointer(cx, cy) {
      if (!bar) return;
      const r = bar.getBoundingClientRect();
      P.x = cx - r.left; P.y = cy - r.top;
      P.active = P.y < baseY + 16;
      wake();
    }
    function go(k) {
      if (!rm.matches) pluck(ix[k], 1.3);
      if (secs[k]) secs[k].scrollIntoView({ behavior: rm.matches ? 'auto' : 'smooth', block: 'start' });
      try { history.replaceState(null, '', links[k].getAttribute('href')); } catch (_) {}
      wake();
    }

    const onPointerMove = e => {
      if (e.pointerType === 'touch') {
        if (scrub && e.pointerId === scrub.id) {
          if (!scrub.moved && Math.abs(e.clientX - scrub.x0) > 8) scrub.moved = true;
          setPointer(e.clientX, e.clientY);
          if (scrub.moved) {
            const k = nearest(P.x);
            if (k !== scrub.k) { scrub.k = k; if (navigator.vibrate) navigator.vibrate(6); }
          }
        }
        return;
      }
      setPointer(e.clientX, e.clientY);
    };

    const onMouseLeave = () => { P.active = false; wake(); };
    const onBlur = () => { P.active = false; wake(); };

    const onPointerDown = e => {
      if (e.pointerType !== 'touch' || !e.target.closest('.rail a')) return;
      scrub = { id: e.pointerId, x0: e.clientX, moved: false, k: -1 };
      setPointer(e.clientX, e.clientY);
    };
    const endScrub = (e, cancelled) => {
      if (!scrub || e.pointerId !== scrub.id) return;
      const s = scrub; scrub = null; P.active = false;
      if (s.moved && !cancelled) { suppress = true; setTimeout(() => { suppress = false; }, 400); go(nearest(P.x)); }
      wake();
    };
    const onPointerUp = e => endScrub(e, false);
    const onPointerCancel = e => endScrub(e, true);

    const onScroll = () => { readProgress(); wake(); };
    const onResize = () => { layout(); };

    const resizeObserverBar = new ResizeObserver(layout);
    const resizeObserverMain = new ResizeObserver(() => { measure(); wake(); });

    // Attach events
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    root.addEventListener('mouseleave', onMouseLeave);
    window.addEventListener('blur', onBlur);
    if (bar) bar.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerCancel);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    
    if (bar) resizeObserverBar.observe(bar);
    if (main) resizeObserverMain.observe(main);

    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);

    links.forEach((a, k) => {
      a.onclick = e => { e.preventDefault(); if (!suppress) go(k); };
      a.onfocus = () => {
        if (a.matches(':focus-visible')) { P.active = true; P.x = ix[k]; P.y = baseY - 26; wake(); }
      };
      a.onblur = () => { P.active = false; wake(); };
    });

    if (brand) brand.onclick = e => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: rm.matches ? 'auto' : 'smooth' });
    };

    layout();
    
    // Cleanup
    return () => {
      isUnmounted = true;
      window.removeEventListener('pointermove', onPointerMove);
      root.removeEventListener('mouseleave', onMouseLeave);
      window.removeEventListener('blur', onBlur);
      if (bar) bar.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerCancel);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
      resizeObserverBar.disconnect();
      resizeObserverMain.disconnect();
    };
  }, []);

  return (
    <>
      <a className="skip" href="#triage">Skip to content</a>
      <header className="nav">
        <div className="bar">
          <div className="inner">
            <a className="brand prompt" href="#triage" aria-label="Sentinel-AI, back to top">
              <span className="prompt-arrow">❯</span>
              <span className="prompt-path">~/sentinel-ai</span>
              <span className="prompt-cursor">_</span>
            </a>
          </div>
          <canvas id="thread" aria-hidden="true"></canvas>
          <nav className="rail" aria-label="Primary">
            <ul>
              <li><a href="#triage" style={{ '--i': 0 }}>Triage</a></li>
              <li><a href="#intents" style={{ '--i': 1 }}>Intents</a></li>
              <li><a href="#policy" style={{ '--i': 2 }}>Policy</a></li>
              <li><a href="#metrics" style={{ '--i': 3 }}>Metrics</a></li>
              <li><a href="#docs" style={{ '--i': 4 }}>Docs</a></li>
            </ul>
          </nav>
        </div>
      </header>
    </>
  );
}
