import * as THREE from 'three';
import { C } from './palette.js';
import { Creature, NBLOCKS, XBLOCKS } from './creature.js';
import { tokenTile } from './tokens.js';
import { clamp, hash } from './util.js';

// The unrolled view of one model: the same architecture the creature body shows, laid out flat.
//   x: token positions (one column per token; columns can be placed by tick)
//   y: depth: row 0 = token embedding, row b+1 = residual stream after block b (b = 0..35)
// Each column is one residual stream. Self-attention moves information between columns of the same model
// (causal: a position reads only itself and earlier positions). The link moves information between the two models'
// columns at the exchange rows (after blocks 6, 9, ..., 27). The top of a column is decoded to the next token.
//
// Usage per frame (pure in t):
//   sheet.begin();                        // clears info field, arcs and beams
//   sheet.addInfo(pos, row0, row1, color, amount)   // light a column segment (e.g. the secret rising)
//   sheet.arc(row, fromPos, toPos, weight, color, t) // a self-attention edge at a row (causal: fromPos <= toPos)
//   beams.beam(senderSheet, sPos, readerSheet, rPos, row, weight, t, color)  // cross-model link edge
//   sheet.end(t); beams.end(t);
// Positions in world space: sheet.nodeWorld(pos, row).
export const ROWS = NBLOCKS + 1; // embedding + after each block
export const XROWS = XBLOCKS.map((b) => b + 1); // exchange rows (after blocks 6..27)

export class ModelSheet {
  constructor(side, { tokens = [], dx = 0.62, dy = 0.13, colX = null, outputs = [], tileH = 0.3, arcs = 48 } = {}) {
    this.side = side;
    this.col = side === 'quill' ? C.quill : side === 'ember' ? C.ember : C.grey;
    this.hi = side === 'quill' ? C.quillHi : side === 'ember' ? C.emberHi : new THREE.Color('#c9cbd6');
    this.N = tokens.length; this.dx = dx; this.dy = dy;
    this.colX = colX || ((i) => i * dx);
    this.group = new THREE.Group();
    this.tokens = tokens;
    const N = this.N;
    // info field texture: N columns x ROWS rows, RGB = color*amount
    this.info = new Float32Array(N * ROWS * 4);
    this.tex = new THREE.DataTexture(this.info, ROWS, Math.max(1, N), THREE.RGBAFormat, THREE.FloatType);
    this.tex.magFilter = THREE.LinearFilter; this.tex.minFilter = THREE.LinearFilter; this.tex.needsUpdate = true;
    this.state = { opacity: 1, streamGlow: 1, flow: 1, rows: 1, exchangeGlow: 0.8, nodeGlow: 0.6, reveal: 1 };
    // residual streams
    this.streams = [];
    const H = (ROWS - 1) * dy;
    this.H = H;
    for (let i = 0; i < N; i++) {
      const mat = new THREE.ShaderMaterial({
        uniforms: { uTex: { value: this.tex }, uCol: { value: this.hi.clone() }, uRow: { value: (i + 0.5) / Math.max(1, N) }, uTime: { value: 0 }, uAmp: { value: 1 }, uReveal: { value: 1 }, uFlow: { value: 1 } },
        vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);} `,
        fragmentShader: `uniform sampler2D uTex; uniform vec3 uCol; uniform float uRow,uTime,uAmp,uReveal,uFlow; varying vec2 vUv;
          void main(){ float y = vUv.y; if (y > uReveal) discard;
            vec4 inf = texture2D(uTex, vec2(y, uRow));
            float flow = 0.75 + 0.25*sin(y*80.0 - uTime*5.0*uFlow);
            vec3 c = uCol*0.5*flow + inf.rgb*0.9;
            gl_FragColor = vec4(c*uAmp, 1.0); }`,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, H, 8, 1, true), mat);
      m.geometry.translate(0, H / 2, 0);
      m.position.set(this.colX(i), 0, 0);
      this.group.add(m); this.streams.push(m);
    }
    // nodes: one per (position, row): the hidden state h at that depth
    const ng = new THREE.SphereGeometry(0.045, 10, 8);
    this.nodeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.nodes = new THREE.InstancedMesh(ng, this.nodeMat, Math.max(1, N * ROWS));
    this.nodes.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, N * ROWS) * 3), 3);
    const m4 = new THREE.Matrix4();
    for (let i = 0; i < N; i++) for (let r = 0; r < ROWS; r++) { m4.makeTranslation(this.colX(i), r * dy, 0); this.nodes.setMatrixAt(i * ROWS + r, m4); }
    this.group.add(this.nodes);
    // block rows: faint lines across; exchange rows teal
    const x0 = this.colX(0) - dx * 0.6, x1 = this.colX(Math.max(0, N - 1)) + dx * 0.6;
    this.rowLines = [];
    for (let r = 1; r < ROWS; r++) {
      const isX = XROWS.includes(r);
      const mat = new THREE.MeshBasicMaterial({ color: isX ? C.link.clone() : this.col.clone(), transparent: true, opacity: isX ? 0.5 : 0.12, blending: THREE.AdditiveBlending, depthWrite: false });
      const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, isX ? 0.012 : 0.006, isX ? 0.012 : 0.006), mat);
      m.position.set((x0 + x1) / 2, r * dy, 0); m.userData.isX = isX;
      this.group.add(m); this.rowLines.push(m);
    }
    // input token tiles under the columns, output tiles above
    this.inTiles = tokens.map((tk, i) => { const tl = tokenTile(tk.text ?? tk, tk.style || side, tileH); tl.position.set(this.colX(i), -0.32, 0); this.group.add(tl); return tl; });
    this.outTiles = outputs.map((txt, i) => { if (!txt) return null; const tl = tokenTile(txt, side, tileH); tl.position.set(this.colX(i), H + 0.32, 0); this.group.add(tl); return tl; });
    // arc pool (self-attention)
    this.arcPool = [];
    for (let k = 0; k < arcs; k++) {
      const mat = new THREE.MeshBasicMaterial({ color: this.hi.clone(), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
      const m = new THREE.Mesh(new THREE.BufferGeometry(), mat); m.visible = false; this.group.add(m);
      this.arcPool.push(m);
    }
    const pg = new THREE.BufferGeometry(); const NP = arcs * 2;
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
    pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
    this.arcPackets = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.12, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); this.arcPackets.frustumCulled = false;
    this.group.add(this.arcPackets);
    this._arcN = 0;
  }

  // local -> world position of the hidden state at (pos, row)
  nodeLocal(pos, row, out = new THREE.Vector3()) { return out.set(this.colX(pos), row * this.dy, 0); }
  nodeWorld(pos, row, out = new THREE.Vector3()) { this.group.updateMatrixWorld(true); return this.nodeLocal(pos, row, out).applyMatrix4(this.group.matrixWorld); }
  topWorld(pos, out = new THREE.Vector3()) { return this.nodeWorld(pos, ROWS - 1 + 2.5, out); }
  bottomWorld(pos, out = new THREE.Vector3()) { return this.nodeWorld(pos, -2.5, out); }

  begin() { this.info.fill(0); this._arcN = 0; }

  // light column `pos` from row r0 to r1 (fractional ok) with color*amount; soft edges
  addInfo(pos, r0, r1, color, amount = 1, soft = 1.5) {
    if (pos < 0 || pos >= this.N) return;
    for (let r = 0; r < ROWS; r++) {
      const w = clamp(Math.min(r - r0 + soft, r1 - r + soft) / soft) * amount;
      if (w <= 0) continue;
      const k = (pos * ROWS + r) * 4;
      this.info[k] += color.r * w; this.info[k + 1] += color.g * w; this.info[k + 2] += color.b * w; this.info[k + 3] += w;
    }
  }

  // self-attention edge at `row`: information moves from fromPos to toPos (fromPos <= toPos, causal)
  arc(row, fromPos, toPos, weight, color = null, t = 0) {
    if (this._arcN >= this.arcPool.length || weight <= 0.002) return;
    const m = this.arcPool[this._arcN++];
    const a = this.nodeLocal(fromPos, row), b = this.nodeLocal(toPos, row);
    const span = Math.abs(b.x - a.x);
    const mid = a.clone().lerp(b, 0.5); mid.y += 0.05 + span * 0.12; mid.z += 0.12 + span * 0.25;
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    m.geometry.dispose(); m.geometry = new THREE.TubeGeometry(curve, 24, 0.008 + 0.022 * weight, 5, false);
    m.material.color.copy(color || this.hi); m.material.opacity = clamp(weight * 1.4) * this.state.opacity; m.visible = true;
    const k = this._arcN - 1, pa = this.arcPackets.geometry.attributes.position.array, ca = this.arcPackets.geometry.attributes.color.array;
    for (let j = 0; j < 2; j++) {
      const u = (t * 0.9 + j * 0.5 + hash(k)) % 1; const p = curve.getPoint(u); const idx = (k * 2 + j) * 3;
      pa[idx] = p.x; pa[idx + 1] = p.y; pa[idx + 2] = p.z;
      const c = (color || this.hi).clone().multiplyScalar(weight * Math.sin(Math.PI * u) * 1.5 * this.state.opacity);
      ca[idx] = c.r; ca[idx + 1] = c.g; ca[idx + 2] = c.b;
    }
  }

  end(t) {
    const s = this.state;
    this.tex.needsUpdate = true;
    for (const m of this.streams) { const u = m.material.uniforms; u.uTime.value = t; u.uAmp.value = s.streamGlow * s.opacity; u.uReveal.value = s.reveal; u.uFlow.value = s.flow; }
    // nodes: base dim color + info
    const c = new THREE.Color();
    for (let i = 0; i < this.N; i++) for (let r = 0; r < ROWS; r++) {
      const k = (i * ROWS + r) * 4; const vis = r / (ROWS - 1) <= s.reveal + 1e-6;
      const isX = XROWS.includes(r);
      c.copy(isX ? C.link : this.col).multiplyScalar(vis ? s.nodeGlow * (isX ? s.exchangeGlow : 0.55) * s.opacity : 0);
      c.r += this.info[k] * 0.8 * s.opacity; c.g += this.info[k + 1] * 0.8 * s.opacity; c.b += this.info[k + 2] * 0.8 * s.opacity;
      this.nodes.setColorAt(i * ROWS + r, c);
    }
    this.nodes.instanceColor.needsUpdate = true;
    for (const l of this.rowLines) { l.material.opacity = (l.userData.isX ? 0.5 * s.exchangeGlow : 0.1) * s.opacity * s.rows; }
    for (const tl of this.inTiles) tl.material.opacity = s.opacity;
    for (const tl of this.outTiles) if (tl) tl.material.opacity = s.opacity;
    for (let k = this._arcN; k < this.arcPool.length; k++) this.arcPool[k].visible = false;
    const pa = this.arcPackets.geometry.attributes.position.array;
    for (let k = this._arcN * 2; k < pa.length / 3; k++) pa[k * 3 + 2] = 1e5;
    this.arcPackets.geometry.attributes.position.needsUpdate = true; this.arcPackets.geometry.attributes.color.needsUpdate = true;
  }
}

// Cross-model link edges between two sheets (in world space). Information moves sender -> reader.
export class LinkBeams {
  constructor(scene, { pool = 64 } = {}) {
    this.group = new THREE.Group(); scene.add(this.group);
    this.pool = [];
    for (let k = 0; k < pool; k++) {
      const mat = new THREE.MeshBasicMaterial({ color: C.link.clone(), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
      const m = new THREE.Mesh(new THREE.BufferGeometry(), mat); m.visible = false; this.group.add(m); this.pool.push(m);
    }
    const NP = pool * 3; const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
    pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
    this.packets = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.16, map: Creature.dotTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); this.packets.frustumCulled = false;
    this.group.add(this.packets);
    this.n = 0; this.opacity = 1;
  }
  begin() { this.n = 0; }
  // weight 0..1 = attention weight; color = packet tint (e.g. gold for the secret); sag bends the beam
  beam(sender, sPos, reader, rPos, row, weight, t = 0, color = null, sag = 0, bulge = 0) {
    if (this.n >= this.pool.length || weight <= 0.002) return;
    const m = this.pool[this.n++];
    const a = sender.nodeWorld(sPos, row), b = reader.nodeWorld(rPos, row);
    const mid = a.clone().lerp(b, 0.5); mid.y += sag; mid.z += bulge;
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    m.geometry.dispose(); m.geometry = new THREE.TubeGeometry(curve, 24, 0.008 + 0.028 * weight, 5, false);
    m.material.color.copy(C.link); m.material.opacity = clamp(0.05 + weight * 0.75) * this.opacity; m.visible = true;
    const k = this.n - 1, pa = this.packets.geometry.attributes.position.array, ca = this.packets.geometry.attributes.color.array;
    const send = sender.side === 'quill' ? C.quillHi : sender.side === 'ember' ? C.emberHi : new THREE.Color('#c9cbd6');
    for (let j = 0; j < 3; j++) {
      const u = (t * 0.8 + j / 3 + hash(k * 1.7)) % 1; const p = curve.getPoint(u); const idx = (k * 3 + j) * 3;
      pa[idx] = p.x; pa[idx + 1] = p.y; pa[idx + 2] = p.z;
      const c = (color ? color.clone() : C.link.clone().lerp(send, 0.5 * (1 - u))).multiplyScalar(weight * Math.sin(Math.PI * u) * 1.8 * this.opacity);
      ca[idx] = c.r; ca[idx + 1] = c.g; ca[idx + 2] = c.b;
    }
  }
  end() {
    for (let k = this.n; k < this.pool.length; k++) this.pool[k].visible = false;
    const pa = this.packets.geometry.attributes.position.array;
    for (let k = this.n * 3; k < pa.length / 3; k++) pa[k * 3 + 2] = 1e5;
    this.packets.geometry.attributes.position.needsUpdate = true; this.packets.geometry.attributes.color.needsUpdate = true;
  }
}
