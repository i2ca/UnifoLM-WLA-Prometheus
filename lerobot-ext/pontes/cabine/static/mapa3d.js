// MAPA DE OCUPAÇÃO do lidar no 3D (03/11) — usado pela cabine do WLA (:8090) e pela do Prometheus (:8092).
// Lê mapa.json (wla/mapa_ocupacao.py, repassado pela própria cabine): cubos ocupados (log-odds > p 0,7), um pouco da
// nuvem crua do lidar, as juntas medidas, a folga das mãos/cotovelos e a mesa achada. Referencial: a PELVIS
// (x frente, y esquerda, z cima), o mesmo do resto do mundo 3D.
import * as THREE from 'three';

const MAX = 150000;

function corAltura(z, c) {           // do chão (azul) ao alto (vermelho), relativo à pelve
  const t = Math.min(1, Math.max(0, (z + 0.8) / 1.5));
  return c.setHSL(0.66 - 0.66 * t, 0.85, 0.52);
}

export class CamadaMapa {
  constructor(cena, url = 'mapa.json') {
    this.url = url; this.D = null; this.t = 0;
    this.cubos = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({color: 0xffffff, roughness: .8, transparent: true, opacity: .55}), MAX);
    this.cubos.count = 0; this.cubos.frustumCulled = false; cena.add(this.cubos);
    // 03/11: o TAMPO DEDUZIDO (atrás dos objetos nenhum raio chega): outra cor, mais transparente
    this.ded = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({color: 0xc08bff, transparent: true, opacity: .35, depthWrite: false}), 40000);
    this.ded.count = 0; this.ded.frustumCulled = false; cena.add(this.ded);
    this.cruaGeo = new THREE.BufferGeometry();
    this.crua = new THREE.Points(this.cruaGeo, new THREE.PointsMaterial({color: 0xd8dde8, size: 0.008, transparent: true, opacity: .55}));
    this.crua.frustumCulled = false; cena.add(this.crua);
    this.mesa = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({color: 0x37d39b, transparent: true, opacity: .16, side: THREE.DoubleSide, depthWrite: false}));
    this.mesaBorda = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1)), new THREE.LineBasicMaterial({color: 0x37d39b}));
    this.mesa.add(this.mesaBorda); this.mesa.visible = false; cena.add(this.mesa);
    this._m = new THREE.Matrix4(); this._c = new THREE.Color();
    // 03/11: a nuvem da ZED CALIBRADA (com cor), do mapa — rápida (vivo_mapa.json, ~4 Hz)
    this.zedGeo = new THREE.BufferGeometry();
    this.zed = new THREE.Points(this.zedGeo, new THREE.PointsMaterial({size: 0.009, vertexColors: true}));
    this.zed.frustumCulled = false; this.zed.visible = false; cena.add(this.zed);
    this.zedT = 0; this.verZed = true;
    // 03/11: o MODELO DE COLISÃO do corpo (cápsulas do mapa_ocupacao.corpo_colisao): o que o mapa NÃO mapeia
    const matCol = new THREE.MeshBasicMaterial({color: 0xff9a3c, transparent: true, opacity: .16, depthWrite: false});
    this.colCil = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 14, 1, true), matCol, 64);
    this.colEsf = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), matCol, 160);
    for (const m of [this.colCil, this.colEsf]) { m.count = 0; m.frustumCulled = false; cena.add(m); }
    this.verColisao = true;
  }
  desenhaColisao(C) {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), y = new THREE.Vector3(0, 1, 0);
    let nc = 0, ne = 0;
    for (const c of (this.verColisao ? C || [] : [])) {
      const a = new THREE.Vector3(c[0], c[1], c[2]), b = new THREE.Vector3(c[3], c[4], c[5]), r = c[6];
      const d = b.clone().sub(a), L = d.length();
      for (const p of [a, b]) { if (ne < 160) this.colEsf.setMatrixAt(ne++, m.compose(p, q.identity(), new THREE.Vector3(r, r, r))); }
      if (L > 1e-3 && nc < 64) {
        q.setFromUnitVectors(y, d.clone().normalize());
        this.colCil.setMatrixAt(nc++, m.compose(a.clone().add(b).multiplyScalar(.5), q, new THREE.Vector3(r, L, r)));
      }
    }
    this.colCil.count = nc; this.colEsf.count = ne;
    this.colCil.instanceMatrix.needsUpdate = true; this.colEsf.instanceMatrix.needsUpdate = true;
  }
  async buscaVivo(url = 'vivo_mapa.json') {
    let V = null;
    try { V = await (await fetch(url, {cache: 'no-store'})).json(); } catch (_) { return null; }
    const z = V?.zed;
    if (z && z.t !== this.zedT && this.verZed) {
      this.zedT = z.t;
      const mm = new Int16Array(Uint8Array.from(atob(z.mm_b64), c => c.charCodeAt(0)).buffer);
      const rgb = Uint8Array.from(atob(z.rgb_b64), c => c.charCodeAt(0));
      const pos = new Float32Array(mm.length), col = new Float32Array(rgb.length);
      for (let i = 0; i < mm.length; i++) pos[i] = mm[i] / 1000;
      for (let i = 0; i < rgb.length; i++) col[i] = rgb[i] / 255;
      this.zedGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      this.zedGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    }
    this.zed.visible = this.verZed && !!z;
    this.desenhaColisao(V?.colisao);
    return V;
  }
  set visivel(v) { this.cubos.visible = v; this.ded.visible = v; this.crua.visible = v && this.verCrua !== false; this.mesa.visible = v && !!this.D?.mesa; }
  async busca() {
    try { this.D = await (await fetch(this.url, {cache: 'no-store'})).json(); } catch (_) { this.D = null; return null; }
    const D = this.D; if (!D || !D.pronto || D.t === this.t) return D; this.t = D.t;
    const o = D.origem, s = D.vox;
    // 03/11: GROSSO -> FINO (multi_b64: i, j, k, lado): blocos de 8/4/2 cm onde ainda há poucas leituras, 1 cm onde já refinou
    const multi = D.multi_b64 && this.refino !== false;
    const I = multi ? Uint8Array.from(atob(D.multi_b64), ch => ch.charCodeAt(0))
            : D.ocupados_b64 ? Uint8Array.from(atob(D.ocupados_b64), ch => ch.charCodeAt(0)) : (D.ocupados || []);
    const passo = multi ? 4 : 3, n = Math.min(MAX, I.length / passo);
    for (let k = 0; k < n; k++) {
      const f = multi ? I[passo * k + 3] : 1, l = s * f;
      const x = o[0] + (I[passo * k] + f / 2) * s, y = o[1] + (I[passo * k + 1] + f / 2) * s, z = o[2] + (I[passo * k + 2] + f / 2) * s;
      this._m.makeScale(l * .94, l * .94, l * .94).setPosition(x, y, z);
      this.cubos.setMatrixAt(k, this._m); this.cubos.setColorAt(k, corAltura(z, this._c));
    }
    this.cubos.count = n; this.cubos.instanceMatrix.needsUpdate = true;
    const J = D.deduzidos_b64 ? Uint8Array.from(atob(D.deduzidos_b64), ch => ch.charCodeAt(0)) : [];
    const nd = Math.min(40000, J.length / 3);
    for (let k = 0; k < nd; k++) {
      this._m.makeScale(s * .92, s * .92, s * .92).setPosition(o[0] + (J[3 * k] + .5) * s, o[1] + (J[3 * k + 1] + .5) * s, o[2] + (J[3 * k + 2] + .5) * s);
      this.ded.setMatrixAt(k, this._m);
    }
    this.ded.count = nd; this.ded.instanceMatrix.needsUpdate = true;
    if (this.cubos.instanceColor) this.cubos.instanceColor.needsUpdate = true;
    this.cruaGeo.setAttribute('position', new THREE.Float32BufferAttribute(D.nuvem || [], 3));
    const M = D.mesa;
    if (M && M.borda_x_m != null) {
      const x0 = M.borda_x_m, x1 = M.x_max_m, y0 = M.y_min_m, y1 = M.y_max_m;
      this.mesa.scale.set(Math.max(.05, x1 - x0), Math.max(.05, y1 - y0), 1);
      this.mesa.position.set((x0 + x1) / 2, (y0 + y1) / 2, M.altura_pelvis_m + s / 2);
      this.mesa.visible = this.cubos.visible;
    } else this.mesa.visible = false;
    return D;
  }
}

// texto curto para a legenda: pontos/s, cubos, mesa e a menor folga
export function resumoMapa(D) {
  if (!D || !D.pronto) return 'mapa do lidar: sem o serviço (mapa_ocupacao.py)';
  if (!D.lidar_vivo) return 'mapa do lidar: lidar sem pontos (init do robô com USE_LIDAR=true?)';
  const f = Object.entries(D.folga || {}).filter(([, v]) => v != null).sort((a, b) => a[1] - b[1]);
  const nomes = {mao_dir: 'mão dir.', mao_esq: 'mão esq.', punho_dir: 'punho dir.', punho_esq: 'punho esq.', cotovelo_dir: 'cotovelo dir.', cotovelo_esq: 'cotovelo esq.'};
  const m = D.mesa ? `mesa ${(D.mesa.altura_pelvis_m * 100).toFixed(0)} cm da pelve, borda a ${D.mesa.borda_x_m != null ? (D.mesa.borda_x_m * 100).toFixed(0) + ' cm' : '?'}` : 'mesa: procurando';
  const fo = f.length ? ` · mais perto: ${nomes[f[0][0]] || f[0][0]} a ${(f[0][1] * 100).toFixed(0)} cm` : '';
  const zed = (D.motor || '').includes('só o lidar') ? ' (só o lidar, nvblox)' : D.calibrado ? (D.zed_vivo ? ` + ZED ${D.zed_hz} Hz` : ' (ZED parada)') : ' (sem calibração)';
  const imu = D.imu?.lidar_vivo ? ` · IMU do lidar ok (ajuste vivo ${D.imu.ajuste_vivo_graus}°)` : '';
  return `${Math.round(D.pontos_s / 1000)} mil pontos/s${zed} · ${D.n_ocupados} cubos de ${Math.round(D.vox * 100)} cm${D.n_enviados < D.n_ocupados ? ` (${D.n_enviados} na tela)` : ''}${D.n_deduzidos ? ` + ${D.n_deduzidos} de tampo deduzido (lilás)` : ''}${D.niveis ? ` · blocos 8/4/2/1 cm: ${D.niveis[8]}/${D.niveis[4]}/${D.niveis[2]}/${D.niveis[1]}` : ''} · ${m}${fo}${imu}${D.robo_vivo ? '' : ' · sem juntas do robô'}`;
}

// mundo 3D inteiro (para a cabine do Prometheus): o G1 do URDF com as juntas do mapa + a camada do mapa
export async function montaMundo(caixa, {urdf = 'robo/g1_body29_hand14.urdf', info = null} = {}) {
  const { OrbitControls } = await import('./OrbitControls.js');
  const URDFLoader = (await import('./URDFLoader.js')).default;
  const cena = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 30);
  cam.up.set(0, 0, 1); cam.position.set(-1.3, -1.0, 0.8);
  const rend = new THREE.WebGLRenderer({antialias: true}); rend.setPixelRatio(devicePixelRatio); rend.setClearColor(0x0a0d14);
  caixa.appendChild(rend.domElement);
  const ctrl = new OrbitControls(cam, rend.domElement); ctrl.target.set(0.35, 0, -0.2); ctrl.enableDamping = true; ctrl.update();
  cena.add(new THREE.AmbientLight(0xffffff, 0.7)); const sol = new THREE.DirectionalLight(0xffffff, 0.9); sol.position.set(1, -1, 2); cena.add(sol);
  const grade = new THREE.GridHelper(3, 30, 0x34405a, 0x1d2433); grade.rotation.x = Math.PI / 2; grade.position.set(0.3, 0, -0.74); cena.add(grade);
  cena.add(new THREE.AxesHelper(0.15));
  let robo = null;
  new URDFLoader().load(urdf, r => {
    r.traverse(o => { if (o.isMesh) o.material = new THREE.MeshStandardMaterial({color: 0xa9b1c4, metalness: .35, roughness: .55}); });
    robo = r; cena.add(r);
  }, undefined, e => console.warn('URDF', e));
  const camada = new CamadaMapa(cena);
  function tamanho() { const w = caixa.clientWidth || 600, h = Math.round(w * 0.62); rend.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); }
  tamanho(); new ResizeObserver(tamanho).observe(caixa);
  (function anima() { requestAnimationFrame(anima); ctrl.update(); rend.render(cena, cam); })();
  async function ciclo() { const D = await camada.busca(); if (info) info.textContent = resumoMapa(D); }
  async function rapido() {   // o G1 e a ZED a ~4 Hz (os cubos a 1 Hz)
    const V = await camada.buscaVivo();
    if (robo && V?.juntas) for (const [n, q] of Object.entries(V.juntas)) if (robo.joints[n]) robo.joints[n].setJointValue(q);
    setTimeout(rapido, 250);
  }
  ciclo(); setInterval(ciclo, 1000); rapido();
  return {cena, camada};
}
