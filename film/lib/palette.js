import * as THREE from 'three';
// One palette for the whole film. Quill = violet, Ember = orange, the link = teal-green (matches the repo's figures).
export const C = {
  bg0: new THREE.Color('#05040a'), bg1: new THREE.Color('#0d0a1c'),
  quill: new THREE.Color('#8b5cf6'), quillHi: new THREE.Color('#c4b5fd'), quillDeep: new THREE.Color('#4c1d95'),
  ember: new THREE.Color('#f97316'), emberHi: new THREE.Color('#fdba74'), emberDeep: new THREE.Color('#9a3412'),
  link: new THREE.Color('#14e3a8'), linkHi: new THREE.Color('#a7f3d0'), linkDeep: new THREE.Color('#065f46'),
  ice: new THREE.Color('#9fd8ff'), white: new THREE.Color('#ffffff'), grey: new THREE.Color('#6b7280'),
  bad: new THREE.Color('#ef4444'),
};
export const CSS = {
  quill: '#a78bfa', quillHi: '#ddd6fe', ember: '#fb923c', emberHi: '#fed7aa', link: '#34f5c0', ice: '#9fd8ff',
  text: '#f5f3ff', dim: 'rgba(245,243,255,0.62)', bad: '#f87171',
};
