import {chromium} from 'playwright';
const b=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage({viewport:{width:1920,height:1080}});
p.on('console',m=>console.log('C',m.text()));
await p.goto('http://localhost:8123/test/t.html');await p.waitForFunction('window.ready');
console.log(await p.evaluate(()=>{const g=document.createElement('canvas').getContext('webgl2');const d=g.getExtension('WEBGL_debug_renderer_info');return g.getParameter(d.UNMASKED_RENDERER_WEBGL)}));
let t0=Date.now();for(let i=0;i<10;i++){await p.evaluate(t=>window.go(t),i*.1);await p.screenshot({path:`test/f${i}.png`});}
console.log('ms/frame',(Date.now()-t0)/10);await b.close();
