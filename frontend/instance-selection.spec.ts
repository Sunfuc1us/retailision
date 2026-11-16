import {test,expect} from '@playwright/test';

const instances=[
  {instance_id:4,sku_id:'FIXTURE',identity_score:.940,detection_score:.989,bbox:[25,50,150,190],mask_path:'/api/results/fixture/mask_001.png'},
  {instance_id:9,sku_id:'FIXTURE',identity_score:.592,detection_score:.974,bbox:[200,90,320,240],mask_path:'/api/results/fixture/mask_001.png'},
  {instance_id:20,sku_id:'FIXTURE',identity_score:.661,detection_score:.849,bbox:[350,230,485,390],mask_path:'/api/results/fixture/mask_001.png'},
];

test.beforeEach(async({page})=>{
  await page.route('**/api/inference/models',r=>r.fulfill({json:{default_model:'fixture',loaded_model:null,models:[{id:'fixture',name:'UI Fixture',config:'fixture.json',checkpoint:'fixture.pth',recognition_version:'fixture',recognition_checkpoint:'fixture.pth',fixture:true,backends:['pytorch']}]}}));
  await page.route('**/api/catalog?model_id=*',r=>r.fulfill({json:new URL(r.request().url()).searchParams.get('q')==='CHANGED'?[]:[{sku_id:'FIXTURE',product:'UI Fixture'}]}));
  await page.route('**/api/results/fixture/mask_001.png',r=>r.fulfill({path:'../tests/fixtures/mask.png',contentType:'image/png'}));
});

async function upload(page:import('@playwright/test').Page){
  await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();
  await page.getByLabel('上传商品图片').setInputFiles({name:'instances.svg',mimeType:'image/svg+xml',buffer:Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="500" height="400"><rect width="500" height="400" fill="#eee"/></svg>')});
  await page.getByLabel('目标商品',{exact:true}).fill('FIXTURE');
  await page.getByRole('button',{name:/分析图片|重新分析/}).click();
}

test('non-sequential instance IDs, scores, both selection directions and reset',async({page})=>{
  await page.route('**/api/inference/predict',r=>r.fulfill({json:{success:true,target_found:true,sku_id:'FIXTURE',instances,candidates:instances,timing:{total_ms:1}}}));
  await upload(page);
  const box4=page.getByRole('button',{name:'图中实例 4，商品检测分数 0.989',exact:true});
  const row4=page.getByRole('button',{name:'选择实例 4',exact:true});
  await expect(box4).toBeVisible();await expect(box4).toHaveText('#4 · 检测 0.989');
  await box4.click();await expect(row4).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#instance-4')).toContainText('商品检测分数 0.989');
  await expect(page.locator('#instance-4')).toContainText('SKU 相似度 0.940');
  await page.getByRole('button',{name:'选择实例 20',exact:true}).click();
  const box20=page.getByRole('button',{name:'图中实例 20，商品检测分数 0.849',exact:true});
  await expect(box20).toHaveAttribute('aria-pressed','true');await expect(box4).toHaveAttribute('aria-pressed','false');
  // Geometry stays aligned with the displayed image after resizing.
  await page.setViewportSize({width:390,height:844});
  const image=await page.locator('canvas').boundingBox();const box=await box20.boundingBox();
  expect(image).not.toBeNull();expect(box).not.toBeNull();
  expect(Math.abs(box!.x-(image!.x+350/500*image!.width))).toBeLessThan(2);
  expect(Math.abs(box!.y-(image!.y+230/400*image!.height))).toBeLessThan(2);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await page.getByRole('button',{name:'Mask',exact:true}).click();
  await expect(box20).toBeVisible();
  await page.getByRole('button',{name:'Original',exact:true}).click();await expect(box20).toHaveCount(0);
  await page.getByRole('button',{name:'选择实例 4',exact:true}).click();await expect(box4).toHaveAttribute('aria-pressed','true');
  await box4.focus();await page.keyboard.press('Enter');await expect(row4).toHaveAttribute('aria-pressed','false');
  await page.getByLabel('目标商品',{exact:true}).fill('CHANGED');
  await expect(page.getByText('未找到该商品，请检查商品名称或 SKU',{exact:true})).toBeVisible();
  await expect(page.locator('.instance-box')).toHaveCount(0);
});

test('unmatched candidates and unavailable scores retain instance selection',async({page})=>{
  await page.route('**/api/inference/predict',r=>r.fulfill({json:{success:true,target_found:false,reason:'no_confident_match',instances:[],candidates:[{...instances[1],identity_score:null,detection_score:undefined,top1_sku:'OTHER',accepted:false,reason:'below_similarity'}],timing:{total_ms:1}}}));
  await upload(page);
  const box=page.getByRole('button',{name:'图中实例 9，商品检测分数 —',exact:true});
  await expect(page.getByRole('status')).toHaveText('未匹配目标 SKU');
  await expect(box).toHaveCount(0);
  await page.getByText('全部商品候选诊断 (1)',{exact:true}).click();
  await box.click();await expect(page.getByRole('button',{name:'选择实例 9',exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#instance-9')).toContainText('SKU 相似度 —');
  await expect(page.locator('#instance-9')).toContainText('未确认 SKU · SKU 相似度不足');
  await expect(page.getByRole('link',{name:'导出候选 Mask',exact:true})).toBeVisible();
  await page.getByText('全部商品候选诊断 (1)',{exact:true}).click();
  await expect(box).toHaveCount(0);
  await page.getByText('全部商品候选诊断 (1)',{exact:true}).click();
  await expect(box).toHaveAttribute('aria-pressed','false');
  await page.getByLabel('目标商品',{exact:true}).fill('CHANGED');
  await expect(page.getByRole('status')).toHaveText('未找到该商品，请检查商品名称或 SKU');
  await expect(page.locator('.diag')).not.toHaveAttribute('open');
  await expect(box).toHaveCount(0);
});

test('wheel zoom, drag and reset preserve instance alignment',async({page})=>{
  await page.route('**/api/inference/predict',r=>r.fulfill({json:{success:true,target_found:true,sku_id:'FIXTURE',instances,candidates:instances,timing:{total_ms:1}}}));
  await upload(page);
  await expect(page.getByLabel('Mask 透明度')).toHaveValue('0.8');
  const viewport=page.getByLabel('图片视图，滚轮缩放，拖动平移，Home 复位');
  const canvas=page.locator('canvas');
  const box=page.getByRole('button',{name:'图中实例 4，商品检测分数 0.989',exact:true});
  await expect(box).toBeVisible();
  const original=(await canvas.boundingBox())!;
  const bounds=(await viewport.boundingBox())!;
  await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.mouse.wheel(0,-250);
  await expect.poll(async()=>(await canvas.boundingBox())!.width).toBeGreaterThan(original.width*1.5);
  const enlarged=(await canvas.boundingBox())!;
  const zoomBox=(await box.boundingBox())!;
  expect(Math.abs(zoomBox.x-(enlarged.x+25/500*enlarged.width))).toBeLessThan(2);
  expect(Math.abs(zoomBox.y-(enlarged.y+50/400*enlarged.height))).toBeLessThan(2);
  await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.mouse.down();await page.mouse.move(bounds.x+bounds.width/2+70,bounds.y+bounds.height/2+40,{steps:5});await page.mouse.up();
  const dragged=(await canvas.boundingBox())!;
  expect(Math.abs(dragged.x-enlarged.x-70)).toBeLessThan(2);
  expect(Math.abs(dragged.y-enlarged.y-40)).toBeLessThan(2);
  await expect(page.locator('.instance-box[aria-pressed=true]')).toHaveCount(0);
  await page.screenshot({path:'../work_dirs/architecture_optimization/zoom-pan.png',fullPage:true});
  await page.getByRole('button',{name:'复位图片视图'}).click();
  await expect.poll(async()=>Math.abs((await canvas.boundingBox())!.width-original.width)).toBeLessThan(2);
  expect(Math.abs((await canvas.boundingBox())!.x-original.x)).toBeLessThan(2);
  await page.setViewportSize({width:390,height:844});
  await expect(viewport).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await page.screenshot({path:'../work_dirs/architecture_optimization/zoom-pan-mobile.png',fullPage:true});
});
