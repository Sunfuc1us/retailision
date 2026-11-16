import {test,expect} from '@playwright/test';
test('target filtering reuses analysis and threshold changes invalidate it',async({page})=>{
 await page.route('**/api/inference/models',r=>r.fulfill({json:{default_model:'fixture',models:[{id:'fixture',name:'Fixture',fixture:true,backends:['pytorch']}]}}));
 await page.route('**/api/catalog?model_id=*',r=>{
  const catalog=[{sku_id:'A',product:'Apple'},{sku_id:'B',product:'Berry'}];
  const query=new URL(r.request().url()).searchParams.get('q');
  return r.fulfill({json:query?catalog.filter(x=>x.sku_id===query):catalog});
 });
 let calls=0;
 await page.route('**/api/inference/predict',r=>{calls++;return r.fulfill({json:{success:true,target_found:true,candidates:[{instance_id:1,sku_id:'A',accepted:true,bbox:[0,0,5,5]},{instance_id:2,sku_id:'B',accepted:true,bbox:[6,6,9,9]}],timing:{total_ms:1}}})});
 await page.goto('http://127.0.0.1:8020');
 await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');
 await page.getByLabel('目标商品',{exact:true}).fill('A');
 await page.getByRole('button',{name:'分析图片',exact:true}).click();await expect(page.getByText('Apple',{exact:true})).toBeVisible();
 await page.getByLabel('目标商品',{exact:true}).fill('B');await expect(page.getByText('Berry',{exact:true})).toBeVisible();expect(calls).toBe(1);
 await page.locator('.settings summary').click();await page.getByLabel('检测阈值').fill('0.95');
 await expect(page.getByText('等待分析',{exact:true})).toBeVisible();expect(calls).toBe(1);
});

test('unmatched target never falls back to all 28 shelf candidates',async({page})=>{
 const catalog=[{sku_id:'COKE',brand:'可口可乐',product:'汽水'},{sku_id:'FANTA',brand:'芬达',product:'汽水'}];
 await page.route('**/api/inference/models',r=>r.fulfill({json:{default_model:'fixture',models:[{id:'fixture',name:'Fixture',fixture:true,backends:['pytorch']}]}}));
 await page.route('**/api/catalog?model_id=*',r=>{
  const q=new URL(r.request().url()).searchParams.get('q');
  return r.fulfill({json:q==='可乐'?[catalog[0]]:q==='芬达'?[catalog[1]]:q==='汽水'?catalog:q==='不存在'?[]:catalog});
 });
 let calls=0;
 const candidates=Array.from({length:28},(_,i)=>({instance_id:i+1,sku_id:i===10?'FANTA':null,accepted:i===10,identity_score:.75,detection_score:.99,bbox:[i*10,0,i*10+8,20]}));
 await page.route('**/api/inference/predict',r=>{calls++;return r.fulfill({json:{success:true,target_found:false,reason:'no_confident_match',sku_id:'COKE',instances:[],candidates,timing:{total_ms:1}}})});
 await page.goto('http://127.0.0.1:8020');
 await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');
 const target=page.getByLabel('目标商品',{exact:true});
 await target.fill('可乐');await page.getByRole('button',{name:'分析图片',exact:true}).click();
 await expect(page.getByRole('status')).toHaveText('未匹配目标 SKU');
 await expect(page.locator('.instance-box')).toHaveCount(0);
 await page.getByRole('button',{name:'Mask',exact:true}).click();
 await expect(page.locator('.instance-box')).toHaveCount(0);
 await page.getByText('全部商品候选诊断 (28)',{exact:true}).click();
 await expect(page.locator('.instance-box')).toHaveCount(28);
 await target.fill('汽水');
 await expect(page.getByRole('status')).toHaveText('匹配到多个商品，请从输入框候选中选择具体商品');
 await expect(page.locator('.instance-box')).toHaveCount(0);
 await target.fill('不存在');
 await expect(page.getByRole('status')).toHaveText('未找到该商品，请检查商品名称或 SKU');
 await expect(page.locator('.instance-box')).toHaveCount(0);
 await target.fill('芬达');
 await expect(page.getByRole('status')).toHaveText('找到目标');
 await expect(page.locator('.instance-box')).toHaveCount(1);
 await expect(page.getByRole('button',{name:'图中实例 11，商品检测分数 0.990',exact:true})).toBeVisible();
 expect(calls).toBe(1);
});
