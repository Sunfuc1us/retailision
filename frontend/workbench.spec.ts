import {test,expect} from '@playwright/test';
test('fixture: upload, target, API request, mask modes and export',async({page})=>{
 await page.route('**/api/inference/models',r=>r.fulfill({json:{default_model:'fixture',loaded_model:null,models:[{id:'fixture',name:'UI Fixture',config:'fixture.json',checkpoint:'fixture.pth',recognition_version:'fixture',recognition_checkpoint:'fixture.pth',fixture:true,backends:['pytorch']}]}}));
 await page.route('**/api/catalog?model_id=*',r=>r.fulfill({json:[{sku_id:'FIXTURE',product:'UI Fixture'}]}));
 await page.route('**/api/results/fixture/mask_001.png',r=>r.fulfill({path:'../tests/fixtures/mask.png',contentType:'image/png'}));
 await page.route('**/api/inference/predict',r=>r.fulfill({json:{success:true,target_found:true,sku_id:'FIXTURE',instances:[{instance_id:1,sku_id:'FIXTURE',identity_score:1,bbox:[20,20,100,100],mask_path:'/api/results/fixture/mask_001.png'}],timing:{total_ms:1}}}));
 await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();await expect(page.getByRole('button',{name:/分析图片|重新分析/})).toBeDisabled();
 await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');await page.getByLabel('目标商品',{exact:true}).fill('FIXTURE');await page.getByRole('button',{name:/分析图片|重新分析/}).click();await expect(page.getByText('找到目标')).toBeVisible();await page.getByRole('button',{name:'Mask',exact:true}).click();await expect(page.getByRole('button',{name:'Mask',exact:true})).toHaveAttribute('aria-pressed','true');await page.waitForFunction(()=>{const c=document.querySelector('canvas');return c?.getContext('2d')?.getImageData(30,30,1,1).data[0]===255});await page.getByRole('button',{name:'Original',exact:true}).click();await expect(page.getByLabel('商品图片与实例 Mask')).toBeVisible();const dl=page.waitForEvent('download');await page.getByRole('button',{name:'导出 JSON'}).click();await dl;
 await page.screenshot({path:'../results/ui/desktop-fixture.png',fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:'../results/ui/mobile-fixture.png',fullPage:true});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
});
test('unavailable model blocks inference',async({page})=>{
 await page.route('**/api/inference/models',r=>r.fulfill({json:{default_model:null,loaded_model:null,models:[]}}));
 await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();
 await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');
 await page.getByLabel('目标商品',{exact:true}).fill('fixture');
 await expect(page.getByRole('button',{name:/分析图片|重新分析/})).toBeDisabled();
 await expect(page.getByLabel('分割模型')).toBeDisabled();
 await expect(page.getByText('尚未配置可用模型',{exact:true})).toBeVisible();
});
