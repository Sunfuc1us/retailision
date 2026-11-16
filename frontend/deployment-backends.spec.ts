import {test,expect} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';

test('real CUDA backend switching, masks and timing in Workbench',async({page})=>{
 test.setTimeout(240000);
 const output='../work_dirs/deployment/ui';await mkdir(output,{recursive:true});
 await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();await page.getByLabel('分割模型').selectOption('coco_pretrained');
 const selector=page.getByLabel('推理 Backend');
 await expect(selector.locator('option')).toHaveCount(3);
 await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');
 await page.getByLabel('目标商品',{exact:true}).fill('FIXTURE_INSTANCE');
 for(const backend of ['pytorch','onnxruntime','tensorrt']){
  await selector.selectOption(backend);
  const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/inference/predict')&&r.request().method()==='POST');
  await page.getByRole('button',{name:/分析图片|重新分析/}).click();
  const response=await responsePromise;expect(response.status()).toBe(200);
  const result=await response.json();expect(result.backend).toBe(backend);expect(result.target_found).toBe(true);
  expect(result.instances.length).toBeGreaterThan(0);expect(result.timing.total_ms).toBeGreaterThan(0);
  const mask=await page.request.get('http://127.0.0.1:8020'+result.instances[0].mask_path);expect(mask.status()).toBe(200);
  expect(mask.headers()['content-type']).toContain('image/png');
  await writeFile(`${output}/${backend}.json`,JSON.stringify(result,null,2));
  await expect(page.getByText('找到目标',{exact:true})).toBeVisible();
  await page.screenshot({path:`${output}/${backend}-desktop.png`,fullPage:true});
 }
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:`${output}/mobile.png`,fullPage:true});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
 await expect(selector).toBeVisible();
});
