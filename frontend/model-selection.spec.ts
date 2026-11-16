import {test,expect} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';

test('start-web default trained weights and real GPU model/backend switching',async({page})=>{
  test.setTimeout(240000);
  const output='../work_dirs/web_model_selection';await mkdir(output,{recursive:true});
  await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();
  const weights=page.getByLabel('分割模型'),backend=page.getByLabel('推理 Backend');
  await expect(weights).toHaveValue('retail_6sku_v002');
  await expect(page.getByLabel('检测阈值')).toHaveValue('0.8');
  await expect(page.getByLabel('检测阈值')).toHaveAttribute('step','0.01');
  await weights.selectOption('retailtest_smoke_v001');
  await expect(page.getByText('iter_120.pth',{exact:true})).toBeVisible();
  await expect(backend.locator('option')).toHaveCount(1);
  await expect(backend).toHaveValue('pytorch');

  async function infer(modelId:string,sku:string,filename:string){
    await page.getByLabel('目标商品',{exact:true}).fill(sku);
    const pending=page.waitForResponse(r=>r.url().endsWith('/api/inference/predict')&&r.request().method()==='POST');
    await page.getByRole('button',{name:/分析图片|重新分析/}).click();
    const response=await pending;expect(response.status()).toBe(200);
    const body=await response.json();expect(body.model_id).toBe(modelId);expect(body.target_found).toBe(true);
    expect(body.backend).toBe(await backend.inputValue());
    const mask=await page.request.get('http://127.0.0.1:8020'+body.instances[0].mask_path);
    expect(mask.status()).toBe(200);expect(mask.headers()['content-type']).toContain('image/png');
    await writeFile(`${output}/${filename}.json`,JSON.stringify(body,null,2));
    await expect(page.getByText('找到目标',{exact:true})).toBeVisible();
    return body;
  }

  await page.getByLabel('上传商品图片').setInputFiles('../datasets/retailtest_smoke_v001/segmentation/images/20261110-141818.474-1.jpg');
  const trained=await infer('retailtest_smoke_v001','VITA_APPLE_JUICE_250ML','trained-first');
  expect(trained.candidates).toHaveLength(3);
  await page.screenshot({path:`${output}/trained-desktop.png`,fullPage:true});

  await weights.selectOption('coco_pretrained');
  await expect(page.getByLabel('目标商品',{exact:true})).toHaveValue('');
  await expect(page.getByText('等待分析',{exact:true})).toBeVisible();
  await expect(backend.locator('option')).toHaveCount(3);
  await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');
  await infer('coco_pretrained','FIXTURE_INSTANCE','coco-pytorch');
  await backend.selectOption('onnxruntime');
  await infer('coco_pretrained','FIXTURE_INSTANCE','coco-onnxruntime');
  await backend.selectOption('tensorrt');
  await infer('coco_pretrained','FIXTURE_INSTANCE','coco-tensorrt');
  await page.screenshot({path:`${output}/coco-desktop.png`,fullPage:true});

  await weights.selectOption('retailtest_smoke_v001');
  await expect(backend).toHaveValue('pytorch');
  await expect(backend.locator('option')).toHaveCount(1);
  await expect(page.getByLabel('目标商品',{exact:true})).toHaveValue('');
  await page.getByLabel('上传商品图片').setInputFiles('../datasets/retailtest_smoke_v001/segmentation/images/20261110-141818.474-2.jpg');
  const switched=await infer('retailtest_smoke_v001','COKE_ZERO_330ML','trained-switched-back');
  expect(switched.candidates).toHaveLength(3);
  const status=await (await page.request.get('http://127.0.0.1:8020/api/inference/models')).json();
  expect(status.loaded_model).toBe('retailtest_smoke_v001');
  expect(status.models.find((x:{id:string})=>x.id==='retailtest_smoke_v001').checkpoint).toMatch(/iter_120\.pth$/);
  await page.getByRole('button',{name:'Mask',exact:true}).click();
  await page.screenshot({path:`${output}/trained-mask.png`,fullPage:true});
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'导出 JSON'}).click();
  expect((await download).suggestedFilename()).toBe('retailvision-result.json');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:`${output}/trained-mobile.png`,fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await expect(weights).toBeEnabled();
});
