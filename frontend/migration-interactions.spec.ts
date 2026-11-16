import {test,expect} from '@playwright/test';
test('migration: real boxes, opacity, PNG/JSON export and unknown target',async({page})=>{
 await page.goto('http://127.0.0.1:8020');await page.locator('.settings summary').click();await page.getByLabel('分割模型').selectOption('coco_pretrained');await page.getByLabel('上传商品图片').setInputFiles('../third_party/mmdetection/demo/demo.jpg');await page.getByLabel('目标商品',{exact:true}).fill('FIXTURE_INSTANCE');await page.getByRole('button',{name:/分析图片|重新分析/}).click();await expect(page.getByText('找到目标')).toBeVisible({timeout:60000});await expect(page.getByText(/^Box /).first()).toBeVisible();
 await page.getByLabel('Mask 透明度').fill('0.8');await expect(page.getByLabel('Mask 透明度')).toHaveValue('0.8');
 const png=page.waitForEvent('download');await page.getByRole('link',{name:'导出 Mask'}).first().click();expect((await png).suggestedFilename()).toMatch(/\.png$/);
 const json=page.waitForEvent('download');await page.getByRole('button',{name:'导出 JSON'}).click();expect((await json).suggestedFilename()).toBe('retailvision-result.json');
 await page.screenshot({path:'../results/ui/migration-overlay.png',fullPage:true});await page.getByLabel('目标商品',{exact:true}).fill('MIGRATION_UNKNOWN_SKU');await page.getByRole('button',{name:/分析图片|重新分析/}).click();await expect(page.getByText('未找到该商品，请检查商品名称或 SKU',{exact:true})).toBeVisible();
});
