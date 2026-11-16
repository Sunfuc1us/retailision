import json
import pytest
from retailvision.data.catalog import Catalog


@pytest.fixture
def catalog(tmp_path):
    path=tmp_path/'catalog.json'
    path.write_text(json.dumps([
        dict(sku_id='VITA_APPLE_JUICE_250ML',brand='维他',product='苹果汁饮料',flavor='苹果味',specification='250ml'),
        dict(sku_id='VITA_BLACKCURRANT_DRINK_250ML',brand='维他',product='风味饮料',flavor='黑加仑味',specification='250ml'),
        dict(sku_id='SPRITE_ORIGINAL_330ML',brand='雪碧',product='汽水',flavor='柠檬味',specification='330ml'),
    ]))
    return Catalog(path)


@pytest.mark.parametrize('query',[
    '维他苹果汁','苹果汁','维他 苹果汁','苹果汁 维他 250ml',
    '维他 苹果汁 ２５０ＭＬ','维他苹果汁饮料 苹果味 250ml',
    ' VITA_APPLE_JUICE_250ML ','vita_apple_juice_250ml','VITA_APPLE',
])
def test_product_name_keywords_and_sku_resolve_to_same_product(catalog,query):
    assert [x['sku_id'] for x in catalog.resolve(query)]==['VITA_APPLE_JUICE_250ML']


def test_brand_and_flavor_do_not_require_intervening_product_field(catalog):
    assert [x['sku_id'] for x in catalog.resolve('维他黑加仑')]==['VITA_BLACKCURRANT_DRINK_250ML']


def test_all_keywords_must_belong_to_one_sku_and_ambiguity_is_preserved(catalog):
    assert len(catalog.resolve('维他'))==2
    assert len(catalog.resolve('250ml'))==2
    for query in ('维他 330ml','雪碧苹果汁','不存在的商品','  '):
        assert catalog.resolve(query)==[]
