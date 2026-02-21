"""
JSON Veri Parser
=================

Girdi dosyalarını okur ve sistem nesnelerine çevirir.
"""

import json
from ..models.container import PaletConfig
from ..models.product import UrunData


def parse_json_input(json_data):
    """
    JSON verisini okur ve sistem nesnelerine çevirir.
    
    Args:
        json_data (dict): Ayrıştırılmış JSON verisi
        
    Returns:
        tuple: (PaletConfig, list[UrunData])
    """
    c = json_data.get("container", {})
    palet_cfg = PaletConfig(
        length=c.get("length", 120),
        width=c.get("width", 100),
        height=c.get("height", 180),
        max_weight=c.get("weight", 1250)
    )

    all_products = []
    for detail in json_data.get("details", []):
        p_info = detail.get("product", {})
        qty_to_produce = detail.get("quantity")
        if qty_to_produce is None:
            qty_to_produce = detail.get("package_quantity", 1)
        qty_to_produce = int(qty_to_produce or 0)
        
        # Koli mi Adet mi kontrolü
        pkg_qty = detail.get("package_quantity")
        if pkg_qty is not None and pkg_qty > 0:
            final_boy = p_info.get("package_length", 0)
            final_en = p_info.get("package_width", 0)
            final_yuk = p_info.get("package_height", 0)
            final_agirlik = p_info.get("package_weight", 0)
            is_pkg = True
        else:
            final_boy = p_info.get("unit_length", 0)
            final_en = p_info.get("unit_width", 0)
            final_yuk = p_info.get("unit_height", 0)
            final_agirlik = p_info.get("unit_weight", 0)
            is_pkg = False

        for _ in range(qty_to_produce):
            u = UrunData(
                urun_id=p_info.get("id"),
                code=p_info.get("code"),
                boy=final_boy,
                en=final_en,
                yukseklik=final_yuk,
                agirlik=final_agirlik,
                is_package=is_pkg
            )
            all_products.append(u)

    return palet_cfg, all_products


def load_json_file(filepath):
    """
    JSON dosyasını yükler.
    
    Args:
        filepath (str): Dosya yolu
        
    Returns:
        dict: Ayrıştırılmış JSON verisi
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
