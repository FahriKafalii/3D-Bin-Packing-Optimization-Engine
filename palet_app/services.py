"""
Django Servis Katmanı (Bridge Layer)
======================================

src/ paketindeki saf algoritmalar ile Django ORM arasında köprü kurar.
Tüm Django-bağımlı iş mantığı bu modüldedir.
"""

import json
import io
from django.core.files.base import ContentFile

from src.models import PaletConfig, UrunData
from src.core.single_pallet import (
    simulate_single_pallet,
    generate_grid_placement,
    DEFAULT_SINGLE_THRESHOLD,
)
from src.core.packing import pack_maximal_rectangles, merge_and_repack, compact_pallet
from src.core.genetic_algorithm import run_ga
from src.core.mix_pallet import mix_palet_yerlestirme_main
from src.utils.helpers import urun_hacmi, group_products_smart
from src.utils.visualization import renk_uret, render_pallet_3d

from .models import Palet, Urun, Optimization


# ====================================================================
# DÖNÜŞTÜRÜCÜLER (Django Model ↔ Algoritma Nesnesi)
# ====================================================================

def django_urun_to_urundata(urun):
    """Django Urun modelini UrunData nesnesine çevirir."""
    urun_data = UrunData(
        urun_id=urun.id,
        code=urun.urun_kodu,
        boy=urun.boy,
        en=urun.en,
        yukseklik=urun.yukseklik,
        agirlik=urun.agirlik,
        quantity=1,
        is_package=False
    )
    urun_data.donus_serbest = urun.donus_serbest
    urun_data.mukavemet = urun.mukavemet
    return urun_data


def container_info_to_config(container_info):
    """Container bilgisini PaletConfig nesnesine çevirir."""
    return PaletConfig(
        length=container_info['length'],
        width=container_info['width'],
        height=container_info['height'],
        max_weight=container_info['weight']
    )


# ====================================================================
# SINGLE PALET SERVİSİ
# ====================================================================

def single_palet_yerlestirme(urunler, container_info, optimization=None):
    """
    Single Palet Sürecini Yöneten Ana Fonksiyon.
    Django modelleriyle çalışır, src/ algoritmasını kullanır.
    
    Args:
        urunler: Django Urun queryset/list
        container_info: Container bilgileri dict
        optimization: Django Optimization nesnesi
        
    Returns:
        tuple: (single_pallets, yerlesmemis_urunler)
    """
    print("--- Single Palet Operasyonu Başlıyor ---")
    
    palet_cfg = container_info_to_config(container_info)
    
    # Django → UrunData dönüşümü
    all_products = [django_urun_to_urundata(urun) for urun in urunler]
    
    # Grupla
    groups = group_products_smart(all_products)
    
    single_pallets = []
    mix_pool = []
    total_palet_id = 1
    
    for key, group_items in groups.items():
        urun_kodu = key[0]
        total_qty = len(group_items)
        
        print(f"Grup İnceleniyor: {urun_kodu}, Adet: {total_qty}")
        
        # Simülasyon (src/ algoritması)
        sim_result = simulate_single_pallet(group_items, palet_cfg)
        
        if sim_result["can_be_single"]:
            capacity = sim_result["capacity"]
            efficiency = sim_result["efficiency"]
            item_volume = group_items[0].boy * group_items[0].en * group_items[0].yukseklik
            pallet_volume = palet_cfg.volume
            
            if total_qty >= capacity:
                num_full_pallets = total_qty // capacity
                remainder = total_qty % capacity
                remainder_fill_ratio = (remainder * item_volume) / pallet_volume if remainder > 0 else 0
                create_partial = (remainder_fill_ratio >= DEFAULT_SINGLE_THRESHOLD)
                
                print(f"  -> ✅ ONAYLANDI. {sim_result['reason']}")
                print(f"  -> Efficiency: {efficiency*100:.1f}% | Capacity: {capacity} items/pallet")
                print(f"  -> Stock: {total_qty} → {num_full_pallets} full pallet(s)")
                
                if remainder > 0:
                    if create_partial:
                        print(f"  -> + 1 partial pallet ({remainder} items, fill: {remainder_fill_ratio*100:.1f}%)")
                    else:
                        print(f"  -> ⚠️  {remainder} remainder items → Mix Pool")
            else:
                num_full_pallets = 0
                remainder = total_qty
                partial_fill_ratio = (remainder * item_volume) / pallet_volume
                create_partial = (partial_fill_ratio >= DEFAULT_SINGLE_THRESHOLD)
                
                if create_partial:
                    print(f"  -> ✅ ONAYLANDI (Partial). Fill: {partial_fill_ratio*100:.1f}%")
                else:
                    print(f"  -> ⚠️  REJECTED. Fill: {partial_fill_ratio*100:.1f}% < {DEFAULT_SINGLE_THRESHOLD:.0%}")
                    mix_pool.extend(group_items)
                    continue
            
            # Full paletler oluştur
            for palet_no in range(num_full_pallets):
                palet_items = group_items[palet_no * capacity:(palet_no + 1) * capacity]
                placements = generate_grid_placement(palet_items, palet_cfg)
                
                palet = _create_django_palet(
                    placements, palet_cfg, optimization, total_palet_id, 'single'
                )
                single_pallets.append(palet)
                total_palet_id += 1
            
            # Partial palet oluştur
            if create_partial and remainder > 0:
                palet_items = group_items[-remainder:]
                placements = generate_grid_placement(palet_items, palet_cfg)
                
                palet = _create_django_palet(
                    placements, palet_cfg, optimization, total_palet_id, 'single'
                )
                single_pallets.append(palet)
                total_palet_id += 1
            elif remainder > 0:
                leftovers = group_items[-remainder:]
                mix_pool.extend(leftovers)
                print(f"  -> ⚠️  {remainder} items sent to Mix Pool")
        else:
            print(f"  -> ❌ REDDEDİLDİ. {sim_result['reason']}")
            mix_pool.extend(group_items)
    
    print(f"--- Single Bitti. {len(single_pallets)} palet. Mix Havuzu: {len(mix_pool)} ürün. ---")
    
    # Mix pool UrunData → Django Urun geri dönüşümü
    yerlesmemis_urunler = []
    for item in mix_pool:
        urun_obj = next((u for u in urunler if u.id == item.id), None)
        if urun_obj:
            yerlesmemis_urunler.append(urun_obj)
    
    return single_pallets, yerlesmemis_urunler


# ====================================================================
# MIX PALET SERVİSİ
# ====================================================================

def chromosome_to_palets(chromosome, palet_cfg, optimization, baslangic_id):
    """
    En iyi kromozomdan Django Palet nesneleri oluşturur.

    Adımlar:
        1. Kromozom decode → ürün sırası
        2. pack_maximal_rectangles → ham palet dict listesi
        3. compact_pallet          → her paleti gravite + orijin yönünde sıkıştır
        4. merge_and_repack        → düşük doluluklu paletleri birleştir (post-opt)
        5. Django Palet nesnelerine dönüştür ve kaydet
    """
    siralanmis_urunler = [chromosome.urunler[i] for i in chromosome.sira_gen]
    pallets = pack_maximal_rectangles(siralanmis_urunler, palet_cfg)

    # ── Kompaksiyon (gravity + origin) ────────────────────────────────
    for p in pallets:
        compact_pallet(p, palet_cfg)

    # ── Merge & Repack post-optimizasyonu ─────────────────────────────
    onceki_sayi = len(pallets)
    pallets = merge_and_repack(pallets, palet_cfg)
    sonraki_sayi = len(pallets)
    if sonraki_sayi < onceki_sayi:
        print(
            f"[MERGE & REPACK] {onceki_sayi} → {sonraki_sayi} palet "
            f"({onceki_sayi - sonraki_sayi} palet azaltıldı)"
        )

    django_paletler = []
    palet_id = baslangic_id

    for pallet_data in pallets:
        palet = _create_django_palet(
            pallet_data['items'], palet_cfg, optimization, palet_id, 'mix',
            items_are_dicts=True
        )
        django_paletler.append(palet)
        palet_id += 1

    return django_paletler


# ====================================================================
# MERGE & REPACK SERVİSİ
# ====================================================================

def merge_repack_service(
    mix_paletler: list,
    palet_cfg,
    optimization,
    baslangic_id: int,
    urun_data_listesi: list,
) -> tuple:
    """
    Django Palet listesini packer dict formatına çevirir, merge_and_repack_v2
    çalıştırır ve sonuç iyileşmişse eski paletleri DB'den silip yenileri kaydeder.

    Args:
        mix_paletler:      Mevcut Django Palet nesneleri (mix türü).
        palet_cfg:         PaletConfig nesnesi.
        optimization:      Django Optimization nesnesi.
        baslangic_id:      Yeni paleter için başlangıç ID'si.
        urun_data_listesi: UrunData nesneleri — item['urun'] referansları için gerekli.

    Returns:
        (final_django_paletler, metrics) — kabul edilmezse orijinal paletler döner.
    """
    from src.core.merge_repack import merge_and_repack_v2, MergeRepackMetrics
    import logging
    logger = logging.getLogger(__name__)

    if len(mix_paletler) < 2:
        return mix_paletler, MergeRepackMetrics.no_op("insufficient_pallets")

    # ── 1) UrunData hızlı erişim sözlüğü ─────────────────────────────
    urun_data_by_id = {ud.id: ud for ud in urun_data_listesi}

    # ── 2) Django Palet → packer dict ────────────────────────────────
    packer_pallets = []
    for palet in mix_paletler:
        konumlar = palet.json_to_dict(palet.urun_konumlari)
        boyutlar = palet.json_to_dict(palet.urun_boyutlari)
        items = []
        total_weight = 0.0
        for urun_id_str, pos in konumlar.items():
            urun_id = int(urun_id_str)
            ud = urun_data_by_id.get(urun_id)
            if ud is None:
                logger.warning("[MR] UrunData id=%s bulunamadı, atlanıyor", urun_id_str)
                continue
            dim = boyutlar.get(urun_id_str, [ud.boy, ud.en, ud.yukseklik])
            items.append({
                'urun': ud,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
                'L': float(dim[0]),
                'W': float(dim[1]),
                'H': float(dim[2]),
            })
            total_weight += ud.agirlik
        packer_pallets.append({'items': items, 'weight': total_weight})

    # ── 3) Merge & Repack v2 ──────────────────────────────────────────
    optimized, metrics = merge_and_repack_v2(packer_pallets, palet_cfg)

    if not metrics.accepted:
        return mix_paletler, metrics

    # ── 4) Eski Django paletleri sil ─────────────────────────────────
    for palet in mix_paletler:
        palet.delete()

    # ── 5) Yeni Django paletleri oluştur ve kaydet ───────────────────
    new_django_paletler = []
    palet_id = baslangic_id
    for pallet_data in optimized:
        p = _create_django_palet(
            pallet_data['items'],
            palet_cfg,
            optimization,
            palet_id,
            'mix',
            items_are_dicts=True,
        )
        new_django_paletler.append(p)
        palet_id += 1

    logger.info(
        "[MR] Service complete: %d → %d mix pallets. %s",
        len(mix_paletler), len(new_django_paletler), metrics.summary(),
    )
    return new_django_paletler, metrics


def merge_repack_mix_service(
    mix_paletler: list,
    palet_cfg,
    optimization,
    baslangic_id: int,
    urun_data_listesi: list,
) -> tuple:
    """
    İteratif BFD Merge & Repack servis katmanı.

    Django Palet listesini packer dict formatına çevirir,
    merge_and_repack_mix (İteratif BFD) çalıştırır ve sonuç iyileşmişse
    eski paletleri DB'den silip yenileri kaydeder.

    Algoritma özeti:
        - En az dolu paleti seç (src).
        - src'deki item'ları hacim desc sırayla (BFD) diğer paletlere taş.
        - Tümü sığar: palet kalkar. Sığmaz: rollback, sonraki src'ye geç.
        - Palet sayısı düşüyor VEYA avg/min doluluk artıyor ise kabul et.

    Args:
        mix_paletler:      Mevcut Django Palet nesneleri (mix türü).
        palet_cfg:         PaletConfig nesnesi.
        optimization:      Django Optimization nesnesi.
        baslangic_id:      Yeni paletler için başlangıç Palet ID'si.
        urun_data_listesi: UrunData nesneleri — item['urun'] referansları için.

    Returns:
        (final_django_paletler, MixMergeMetrics)
        - Kabul edilmezse orijinal liste, DB değiştirilmeden döner.
    """
    from src.core.merge_repack import merge_and_repack_mix, MixMergeMetrics
    import logging
    svc_logger = logging.getLogger(__name__)

    if len(mix_paletler) < 2:
        return mix_paletler, MixMergeMetrics.no_op("insufficient_pallets")

    # ── 1) UrunData hızlı erişim sözlüğü ─────────────────────────────
    urun_data_by_id = {ud.id: ud for ud in urun_data_listesi}

    # ── 2) Django Palet → packer dict ───────────────────────────────
    packer_pallets = []
    for palet in mix_paletler:
        konumlar = palet.json_to_dict(palet.urun_konumlari)
        boyutlar = palet.json_to_dict(palet.urun_boyutlari)
        items = []
        total_weight = 0.0
        for urun_id_str, pos in konumlar.items():
            urun_id = int(urun_id_str)
            ud = urun_data_by_id.get(urun_id)
            if ud is None:
                svc_logger.warning("[MixMerge] UrunData id=%s bulunamadı", urun_id_str)
                continue
            dim = boyutlar.get(urun_id_str, [ud.boy, ud.en, ud.yukseklik])
            items.append({
                'urun': ud,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
                'L': float(dim[0]),
                'W': float(dim[1]),
                'H': float(dim[2]),
            })
            total_weight += ud.agirlik
        packer_pallets.append({'items': items, 'weight': total_weight})

    # ── 3) İteratif BFD ──────────────────────────────────────────
    optimized, metrics = merge_and_repack_mix(packer_pallets, palet_cfg)

    if not metrics.accepted:
        return mix_paletler, metrics

    # ── 4) Eski Django paletleri sil ───────────────────────────────
    for palet in mix_paletler:
        palet.delete()

    # ── 5) Yeni Django paletleri oluştur ve kaydet ────────────────
    new_django_paletler = []
    palet_id = baslangic_id
    for pallet_data in optimized:
        p = _create_django_palet(
            pallet_data['items'],
            palet_cfg,
            optimization,
            palet_id,
            'mix',
            items_are_dicts=True,
        )
        new_django_paletler.append(p)
        palet_id += 1

    svc_logger.info(
        "[MixMerge] Service done: %d → %d pallets. %s",
        len(mix_paletler), len(new_django_paletler), metrics.summary(),
    )
    return new_django_paletler, metrics


def mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization):
    """
    mix_palet_yerlestirme fonksiyonunun döndürdüğü dict listesini
    Django Palet nesnelerine dönüştürür.
    """
    django_paletler = []
    
    for pallet_dict in mix_palet_data:
        palet = Palet(
            optimization=optimization,
            palet_id=pallet_dict['id'],
            palet_tipi=None,
            palet_turu='mix',
            custom_en=palet_cfg.width,
            custom_boy=palet_cfg.length,
            custom_max_yukseklik=palet_cfg.height,
            custom_max_agirlik=palet_cfg.max_weight
        )
        
        urun_konumlari = {}
        urun_boyutlari = {}
        toplam_agirlik = 0.0
        kullanilan_hacim = 0.0
        
        for item in pallet_dict.get('items', []):
            urun = item['urun']
            urun_id = str(urun.id)
            urun_konumlari[urun_id] = [item['x'], item['y'], item['z']]
            urun_boyutlari[urun_id] = [item['L'], item['W'], item['H']]
            toplam_agirlik += urun.agirlik
            kullanilan_hacim += (item['L'] * item['W'] * item['H'])
        
        palet.urun_konumlari = urun_konumlari
        palet.urun_boyutlari = urun_boyutlari
        palet.toplam_agirlik = toplam_agirlik
        palet.kullanilan_hacim = kullanilan_hacim
        palet.save()
        
        django_paletler.append(palet)
    
    return django_paletler


# ====================================================================
# GÖRSELLEŞTİRME SERVİSİ
# ====================================================================

def palet_gorsellestir(palet, urunler, save_to_file=True):
    """
    Matplotlib ile 3D palet görselleştirme (Django entegrasyonu).
    
    Args:
        palet: Django Palet modeli
        urunler: Django Urun queryset
        save_to_file: True ise ContentFile döndürür
        
    Returns:
        ContentFile veya BytesIO
    """
    PL, PW, PH = palet.boy, palet.en, palet.max_yukseklik
    
    urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
    urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)
    
    items = []
    for urun in urunler:
        uid = str(urun.id)
        if uid not in urun_konumlari:
            continue
        
        pos = urun_konumlari[uid]
        dim = urun_boyutlari[uid]
        
        if isinstance(pos, list):
            pos = tuple(pos)
        if isinstance(dim, list):
            dim = tuple(dim)
        
        items.append({
            'urun_kodu': urun.urun_kodu,
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'L': dim[0], 'W': dim[1], 'H': dim[2]
        })
    
    title = (f'Palet {palet.palet_id} - {palet.palet_turu.upper()}\n'
             f'Doluluk: {palet.doluluk_orani():.1f}%')
    
    buf = render_pallet_3d(PL, PW, PH, items, title=title)
    
    if save_to_file:
        return ContentFile(buf.read())
    return buf


def ozet_grafikler_olustur(optimization):
    """Özet grafikler oluşturur - Plotly ile interaktif HTML."""
    paletler = Palet.objects.filter(optimization=optimization)
    single = paletler.filter(palet_turu='single').count()
    mix = paletler.filter(palet_turu='mix').count()
    
    optimization.single_palet = single
    optimization.mix_palet = mix
    optimization.toplam_palet = single + mix
    optimization.save()
    
    # Palet verilerini hazırla
    palet_data_list = []
    for p in paletler:
        palet_data_list.append({
            'palet_id': p.palet_id,
            'palet_turu': p.palet_turu,
            'doluluk': p.doluluk_orani()
        })
    
    # src/ visualization modülünü kullan
    from src.utils.visualization import render_summary_charts
    return render_summary_charts(palet_data_list)


# ====================================================================
# YARDIMCI FONKSİYONLAR (private)
# ====================================================================

def _create_django_palet(placements, palet_cfg, optimization, palet_id, palet_turu, items_are_dicts=False):
    """
    Yerleşim verilerinden Django Palet nesnesi oluşturur ve kaydeder.
    
    Args:
        placements: Yerleşim listesi
        palet_cfg: PaletConfig nesnesi
        optimization: Django Optimization nesnesi
        palet_id: Palet ID
        palet_turu: 'single' veya 'mix'
        items_are_dicts: True ise placements dict formatında (pack_maximal_rectangles çıktısı)
    """
    palet = Palet(
        optimization=optimization,
        palet_id=palet_id,
        palet_tipi=None,
        palet_turu=palet_turu,
        custom_en=palet_cfg.width,
        custom_boy=palet_cfg.length,
        custom_max_yukseklik=palet_cfg.height,
        custom_max_agirlik=palet_cfg.max_weight
    )
    
    urun_konumlari = {}
    urun_boyutlari = {}
    toplam_agirlik = 0.0
    kullanilan_hacim = 0.0
    
    for item in placements:
        if items_are_dicts:
            urun = item['urun']
            urun_id = str(urun.id)
            urun_konumlari[urun_id] = [item['x'], item['y'], item['z']]
            urun_boyutlari[urun_id] = [item['L'], item['W'], item['H']]
            toplam_agirlik += urun.agirlik
            kullanilan_hacim += (item['L'] * item['W'] * item['H'])
        else:
            urun = item['urun']
            urun_id = str(urun.id)
            urun_konumlari[urun_id] = [item['x'], item['y'], item['z']]
            urun_boyutlari[urun_id] = [item['L'], item['W'], item['H']]
            toplam_agirlik += urun.agirlik
            kullanilan_hacim += (item['L'] * item['W'] * item['H'])
    
    palet.urun_konumlari = urun_konumlari
    palet.urun_boyutlari = urun_boyutlari
    palet.toplam_agirlik = toplam_agirlik
    palet.kullanilan_hacim = kullanilan_hacim
    palet.save()
    
    return palet
