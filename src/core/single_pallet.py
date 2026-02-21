"""
Single (Tek Tip) Palet Yerleştirme Algoritması
=================================================

Aynı boyuttaki ürünleri tek palete grid düzeninde yerleştirir.
Adaptive threshold ile Single Palet onayı verir:
  - Sabit %90 yerine, SKU'nun geometrik taban doluluk oranına
    ve stok büyüklüğüne göre per-SKU eşik hesaplanır.
  - Geometrik kısıtlarla %90'a ulaşamayan SKU'lar haksız yere
    reddedilmez; gerçekçi en iyi threshold seçilir.

Özellikler:
    - Mixed-Orientation Tiling (Karma yönelim döşemesi)
    - Efficiency-Based Evaluation (Verimlilik bazlı değerlendirme)
    - Multi-Strategy Layer Configuration (Çok stratejili katman yapısı)
    - Amazon-like Adaptive Single Threshold
"""

import math

from ..utils.helpers import urun_hacmi
from ..models.container import PaletConfig


# ====================================================================
# ADAPTİVE SINGLE THRESHOLD SABİTLERİ  (Amazon-like per-SKU eşik)
# ====================================================================
# Formul: dynamic_threshold = clamp(
#     BASE_SINGLE_THRESHOLD - K * (base_fill - 0.70),
#     MIN_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD
# )
# Mantık:
#   - base_fill yüksekse (geometrik olarak iyi uyan SKU):
#     eşik düşer; SKU zaten yüksek verimlilik üretir, kolayca kabul edilir.
#   - base_fill düşükse (geometrik olarak zayıf uyan SKU):
#     eşik artarak MAX'a yaklaşır, gerçekten iyi dolmayan SKU kabul edilmez.
#   - K katsayısı: eşik hassasiyetini kontrol eder.

DEFAULT_SINGLE_THRESHOLD = 0.85  # Genel amaçlı tek-tip palet kabul eşiği
BASE_SINGLE_THRESHOLD = 0.82   # Baz eşik (base_fill=0.70 için)
K_FILL_SLOPE          = 0.35   # base_fill kayması başına eşik düşüş hızı
MIN_SINGLE_THRESHOLD  = 0.70   # Mutlak alt sınır
MAX_SINGLE_THRESHOLD  = 0.90   # Mutlak üst sınır

# --- Geriye-uyumluluk sabitler (eski kod bunları önemsemiyorsa sorun yok) ---
GEOM_MARGIN    = 0.03
STOCK_BONUS    = 0.03
STOCK_BONUS_5PLUS = 0.02

# Histerez: eşiğe bu kadar eksik kalan verimlilik de kabul edilir.
# Örn: eşik=0.82, verimlilik=0.816 → 0.816+0.005=0.821 ≥ 0.82 → KABUL
HYSTERESIS = 0.005

# Taban doluluk oranı bu eşiğin altında ise geometri çok kötü demektir;
# tek palet hiçbir zaman kabul edilmez (MAX eşiği döndürülür).
_MIN_BASE_FILL_FOR_SINGLE = 0.55


# ====================================================================
# ADAPTİVE THRESHOLD YARDIMCİ FONKSİYONLARI
# ====================================================================

def compute_max_base_fill(container_L: float, container_W: float,
                          item_L: float, item_W: float
                          ) -> tuple:
    """
    SKU için teorik en iyi 2-D taban doluluk oranını hesaplar.

    Sadece 0° ve 90° rotasyon değerlendirilir (karma tiling dışarıda;
    bu fonksiyon eşik hesabı içindir, fiili yerleştirme değil).

    Args:
        container_L: Palet uzunluğu (cm)
        container_W: Palet genişliği (cm)
        item_L:      Ürün uzunluğu (cm)
        item_W:      Ürün genişliği (cm)

    Returns:
        (max_fill, best_orientation, base_count)
        - max_fill        : float [0..1]  en iyi taban doluluk oranı
        - best_orientation: str   '0deg' | '90deg'
        - base_count      : int   tek katmandaki en fazla ürün sayısı
    """
    if container_L <= 0 or container_W <= 0 or item_L <= 0 or item_W <= 0:
        return 0.0, '0deg', 0

    base_area = container_L * container_W

    # Yönelim 0°: ürün (item_L × item_W)
    nx0 = math.floor(container_L / item_L)
    ny0 = math.floor(container_W / item_W)
    count0 = nx0 * ny0
    fill0  = (count0 * item_L * item_W) / base_area if count0 > 0 else 0.0

    # Yönelim 90°: ürün (item_W × item_L)
    nx1 = math.floor(container_L / item_W)
    ny1 = math.floor(container_W / item_L)
    count1 = nx1 * ny1
    fill1  = (count1 * item_L * item_W) / base_area if count1 > 0 else 0.0

    if fill0 >= fill1:
        return fill0, '0deg', count0
    return fill1, '90deg', count1


def compute_adaptive_single_threshold(max_fill: float,
                                      full_pallet_count: int = 0) -> float:
    """
    Per-SKU adaptive single-pallet kabul eşiği hesaplar.

    Formul:
        t = BASE_SINGLE_THRESHOLD - K_FILL_SLOPE * (max_fill - 0.70)
        t = clamp(t, MIN_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD)

    İyorumlama:
        - max_fill > 0.70 (iyi uyum): eşik düşer → verimlilik zaten yüksek olduğundan kabul kolay.
        - max_fill < 0.70 (kötü uyum): eşik artar, MAX'a kenetlenir → gerçekten iyi dolmayan reddedilir.
        - max_fill == 0  (geometrik imkânsız): MAX eşiki → reddeçen olmaz.

    Args:
        max_fill:          compute_max_base_fill()'dan gelen taban doluluk oranı [0..1]
        full_pallet_count: (opsiyonel, eski API uyumu için; artık formulül dışında)

    Returns:
        float: Eşik [MIN_SINGLE_THRESHOLD .. MAX_SINGLE_THRESHOLD]
    """
    if max_fill <= 0.0:
        return MAX_SINGLE_THRESHOLD   # Geometrik imkânsız → güvenli red

    # Taban doluluk çok düşük → SKU geometrisi paleti verimli dolduramaz,
    # tek palet hiçbir zaman kabul sınırını geçemez.
    if max_fill < _MIN_BASE_FILL_FOR_SINGLE:
        return MAX_SINGLE_THRESHOLD   # eşik=%90 → etkin red

    t = BASE_SINGLE_THRESHOLD - K_FILL_SLOPE * (max_fill - 0.70)
    return max(MIN_SINGLE_THRESHOLD, min(MAX_SINGLE_THRESHOLD, t))


def solve_best_layer_configuration(palet_L, palet_W, item_L, item_W):
    """
    Mixed-Orientation Tiling Algorithm.
    
    Palet tabanına sığabilecek maksimum ürün sayısını hesaplar.
    3 strateji dener:
        1. %100 Yönelim A (item_L × item_W)
        2. %100 Yönelim B (item_W × item_L) 
        3. Karma Yönelim (A satırları + B satırları)
    
    Args:
        palet_L: Palet uzunluğu
        palet_W: Palet genişliği
        item_L: Ürün uzunluğu
        item_W: Ürün genişliği
        
    Returns:
        tuple: (items_per_layer, layout_description, config_dict)
    """
    best_count = 0
    best_layout_desc = ""
    best_config = {'type_a_rows': 0, 'type_b_rows': 0, 'cols_a': 0, 'cols_b': 0, 'orientation': 0}
    
    # Strategy 1: 100% Orientation A
    cols_a = int(palet_L // item_L)
    rows_a_max = int(palet_W // item_W)
    
    if cols_a > 0 and rows_a_max > 0:
        count = cols_a * rows_a_max
        if count > best_count:
            best_count = count
            best_layout_desc = f"{rows_a_max} rows of {cols_a} items ({item_L}×{item_W})"
            best_config = {
                'type_a_rows': rows_a_max, 'type_b_rows': 0,
                'cols_a': cols_a, 'cols_b': 0, 'orientation': 0
            }
    
    # Strategy 2: 100% Orientation B
    cols_b = int(palet_L // item_W)
    rows_b_max = int(palet_W // item_L)
    
    if cols_b > 0 and rows_b_max > 0:
        count = cols_b * rows_b_max
        if count > best_count:
            best_count = count
            best_layout_desc = f"{rows_b_max} rows of {cols_b} items ({item_W}×{item_L})"
            best_config = {
                'type_a_rows': 0, 'type_b_rows': rows_b_max,
                'cols_a': 0, 'cols_b': cols_b, 'orientation': 1
            }
    
    # Strategy 3: Mixed-Orientation Tiling (A satırları + B satırları)
    if cols_a > 0 and cols_b > 0:
        for i in range(1, rows_a_max + 1):
            count_a = cols_a * i
            used_width_a = i * item_W
            remaining_width = palet_W - used_width_a
            rows_b_possible = int(remaining_width // item_L)
            count_b = cols_b * rows_b_possible
            total_count = count_a + count_b
            
            if total_count > best_count:
                best_count = total_count
                best_layout_desc = (f"{i} rows Type-A ({cols_a} items @ {item_L}×{item_W}) + "
                                   f"{rows_b_possible} rows Type-B ({cols_b} items @ {item_W}×{item_L})")
                best_config = {
                    'type_a_rows': i, 'type_b_rows': rows_b_possible,
                    'cols_a': cols_a, 'cols_b': cols_b, 'orientation': 2
                }
    
    # Strategy 3b: Reverse Mixed (B satırları + A satırları)
    if cols_a > 0 and cols_b > 0:
        for i in range(1, rows_b_max + 1):
            count_b = cols_b * i
            used_width_b = i * item_L
            remaining_width = palet_W - used_width_b
            rows_a_possible = int(remaining_width // item_W)
            count_a = cols_a * rows_a_possible
            total_count = count_a + count_b
            
            if total_count > best_count:
                best_count = total_count
                best_layout_desc = (f"{i} rows Type-B ({cols_b} items @ {item_W}×{item_L}) + "
                                   f"{rows_a_possible} rows Type-A ({cols_a} items @ {item_L}×{item_W})")
                best_config = {
                    'type_a_rows': rows_a_possible, 'type_b_rows': i,
                    'cols_a': cols_a, 'cols_b': cols_b, 'orientation': 3
                }
    
    return best_count, best_layout_desc, best_config


def generate_grid_placement(items_to_place, palet_cfg: PaletConfig):
    """
    Mixed-Orientation Grid Placement.
    
    Ürünler için X, Y, Z koordinatları üretir.
    Karma yönelim döşemesini destekler.
    
    Args:
        items_to_place: UrunData listesi (aynı tip)
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        list[dict]: Her biri {'urun', 'x', 'y', 'z', 'L', 'W', 'H'} içeren yerleşim
    """
    if not items_to_place:
        return []
    
    u0 = items_to_place[0]
    PL, PW, PH = palet_cfg.length, palet_cfg.width, palet_cfg.height
    
    items_per_layer, layout_desc, layer_config = solve_best_layer_configuration(
        PL, PW, u0.boy, u0.en
    )
    
    if items_per_layer == 0:
        return []
    
    item_H = u0.yukseklik
    max_layers = int(PH // item_H)
    
    placements = []
    item_idx = 0
    
    type_a_rows = layer_config['type_a_rows']
    type_b_rows = layer_config['type_b_rows']
    cols_a = layer_config['cols_a']
    cols_b = layer_config['cols_b']
    orientation = layer_config['orientation']
    
    if orientation in [0, 2, 3]:
        item_L_a, item_W_a = u0.boy, u0.en
    else:
        item_L_a, item_W_a = u0.en, u0.boy
    
    if orientation in [1, 2, 3]:
        item_L_b, item_W_b = u0.en, u0.boy
    else:
        item_L_b, item_W_b = u0.boy, u0.en
    
    for layer in range(max_layers):
        z = layer * item_H
        y_offset = 0
        
        for row in range(type_a_rows):
            y = y_offset + row * item_W_a
            for col in range(cols_a):
                if item_idx >= len(items_to_place):
                    return placements
                x = col * item_L_a
                placements.append({
                    'urun': items_to_place[item_idx],
                    'x': x, 'y': y, 'z': z,
                    'L': item_L_a, 'W': item_W_a, 'H': item_H
                })
                item_idx += 1
        
        y_offset += type_a_rows * item_W_a
        
        for row in range(type_b_rows):
            y = y_offset + row * item_W_b
            for col in range(cols_b):
                if item_idx >= len(items_to_place):
                    return placements
                x = col * item_L_b
                placements.append({
                    'urun': items_to_place[item_idx],
                    'x': x, 'y': y, 'z': z,
                    'L': item_L_b, 'W': item_W_b, 'H': item_H
                })
                item_idx += 1
    
    return placements


def simulate_single_pallet(urun_listesi, palet_cfg: PaletConfig):
    """
    Efficiency-Based Single Pallet Simulation.
    
    Ürün grubunun Single Palet'e uygun olup olmadığını değerlendirir.
    %90+ verimlilik eşiği kullanır.
    
    Args:
        urun_listesi: Aynı tipteki UrunData listesi
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        dict: {
            'can_be_single': bool,
            'capacity': int,
            'pack_count': int,
            'efficiency': float,
            'layout_desc': str,
            'reason': str
        }
    """
    if not urun_listesi:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": "", "reason": "Empty product list"
        }
    
    u0 = urun_listesi[0]
    PL, PW, PH = palet_cfg.length, palet_cfg.width, palet_cfg.height
    max_w = palet_cfg.max_weight
    
    # 1. Optimal katman konfigürasyonu
    items_per_layer, layout_desc, layer_config = solve_best_layer_configuration(
        PL, PW, u0.boy, u0.en
    )
    
    if items_per_layer == 0:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": "No valid configuration",
            "reason": "Product dimensions exceed pallet size"
        }
    
    # 2. Maksimum katman sayısı
    max_layers = int(PH // u0.yukseklik)
    if max_layers == 0:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": layout_desc,
            "reason": "Product height exceeds pallet height"
        }
    
    # 3. Hacim bazlı kapasite
    capacity_by_volume = items_per_layer * max_layers
    
    # 4. Ağırlık bazlı kapasite
    capacity_by_weight = int(max_w / u0.agirlik) if u0.agirlik > 0 else 999999
    
    # 5. Final kapasite
    capacity = min(capacity_by_volume, capacity_by_weight)
    
    # 6. Verimlilik hesabı
    item_volume = urun_hacmi(u0)
    pallet_volume = palet_cfg.volume
    efficiency = (capacity * item_volume) / pallet_volume
    
    # 7. Adaptive threshold hesabı
    #    full_pallet_count: stoğun kaç tam palet oluşturduğu (stok bonusu için)
    full_pallet_count = capacity // max(1, items_per_layer) if items_per_layer > 0 else 0
    # Gerçek stok üzerinden de hesapla; hangisi küçükse onu kullan
    stock_pallets = len(urun_listesi) // max(1, capacity) if capacity > 0 else 0
    full_pallet_count = max(full_pallet_count, stock_pallets)

    max_fill, best_orient, base_count = compute_max_base_fill(
        PL, PW, u0.boy, u0.en
    )
    adaptive_threshold = compute_adaptive_single_threshold(max_fill, full_pallet_count)

    # Eski davranışı koru: efficiency >= 0.90 her zaman kabul
    # Histerez: verimlilik eşiğe HYSTERESIS kadar yakın ise kabul et
    hysteresis_margin = (efficiency - adaptive_threshold) * 100  # %
    is_suitable = (efficiency + HYSTERESIS >= adaptive_threshold)

    # 8. Mevcut stok için pack_count
    current_stock = len(urun_listesi)
    pack_count = min(current_stock, capacity)

    # 9. Detaylı açıklama (genişletilmiş loglama)
    if is_suitable:
        if capacity_by_volume < capacity_by_weight:
            constraint = "volume-limited"
        elif capacity_by_weight < capacity_by_volume:
            constraint = "weight-limited"
        else:
            constraint = "perfectly balanced"
        reason = (
            f"ONAYLANDI | Efficiency: {efficiency*100:.1f}% "
            f">= dynamic_threshold {adaptive_threshold*100:.1f}% ({constraint}) | "
            f"hysteresis_margin: {hysteresis_margin:+.2f}pp | "
            f"base_fill: {max_fill*100:.1f}% [{best_orient}, base_count/layer={base_count}] | "
            f"full_pallet_count: {full_pallet_count} | "
            f"Capacity: {capacity} items | Layout: {layout_desc}"
        )
    else:
        reason = (
            f"REDDEDILDI | Efficiency: {efficiency*100:.1f}% "
            f"< dynamic_threshold {adaptive_threshold*100:.1f}% | "
            f"hysteresis_margin: {hysteresis_margin:+.2f}pp | "
            f"base_fill: {max_fill*100:.1f}% [{best_orient}, base_count/layer={base_count}] | "
            f"full_pallet_count: {full_pallet_count} | "
            f"Capacity: {capacity} items | Layout: {layout_desc} | "
            f"Esik [{MIN_SINGLE_THRESHOLD*100:.0f}%-{MAX_SINGLE_THRESHOLD*100:.0f}%] araliginda (base_fill bazli)"
        )

    return {
        "can_be_single": is_suitable,
        "capacity": capacity,
        "pack_count": pack_count,
        "efficiency": efficiency,
        "layout_desc": layout_desc,
        "reason": reason,
        # Ek debug alanları (mevcut API'yi bozmaz, sadece ekler)
        "adaptive_threshold": adaptive_threshold,
        "max_base_fill": max_fill,
        "best_base_orientation": best_orient,
        "base_count_per_layer": base_count,
        "full_pallet_count": full_pallet_count,
    }


# ====================================================================
# SANITY CHECK (bağımsız, pytest veya doğrudan çalıştırılabilir)
# ====================================================================

def _run_sanity_checks():
    """
    compute_max_base_fill ve compute_adaptive_single_threshold için
    hızlı doğrulama testleri.  Dış bağımlılık gerektirmez.

    Çalıştırma:
        python -c "from src.core.single_pallet import _run_sanity_checks; _run_sanity_checks()"
    """
    FAIL = []

    def chk(name, val, lo, hi):
        ok = lo <= val <= hi
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {val:.4f}  (beklenen [{lo:.4f}, {hi:.4f}])")
        if not ok:
            FAIL.append(name)

    print("\n=== Adaptive Single Threshold Sanity Checks (yeni formül) ===\n")

    # --- Test 1: 40×30 ürün, 120×100 palet ---
    # max_fill=0.90; t = 0.82 - 0.35*(0.90-0.70) = 0.82 - 0.07 = 0.75
    # clamp([0.70, 0.90]) → 0.75
    mf1, ori1, bc1 = compute_max_base_fill(120, 100, 40, 30)
    t1 = compute_adaptive_single_threshold(mf1)
    print(f"  Item 40×30 on 120×100 | max_fill={mf1:.4f} [{ori1}, base_count={bc1}]")
    chk("max_fill ~0.90",    mf1, 0.85, 1.01)
    chk("t1 ~0.75",          t1,  0.73, 0.77)
    chk("t1 in [0.70,0.90]", t1,  MIN_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD)

    print()

    # --- Test 2: 45×35 ürün, 120×100 palet ---
    # max_fill=0.7875; t = 0.82 - 0.35*(0.7875-0.70) = 0.82 - 0.030625 = 0.789
    # clamp([0.70, 0.90]) → ~0.789
    mf2, ori2, bc2 = compute_max_base_fill(120, 100, 45, 35)
    t2 = compute_adaptive_single_threshold(mf2)
    print(f"  Item 45×35 on 120×100 | max_fill={mf2:.4f} [{ori2}, base_count={bc2}]")
    chk("max_fill ~0.79",    mf2, 0.78, 0.80)
    chk("t2 ~0.789",         t2,  0.77, 0.80)
    chk("t2 in [0.70,0.90]", t2,  MIN_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD)

    print()

    # --- Test 3: Çok kötü uyum, base_fill=0.40 ---
    # t = 0.82 - 0.35*(0.40-0.70) = 0.82 + 0.105 = 0.925 → clamp → MAX (0.90)
    t3 = compute_adaptive_single_threshold(0.40)
    chk("base_fill=0.40 -> MAX (0.90)", t3, MAX_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD)

    # --- Test 4: max_fill mükemmel = 1.0 ---
    # t = 0.82 - 0.35*(1.0-0.70) = 0.82 - 0.105 = 0.715 → clamp → max(0.70, 0.715) = 0.715
    t4 = compute_adaptive_single_threshold(1.0)
    chk("base_fill=1.0 -> ~0.715",  t4, 0.70, 0.75)

    print()

    # --- Test 5: max_fill==0 → MAX ---
    t_zero = compute_adaptive_single_threshold(0.0)
    chk("max_fill=0 -> MAX threshold", t_zero, MAX_SINGLE_THRESHOLD, MAX_SINGLE_THRESHOLD)

    # --- Test 6: 200 rastgele senaryo, clamp garantisi ---
    import random
    random.seed(42)
    for _ in range(200):
        mf = random.uniform(0.0, 1.2)
        t  = compute_adaptive_single_threshold(mf)
        if not (MIN_SINGLE_THRESHOLD <= t <= MAX_SINGLE_THRESHOLD):
            FAIL.append(f"clamp_violation: mf={mf:.3f} t={t:.3f}")
    if not any("clamp" in f for f in FAIL):
        print(f"  [PASS] 200 rastgele senaryo, threshold her zaman "
              f"[{MIN_SINGLE_THRESHOLD}, {MAX_SINGLE_THRESHOLD}] içinde")

    print()
    if FAIL:
        print(f"UYARI: {len(FAIL)} test BASARISIZ: {FAIL}")
    else:
        print("Tum sanity check'ler gecti.\n")
    return len(FAIL) == 0
