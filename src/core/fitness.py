"""
Fitness Değerlendirme Modülü
==============================

Genetik Algoritma bireylerinin (kromozomların) kalitesini ölçer.

Değerlendirme Kriterleri (Öncelik Sırasıyla):
    1. Palet sayısını minimize et (EN YÜKSEK ÖNCELİK)
    2. Doluluk oranını maksimize et
    3. Fiziksel kısıtları asla ihlal etme (KIRMIZI ÇİZGİ)
    4. Ağırlık merkezi dengesini koru
    5. İstifleme kurallarına uy

Amazon-benzeri gerçekçi istif metrikleri (Ek):
    6. Void Penalty   – bounding-box içindeki boş hacim cezası
    7. Edge Bias      – kenarlara yaslama ödülü
    8. Cavity Penalty – iç boşluk/baca kolonları cezası
"""

import logging
import math
import os
from dataclasses import dataclass
from .packing_first_fit import pack_maximal_rectangles_first_fit
from .packing import snap_z, compute_corner_support
from ..utils.helpers import urun_hacmi
from ..models.container import PaletConfig

logger = logging.getLogger(__name__)

DEBUG_SUPPORT = os.getenv("DEBUG_SUPPORT") == "1"
MIN_SUPPORT_RATIO = 0.40

# ====================================================================
# GERÇEKÇİ İSTİF KONFİGÜRASYONU  (Amazon-like packing)
# ====================================================================
# Bu sabitleri buradan ayarlayın; başka dosyaya dokunmanıza gerek yok.

# Void penalty: bounding-box içindeki boş hacim oranı cezası
W_VOID   = 0.8     # [0.6 – 1.2]  Büyüdükçe kompaktlık baskısı artar

# Edge bias: ortalama duvar uzaklığına göre ödül
W_EDGE   = 0.15    # [0.1 – 0.3]  Küçük tutun; palet sayısını etkilemesin

# Cavity penalty: XY footprint'teki iç boşluk (baca/oyuk) cezası
W_CAVITY = 0.35    # [0.2 – 0.6]  Çok büyütmeyin; hesap ya­vaşlar

# Cavity grid çözünürlüğü (cm). Küçüldükçe hassas ama yavaş.
CAVITY_GRID = 5.0  # cm

# Her N fitness değerlendirmesinde bir cavity hesabı yapılır (throttle).
CAVITY_THROTTLE = 4   # 1 = her seferinde, 4 = her 4. bireyde

# Küçük epsilon (float hassasiyeti)
EPS_FITNESS = 1e-9

# ---------------------------------------------------------------
# UNDERFILL + VARYANS  (az dolu paletleri cezalandırır)
# ---------------------------------------------------------------
# Bir paletin doluluk oranı bu değerin altında kalırsa cezalandırılır.
MIN_UTIL    = 0.45    # Eşik doluluk oranı  [0..1]

# Ceza büyüklükleri – doluluk türevi (gerçek fitness birimiyle uyumlu)
W_UNDERFILL = 8000    # (MIN_UTIL - u)^2 çarpanı  [5000–15000]
W_VARIANCE  = 3000    # Palet doluluk varyansı çarpanı  [1000–5000]

# AMAZON-VARI DENGESİZLİK ÖDÜLÜ
# Paletlerin doluluk oranlarının karesini alarak, dengesiz dağılımı (örn: %95, %5)
# dengeli dağılıma (örn: %50, %50) tercih etmesini sağlarız.
W_DENSITY_SQUARE = 5000

# ---------------------------------------------------------------
# KOMPAKSIYON METRİKLERİ  (fragmantasyon + dikey düzleme)
# ---------------------------------------------------------------
# Boş hücre başına ceza; 1 adet bileşik boşluk normaldir (0'dan başla).
W_FRAGMENTATION       = 500    # Ekstra her parça için  [200–1000]
# Üst-z varyansı çarpanı (cm² biriminde); düzün top = düşük varyans.
W_VERTICAL_COMPACTION = 200    # [100–500]

# ---------------------------------------------------------------
# İÇ SAYAÇ – cavity throttle için (modül seviyesi, thread-safe değil
# ama GA tek-thread'li olduğu için sorun olmaz)
_cavity_eval_counter = 0


# ====================================================================
# ADAPTİF AĞIRLIK SİSTEMİ
# ====================================================================

class AdaptiveWeights:
    """
    Performansa göre otomatik ayarlanan fitness ağırlıkları.
    
    GA ilerledikçe mevcut çözüm kalitesine göre ağırlıkları
    dinamik olarak ayarlar.
    """
    
    def __init__(self):
        self.w_pallet_count = 50_000   # Raised: must dominate w_volume
        self.w_optimal_bonus = 150_000
        self.w_volume = 500            # Lowered: secondary to pallet count
        self.w_weight_violation = 1_000_000
        self.w_physical_violation = 10_000_000
        self.w_cog_penalty = 0
        self.w_stacking_penalty = 100_000
        self.MAX_VOLUME = 1_000        # Cap for adapted w_volume
        self.MAX_PALLET_COUNT = 100_000
        
    def adapt(self, best_chromosome, theo_min_pallets):
        """Performansa göre ağırlıkları ayarla."""
        if not best_chromosome:
            return
            
        palet_sayisi = best_chromosome.palet_sayisi
        doluluk = best_chromosome.ortalama_doluluk
        
        # KURAL 1: Palet sayısı fazlaysa → pallet_count artır
        if palet_sayisi > theo_min_pallets + 2:
            self.w_pallet_count = min(self.w_pallet_count * 1.1, self.MAX_PALLET_COUNT)
            print(f"  Palet sayisi yuksek -> w_pallet_count artirildi: {self.w_pallet_count:.0f}")
        elif palet_sayisi <= theo_min_pallets:
            self.w_pallet_count = max(self.w_pallet_count * 0.95, 10000)
            
        # KURAL 2: Doluluk düşükse → volume weight artır
        if doluluk < 0.65:
            self.w_volume = min(self.w_volume * 1.1, self.MAX_VOLUME)
            print(f"  Doluluk dusuk (%{doluluk*100:.1f}) -> w_volume artirildi: {self.w_volume:.0f}")
        elif doluluk > 0.85:
            self.w_volume = max(self.w_volume * 0.98, 15000)
            
        # KURAL 3: Optimal bonus'u dinamik ayarla
        if palet_sayisi == theo_min_pallets:
            self.w_optimal_bonus = min(self.w_optimal_bonus * 1.1, 300000)
        else:
            self.w_optimal_bonus = max(self.w_optimal_bonus * 0.95, 100000)
    
    def to_dict(self):
        return {
            "w_pallet_count": self.w_pallet_count,
            "w_optimal_bonus": self.w_optimal_bonus,
            "w_volume": self.w_volume,
            "w_weight_violation": self.w_weight_violation,
            "w_physical_violation": self.w_physical_violation,
            "w_cog_penalty": self.w_cog_penalty,
            "w_stacking_penalty": self.w_stacking_penalty
        }


# Global adaptive weights instance
_adaptive_weights = AdaptiveWeights()


def get_weights():
    """Mevcut ağırlıkları döndür."""
    return _adaptive_weights.to_dict()


def get_ga_weights():
    """Geriye uyumluluk için güncel ağırlıkları döndür."""
    return _adaptive_weights.to_dict()


def adapt_weights(best_chromosome, theo_min_pallets):
    """Ağırlıkları performansa göre ayarla."""
    _adaptive_weights.adapt(best_chromosome, theo_min_pallets)


# Geriye uyumluluk
GA_WEIGHTS = get_ga_weights()


# ====================================================================
# FITNESS SONUÇ YAPISI
# ====================================================================

@dataclass
class FitnessResult:
    """Fitness hesaplama sonucu."""
    fitness: float
    palet_sayisi: int
    ortalama_doluluk: float


# ====================================================================
# YARDIMCI FONKSİYONLAR
# ====================================================================

def calculate_center_of_gravity(items):
    """Paletin ağırlık merkezini hesaplar."""
    if not items:
        return 0, 0, 0
    
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    weighted_z = 0
    
    for item in items:
        center_x = item['x'] + item['L'] / 2
        center_y = item['y'] + item['W'] / 2
        center_z = item['z'] + item['H'] / 2
        
        weight = getattr(item['urun'], 'agirlik', 1.0)
        if weight <= 0: weight = 1.0
        
        weighted_x += center_x * weight
        weighted_y += center_y * weight
        weighted_z += center_z * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0, 0, 0
    
    cog_x = weighted_x / total_weight
    cog_y = weighted_y / total_weight
    cog_z = weighted_z / total_weight
    
    return cog_x, cog_y, cog_z

def calculate_cog_penalty(pallet, pallet_length, pallet_width):
    """
    Paletin ağırlık merkezini (Center of Gravity) hesaplar ve 
    ideal merkezden sapmaya göre bir ceza puanı üretir.
    """
    if not pallet['items'] or pallet.get('weight', 0) == 0:
        return 0.0

    cog_x, cog_y, cog_z = calculate_center_of_gravity(pallet['items'])
    if cog_x == 0 and cog_y == 0 and cog_z == 0:
        return 0.0

    # İdeal Merkez Koordinatları (Paletin tam ortası)
    ideal_x = pallet_length / 2.0
    ideal_y = pallet_width / 2.0

    # Sapma Oranları (0 ile 1 arası)
    dev_x = abs(cog_x - ideal_x) / ideal_x if ideal_x > 0 else 0
    dev_y = abs(cog_y - ideal_y) / ideal_y if ideal_y > 0 else 0

    # Ceza Hesaplama
    # 1. Yatay Denge Cezası (Merkezden uzaklaştıkça ceza artar)
    xy_penalty = (dev_x**2 + dev_y**2) * 5000  
    
    # 2. Dikey Denge Cezası (Ağırlık merkezi yukarı çıktıkça ceza artar)
    z_penalty = cog_z * 10  

    return xy_penalty + z_penalty


def check_stacking_violations(items):
    """
    İstifleme hatalarını kontrol eder (havada duran kutular).
    Packing motoru ile tutarlı: minimum %40 destek alanı gerektirir.
    O(n) layer grouping ile optimize edilmiştir.

    snap_z kullanımı: layer_map anahtarı her iki tarafta da snap_z ile
    normalize ediliyor; float drift hatası önlenir.
    """
    violations = 0

    layer_map = {}
    for item in items:
        top = snap_z(item["z"] + item["H"])
        layer_map.setdefault(top, []).append(item)

    for item in items:
        if item["z"] <= 1e-6:
            continue

        item_bottom = snap_z(item["z"])
        item_area = item["L"] * item["W"]
        if item_area == 0:
            continue

        item_x1 = item["x"]
        item_x2 = item["x"] + item["L"]
        item_y1 = item["y"]
        item_y2 = item["y"] + item["W"]

        supported_area = 0.0
        for other in layer_map.get(item_bottom, []):
            if other is item:
                continue
            overlap_x = max(0, min(item_x2, other["x"] + other["L"]) - max(item_x1, other["x"]))
            overlap_y = max(0, min(item_y2, other["y"] + other["W"]) - max(item_y1, other["y"]))
            supported_area += overlap_x * overlap_y

        supported_area = min(supported_area, item_area)

        if (supported_area / item_area) < MIN_SUPPORT_RATIO:
            violations += 1

    return violations


# ====================================================================
# KÖŞE / OVERHANG PER-İTEM CEZA HESABI
# ====================================================================

def _calculate_corner_overhang_penalty(items):
    """
    Her z>0 kutu için köşe destek eksikliği ve çıkıntı mesafesinden
    oşturturulan toplam penalty oranını hesaplar.

    CORNER_HARD_REJECT=True ise packing zaten reddeder;
    bu fonksiyon soft-reject modunda veya fitness şekillendirmede kullanılır.

    layer_map O(1) lookup: snap_z ile normalize edilmiş anahtarlar kullanılır.

    Döner: (corner_score, overhang_score)  ikisi de [0..1]
    """
    if not items:
        return 0.0, 0.0

    # Sadece z>0 üzeri kutular değerlendirilir
    elevated = [i for i in items if i['z'] > 1e-6]
    if not elevated:
        return 0.0, 0.0

    # layer_map: snap_z ile normalize (packing.py ile aynı anahtar)
    layer_map = {}
    for it in items:
        key = snap_z(it['z'] + it['H'])
        layer_map.setdefault(key, []).append(it)

    total_missing_corners = 0
    total_overhang_norm   = 0.0
    n = len(elevated)
    max_diag = 1.0   # normalize için (palet boyutuna göre dinamik yapabilirsiniz)

    for item in elevated:
        key = snap_z(item['z'])
        support_layer = layer_map.get(key, [])
        n_corners, max_oh = compute_corner_support(
            item['x'], item['y'], item['z'],
            item['L'], item['W'],
            support_layer
        )
        # Eksik köşe sayısı (0-4)
        missing = max(0, 4 - n_corners)
        total_missing_corners += missing
        # Overhang: normalize [0..1] (MAX_OVERHANG_CM baz alınarak)
        from .packing import MAX_OVERHANG_CM as _MAX_OH
        oh_norm = max(0.0, min(1.0, max_oh / max(_MAX_OH, 1.0)))
        total_overhang_norm += oh_norm

    corner_score  = max(0.0, min(1.0, total_missing_corners / (4.0 * n)))
    overhang_score = max(0.0, min(1.0, total_overhang_norm  / n))
    return corner_score, overhang_score

def _calculate_void_penalty(items):
    """
    Void Penalty: Bounding-box hacmi ile gerçek kutu hacmi arasındaki
    oranı ölçerek iç boşlukları (U şekli, oyuklar) cezalandırır.

    void_ratio = (bbox_vol - items_vol) / max(bbox_vol, eps)
    Döner: float [0..1]
    """
    if not items:
        return 0.0

    min_x = min(i['x'] for i in items)
    max_x = max(i['x'] + i['L'] for i in items)
    min_y = min(i['y'] for i in items)
    max_y = max(i['y'] + i['W'] for i in items)
    bbox_top_z = max(i['z'] + i['H'] for i in items)

    bbox_vol = (max_x - min_x) * (max_y - min_y) * bbox_top_z
    items_vol = sum(i['L'] * i['W'] * i['H'] for i in items)

    if bbox_vol < EPS_FITNESS:
        return 0.0

    void_ratio = (bbox_vol - items_vol) / bbox_vol
    return max(0.0, min(1.0, void_ratio))   # clamp [0, 1]


def _calculate_edge_score(items, palet_l, palet_w):
    """
    Edge Bias: Ürünlerin duvarlara ortalama uzaklığını ölçer.
    Ürünler kenara ne kadar yakınsa skor o kadar yüksek.

    edge_score = 1 - clamp(avg_min_dist / max(L, W), 0..1)
    Döner: float [0..1]
    """
    if not items:
        return 0.0

    max_dim = max(palet_l, palet_w)
    if max_dim < EPS_FITNESS:
        return 0.0

    dist_sum = 0.0
    for i in items:
        # Her item için 4 duvara olan uzaklıkların minimumu
        d_left   = i['x']
        d_bottom = i['y']
        d_right  = palet_l - (i['x'] + i['L'])
        d_front  = palet_w - (i['y'] + i['W'])
        dist_sum += min(d_left, d_bottom, d_right, d_front)

    avg_dist  = dist_sum / len(items)
    norm_dist = max(0.0, min(1.0, avg_dist / max_dim))
    return 1.0 - norm_dist   # Uzaklık azaldıkça skor artar


def _calculate_cavity_penalty(items, palet_l, palet_w, grid_size=CAVITY_GRID):
    """
    Cavity Penalty: XY ayak izindeki iç boşlukları (baca kolonu, oyuk)
    tespit eder. Basit flood-fill ile boundary'den erişilemeyen boş
    hücreleri "iç boşluk" sayar.

    cavity_ratio = iç_boş_hücre / toplam_hücre
    Döner: float [0..1]

    Performans notu: grid_size büyürse daha hızlı, daha az hassas.
    """
    if not items or palet_l <= 0 or palet_w <= 0:
        return 0.0

    if grid_size <= 0:
        grid_size = CAVITY_GRID

    cols = max(1, int(math.ceil(palet_l / grid_size)))
    rows = max(1, int(math.ceil(palet_w / grid_size)))

    # occupied[r][c] = True ise o hücrede en az 1 kutu var
    occupied = [[False] * cols for _ in range(rows)]

    for item in items:
        c0 = int(item['x'] / grid_size)
        c1 = int(math.ceil((item['x'] + item['L']) / grid_size))
        r0 = int(item['y'] / grid_size)
        r1 = int(math.ceil((item['y'] + item['W']) / grid_size))
        for r in range(max(0, r0), min(rows, r1)):
            for c in range(max(0, c0), min(cols, c1)):
                occupied[r][c] = True

    # Boundary'den flood-fill: boş kenara bağlı hücreler "dış boşluk"
    from collections import deque
    reachable = [[False] * cols for _ in range(rows)]
    queue = deque()

    for r in range(rows):
        for c in [0, cols - 1]:
            if not occupied[r][c] and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if not occupied[r][c] and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not occupied[nr][nc] and not reachable[nr][nc]:
                    reachable[nr][nc] = True
                    queue.append((nr, nc))

    # İç boşluk: dolu değil ve sınırdan da erişilemiyor
    inner_void = sum(
        1 for r in range(rows) for c in range(cols)
        if not occupied[r][c] and not reachable[r][c]
    )
    total_cells = rows * cols
    return max(0.0, min(1.0, inner_void / total_cells))


# ====================================================================
# KOMPAKSIYON METRİK FONKSİYONLARI
# ====================================================================

def compute_void_volume(pallet_items, palet_cfg):
    """
    Gerçek boş hacim (cm³).

    Tüm ürünlerin toplam hacmi palet hacminden çıkarılır.
    Mevcut fill_ratio bonusuyla örtüşse de dışarıdan çağrılabilir
    bağımsız bir metrik olarak kullanılabilir.
    """
    item_vol = sum(i['L'] * i['W'] * i['H'] for i in pallet_items)
    return max(0.0, palet_cfg.volume - item_vol)


def compute_fragmentation_score(pallet_items, palet_l, palet_w, grid_size=10.0):
    """
    XY projeksiyonundaki birbirinden kopuk boş bölge sayısını döndürür.

    Kaba bir ızgara üzerinde BFS yaparak bağlantılı boş alan bileşenlerini
    sayar. Tek bileşen (1) normaldir; fazlası parçalanmış doluluk gösterir.

    grid_size büyüdükçe hızlanır, hassasiyet azalır.
    O(cols × rows) — 120×80 palet için 12×8 = 96 hücre.
    """
    from collections import deque
    if not pallet_items:
        return 0
    cols = max(1, math.ceil(palet_l / grid_size))
    rows = max(1, math.ceil(palet_w / grid_size))
    occupied = [[False] * cols for _ in range(rows)]
    for item in pallet_items:
        c0 = max(0, int(item['x'] / grid_size))
        c1 = min(cols, math.ceil((item['x'] + item['L']) / grid_size))
        r0 = max(0, int(item['y'] / grid_size))
        r1 = min(rows, math.ceil((item['y'] + item['W']) / grid_size))
        for r in range(r0, r1):
            for c in range(c0, c1):
                occupied[r][c] = True
    visited = [[False] * cols for _ in range(rows)]
    components = 0
    for sr in range(rows):
        for sc in range(cols):
            if not occupied[sr][sc] and not visited[sr][sc]:
                components += 1
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not occupied[nr][nc] and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
    return components


def compute_vertical_compaction_score(pallet_items):
    """
    Üst-z (z+H) değerlerinin varyansı.

    Düşük varyans → daha düz üst yüzey → daha kompakt istif.
    Tek öğede anlamsız; 0.0 döndürülür.
    """
    if len(pallet_items) < 2:
        return 0.0
    tops = [i['z'] + i['H'] for i in pallet_items]
    mean_t = sum(tops) / len(tops)
    return sum((t - mean_t) ** 2 for t in tops) / len(tops)


# ====================================================================
# ANA FITNESS FONKSİYONU
# ====================================================================

def evaluate_fitness(chromosome, palet_cfg: PaletConfig) -> FitnessResult:
    """
    Kromozomun başarısını ölçer - Adaptif Ağırlıklar ile.
    
    Motor AUTO-ORIENTATION kullanır (rot_gen gerekmez).
    GA yalnızca SEQUENCE (ürün sırası) optimize eder.

    Ek metrikler (Amazon-like gerçekçilik):
        - Void Penalty   : bounding-box iç boşluklarını cezalandırır
        - Edge Bias      : kenarlara yaslı yerleşim ödüllendirilir
        - Cavity Penalty : iç oyuk/baca kolonları cezalandırılır
    """
    global _cavity_eval_counter
    weights = get_weights()
    
    # 1. Yerleştirme Motorunu Çalıştır (Maximal Rectangles + Auto-Orientation + First-Fit)
    # Enable debug_support if DEBUG_SUPPORT=1 env var is set
    pallets = pack_maximal_rectangles_first_fit(chromosome.urunler, palet_cfg, debug_support=DEBUG_SUPPORT)
    
    if not pallets:
        chromosome.fitness = -1e9
        return FitnessResult(-1e9, 0, 0.0)

    P_GA = len(pallets)
    
    # 2. Teorik Minimum Palet Sayısı
    total_load_vol = sum(urun_hacmi(u) for u in chromosome.urunler)
    theo_min = max(1, math.ceil(total_load_vol / palet_cfg.volume))
    
    # 3. FITNESS HESAPLAMA
    fitness_score = 0.0
    total_fill_ratio = 0.0
    has_violation = False

    # Cavity throttle: her CAVITY_THROTTLE bireyde bir hesap yap
    _cavity_eval_counter += 1
    run_cavity = (_cavity_eval_counter % CAVITY_THROTTLE == 0)
    
    # --- ÖNCELİK 1: PALET SAYISI ---
    if P_GA == theo_min:
        fitness_score += weights["w_optimal_bonus"]
    elif P_GA < theo_min:
        fitness_score += weights["w_optimal_bonus"] * 2
    else:
        extra_pallets = P_GA - theo_min
        fitness_score -= weights["w_pallet_count"] * extra_pallets
    
    # --- ÖNCELİK 2: DOLULUK ORANI (global avg — per-pallet sum kaldırıldı) ---
    # Uses avg_doluluk (scalar [0..1]) instead of summing per-pallet fill^4,
    # which prevented volume inflation from incentivising more pallets.
    for pallet in pallets:
        p_vol = sum(i["L"] * i["W"] * i["H"] for i in pallet["items"])
        fill_ratio = p_vol / palet_cfg.volume
        total_fill_ratio += fill_ratio
        
        # AMAZON-VARI DENGESİZLİK ÖDÜLÜ:
        # Doluluk oranının karesini alarak, bir paleti ağzına kadar doldurmayı
        # iki paleti yarım doldurmaya tercih etmesini sağlıyoruz.
        fitness_score += W_DENSITY_SQUARE * (fill_ratio ** 2)

    avg_doluluk = total_fill_ratio / P_GA if P_GA > 0 else 0.0
    fitness_score += weights["w_volume"] * avg_doluluk

    # --- UNDERFILL + VARYANS CEZALARI ---
    pallet_utils = [
        sum(i["L"] * i["W"] * i["H"] for i in p["items"]) / palet_cfg.volume
        for p in pallets
    ]

    # Underfill: her az-dolu palet için karesel ceza
    underfill_sum = sum((MIN_UTIL - u) ** 2 for u in pallet_utils if u < MIN_UTIL)
    if underfill_sum > 0:
        fitness_score -= W_UNDERFILL * underfill_sum
        if logger.isEnabledFor(logging.DEBUG):
            underfill_count = sum(1 for u in pallet_utils if u < MIN_UTIL)
            logger.debug(
                "[FITNESS] %d/%d palet az-dolu (<%d%%) | utils=%s | underfill_penalty=%.0f",
                underfill_count, P_GA, int(MIN_UTIL * 100),
                [f"{u:.1%}" for u in pallet_utils],
                W_UNDERFILL * underfill_sum,
            )

    # Varyans: paletler arası doluluk tutarsızlığını cezalandır
    # AMAZON-VARI İÇİN İPTAL EDİLDİ: Artık varyans (dengesizlik) istiyoruz!
    # if len(pallet_utils) > 1:
    #     mean_u = sum(pallet_utils) / len(pallet_utils)
    #     variance = sum((u - mean_u) ** 2 for u in pallet_utils) / len(pallet_utils)
    #     fitness_score -= W_VARIANCE * variance
    #     if logger.isEnabledFor(logging.DEBUG):
    #         logger.debug(
    #             "[FITNESS] Doluluk varyansı=%.4f | variance_penalty=%.0f",
    #             variance, W_VARIANCE * variance,
    #         )

        # --- KIRMIZI ÇİZGİ: FİZİKSEL KISIT İHLALLERİ ---
    for pallet in pallets:
        if pallet["weight"] > palet_cfg.max_weight:
            fitness_score -= weights["w_weight_violation"]
            has_violation = True
        
        for item in pallet["items"]:
            if (item["x"] + item["L"] > palet_cfg.length or 
                item["y"] + item["W"] > palet_cfg.width or 
                item["z"] + item["H"] > palet_cfg.height):
                fitness_score -= weights["w_physical_violation"]
                has_violation = True
        
        # Ağırlık Merkezi (Center of Gravity) Cezası
        cog_penalty = calculate_cog_penalty(pallet, palet_cfg.length, palet_cfg.width)
        fitness_score -= cog_penalty
        
        stacking_violations = check_stacking_violations(pallet["items"])
        if stacking_violations > 0:
            fitness_score -= weights["w_stacking_penalty"] * stacking_violations
            has_violation = True

        items = pallet["items"]

        # --- AMAZON-LIKE METRİKLER: Void / Edge / Cavity ---

        # Void Penalty: bounding-box iç boşluğu (bozulmuş yığınlar, U boşlukları)
        void_ratio = _calculate_void_penalty(items)
        fitness_score -= W_VOID * void_ratio

        # Edge Bias: duvara yakın yerleşim ödülü (küçük ama yönlendirici)
        edge_score = _calculate_edge_score(items, palet_cfg.length, palet_cfg.width)
        fitness_score += W_EDGE * edge_score

        # Cavity Penalty: iç oyuklar / baca kolonları (throttled)
        if run_cavity:
            cavity_ratio = _calculate_cavity_penalty(items, palet_cfg.length, palet_cfg.width)
            fitness_score -= W_CAVITY * cavity_ratio

        # Corner/Overhang Penalty: z>0 desteksiz köşeler ve çıkıntı mesafesi
        # CORNER_HARD_REJECT=True ise packing zaten reddeder; burada soft ek baskı.
        corner_score, overhang_score = _calculate_corner_overhang_penalty(items)
        from .packing import W_CORNER_PENALTY
        fitness_score -= W_CORNER_PENALTY * corner_score
        fitness_score -= W_CORNER_PENALTY * overhang_score

        # Fragmantasyon cezası: XY'deki kopuk boş bölge sayısı
        frag = compute_fragmentation_score(items, palet_cfg.length, palet_cfg.width)
        fitness_score -= W_FRAGMENTATION * max(0, frag - 1)   # 1 bölge normaldir

        # Dikey kompaksiyon cezası: üst-z varyansı
        vc = compute_vertical_compaction_score(items)
        fitness_score -= W_VERTICAL_COMPACTION * vc
    
    # Numerik stabilite koruması
    if not isinstance(fitness_score, (int, float)) or math.isnan(fitness_score):
        fitness_score = -1e9

    # Sonuç
    chromosome.fitness = fitness_score
    chromosome.palet_sayisi = P_GA
    chromosome.ortalama_doluluk = avg_doluluk
    
    return FitnessResult(fitness_score, P_GA, avg_doluluk)
