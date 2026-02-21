"""
Merge & Repack — Post-Optimizasyon Palet Konsolidasyonu
=========================================================

GA / DE / Greedy çıktısındaki düşük-doluluklu son paletleri havuza alır,
random restart + volume-desc greedy ile yeniden paketler ve daha iyi bir
çözüm bulunursa kabul eder.

Tasarım ilkeleri:
    - Django'dan tamamen bağımsız (saf Python / packer dict).
    - Mevcut `pack_maximal_rectangles` ve `compact_pallet` altyapısını kullanır.
    - Orijinal UrunData referansları kopyalanmaz, sadece sıralamaları değişir.
    - Lexicographic skorlama: önce palet sayısı ↓, sonra ortalama doluluk ↑.

Kullanım (services.py'den):
    from src.core.merge_repack import merge_and_repack_v2
    pallets, metrics = merge_and_repack_v2(pallets, palet_cfg)
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .packing import (
    compact_pallet,
    _try_add_item,
    _save_pallet_state,
    _restore_pallet_state,
    _rebuild_pallet_state,
)
from .packing_first_fit import pack_maximal_rectangles_first_fit

logger = logging.getLogger(__name__)

# ====================================================================
# SABİTLER
# ====================================================================

# Kabul eşiği: eşit palet sayısında minimum doluluk artışı
_MIN_UTIL_GAIN   = 0.02   # 2 puan iyileşme şart
# Ağırlık baskısı eşiği: bu oranın üzerindeyse K küçültülür
_WEIGHT_PRESSURE_THRESHOLD = 0.92
# Restart başına tahmini süre sınırı (ms) — bu aşılırsa erken dur
_RESTART_TIME_LIMIT_MS = 5_000   # 5 saniye hard timeout


# ====================================================================
# METRİK VERİ YAPISI
# ====================================================================

@dataclass
class MergeRepackMetrics:
    """
    Merge & Repack operasyonunun tam istatistiği.

    Hem loglama hem de Django islem_adimi_ekle() için kullanılır.
    """
    k_selected:         int   = 0      # Seçilen bottom-K palet sayısı
    util_before:        float = 0.0    # Seçilen paletlerin ortalama doluluk oranı (önce)
    util_after:         float = 0.0    # Sonuç paletlerin ortalama doluluk oranı (sonra)
    pallets_before:     int   = 0      # Seçilen palet sayısı (önce)
    pallets_after:      int   = 0      # Sonuç palet sayısı (sonra)
    restarts_tried:     int   = 0      # Gerçekten çalıştırılan restart sayısı
    cache_hits:         int   = 0      # Tekrarlanan sıralama → atlanan restart sayısı
    accepted:           bool  = False  # Değişiklik kabul edildi mi?
    fallback_reason:    str   = ""     # Kabul edilmediyse neden?
    elapsed_ms:         float = 0.0    # Toplam süre (ms)
    util_histogram_before: List[float] = field(default_factory=list)
    util_histogram_after:  List[float] = field(default_factory=list)
    score_history:      List[Tuple[int, float, float]] = field(default_factory=list)
    # score_history: [(restart_no, palet_sayisi, avg_util), ...] — her iyileşmede kaydet

    @classmethod
    def no_op(cls, reason: str = "insufficient_pallets") -> "MergeRepackMetrics":
        """Hiçbir işlem yapılmadığında boş metrik."""
        m = cls()
        m.fallback_reason = reason
        return m

    @classmethod
    def failed(cls, reason: str) -> "MergeRepackMetrics":
        m = cls()
        m.fallback_reason = reason
        return m

    def summary(self) -> str:
        """Django islem_adimi_ekle() için özet satır."""
        if not self.accepted:
            return (
                f"[Merge&Repack] KABUL EDİLMEDİ — {self.fallback_reason} | "
                f"K={self.k_selected} "
                f"restarts={self.restarts_tried} cache_hits={self.cache_hits} "
                f"süre={self.elapsed_ms:.0f}ms"
            )
        delta_util = self.util_after - self.util_before
        delta_p    = self.pallets_before - self.pallets_after
        return (
            f"[Merge&Repack] ✅ KABUL — "
            f"Palet: {self.pallets_before}→{self.pallets_after} "
            f"(Δ{delta_p:+d}) | "
            f"Doluluk: {self.util_before:.1%}→{self.util_after:.1%} "
            f"(Δ{delta_util:+.1%}) | "
            f"restarts={self.restarts_tried} cache_hits={self.cache_hits} "
            f"süre={self.elapsed_ms:.0f}ms"
        )

    def debug_log(self) -> None:
        """Ayrıntılı debug çıktısı (logger.debug)."""
        logger.debug("[MR] summary: %s", self.summary())
        logger.debug(
            "[MR] hist_before=%s hist_after=%s",
            [f"{u:.2%}" for u in self.util_histogram_before],
            [f"{u:.2%}" for u in self.util_histogram_after],
        )
        for restart_no, p_count, avg_util in self.score_history:
            logger.debug(
                "[MR] score_history restart=%-3d pallets=%-2d util=%.2%%",
                restart_no, p_count, avg_util * 100,
            )


# ====================================================================
# YARDIMCI FONKSİYONLAR
# ====================================================================

def _pallet_util(pallet: dict, palet_vol: float) -> float:
    """Palet doluluk oranı (0.0–1.0)."""
    if not pallet.get('items'):
        return 0.0
    used = sum(i['L'] * i['W'] * i['H'] for i in pallet['items'])
    return min(1.0, used / palet_vol)


def _avg_util(pallets: list, palet_vol: float) -> float:
    """Palet listesinin ortalama doluluk oranı."""
    if not pallets:
        return 0.0
    return sum(_pallet_util(p, palet_vol) for p in pallets) / len(pallets)


def _lex_score(pallets: list, palet_vol: float) -> Tuple[int, float]:
    """
    Lexicographic skor: (palet_sayisi, -avg_util).
    Python tuple karşılaştırmasıyla: küçük = daha iyi.
    """
    return (len(pallets), -_avg_util(pallets, palet_vol))


def _weight_pressure(pallets: list, max_weight: float) -> float:
    """En dolu paletin ağırlık baskısı (0.0–1.0)."""
    if not pallets or max_weight <= 0:
        return 0.0
    return max(p.get('weight', 0.0) / max_weight for p in pallets)


def shuffled_variant(
    base_order: List[int],
    strength: int,
    seed: int,
) -> List[int]:
    """
    Katmanlı pertürbasyon: volume-desc sırasını korumak için
    arka kısmı karıştır, ön kısmı (büyük hacimler) sabit tut.

    strength=0-1 → son %30 karıştır   (küçük ürünler)
    strength=2-4 → son %50 karıştır
    strength≥5   → tüm liste karıştır  (aggressive)
    """
    result = list(base_order)
    n = len(result)
    rng = random.Random(seed)

    if strength <= 1:
        start = int(n * 0.70)
    elif strength <= 4:
        start = int(n * 0.50)
    else:
        start = 0

    tail = result[start:]
    rng.shuffle(tail)
    result[start:] = tail
    return result


# ====================================================================
# ANA FONKSİYON
# ====================================================================

def merge_and_repack_v2(
    pallets: List[dict],
    palet_cfg,
    n_restarts: Optional[int] = None,
    k_pallets: Optional[int] = None,
    min_util_gain: float = _MIN_UTIL_GAIN,
    min_support_ratio: float = 0.40,
) -> Tuple[List[dict], MergeRepackMetrics]:
    """
    Merge & Repack v2 — Random Restart ile palet konsolidasyonu.

    Args:
        pallets:          pack_maximal_rectangles formatındaki palet listesi.
                          Her dict: {'items': [{urun, x, y, z, L, W, H}], 'weight': float}.
                          Dizinin içindeki 'urun' alanları UrunData nesnesi OLMALI.
        palet_cfg:        PaletConfig (length, width, height, max_weight).
        n_restarts:       Kaç random restart denensin? None → otomatik.
        k_pallets:        Kaç bottom-K palet seçilsin? None → otomatik.
        min_util_gain:    Eşit palet sayısında kabul için minimum doluluk artışı.
        min_support_ratio:Packing motoruna iletilen gravity constraint parametresi.

    Returns:
        (optimized_pallets, metrics)
        - optimized_pallets: Geliştirilmiş (veya orijinal fallback) palet listesi.
        - metrics:           MergeRepackMetrics nesnesi.
    """
    t_start = time.perf_counter()
    palet_vol = palet_cfg.volume

    # ── Yeterli palet var mı? ─────────────────────────────────────────
    if len(pallets) < 2:
        return pallets, MergeRepackMetrics.no_op("insufficient_pallets (<2)")

    # ── AŞAMA 1: Bottom-K seç ─────────────────────────────────────────────
    utils = sorted(
        enumerate(pallets),
        key=lambda t: _pallet_util(t[1], palet_vol),
    )   # artan doluluk sırası

    # Otomatik K:  toplam paletin %40'ı, minimum 2, maksimum 3
    K = k_pallets or min(3, max(2, math.ceil(len(pallets) * 0.4)))

    # Ağırlık baskısı yüksekse K'yı azalt (merge başarısız olmaması için)
    bottom_k_pallets = [pallets[i] for i, _ in utils[:K]]
    wp = _weight_pressure(bottom_k_pallets, palet_cfg.max_weight)
    if wp >= _WEIGHT_PRESSURE_THRESHOLD:
        K = max(2, K - 1)
        logger.debug("[MR] High weight pressure %.2f → K reduced to %d", wp, K)

    merge_indices: Set[int] = {i for i, _ in utils[:K]}
    merge_pool_pallets = [pallets[i] for i in merge_indices]
    stable_pallets     = [p for i, p in enumerate(pallets) if i not in merge_indices]

    util_before       = _avg_util(merge_pool_pallets, palet_vol)
    hist_before       = [round(_pallet_util(p, palet_vol), 4) for p in merge_pool_pallets]
    original_count    = K

    metrics = MergeRepackMetrics(
        k_selected=K,
        util_before=util_before,
        pallets_before=original_count,
        util_histogram_before=hist_before,
    )

    # ── AŞAMA 2: Havuz oluştur ────────────────────────────────────────
    # item dict içindeki 'urun' alanı UrunData referansı — kopyalanmıyor
    pool_items = [item for p in merge_pool_pallets for item in p['items']]

    if not pool_items:
        metrics.fallback_reason = "empty_pool"
        elapsed = (time.perf_counter() - t_start) * 1000
        metrics.elapsed_ms = elapsed
        return pallets, metrics

    # ── AŞAMA 3: Base sıralama — volume descending (greedy için optimal başlangıç)
    pool_items_sorted = sorted(pool_items, key=lambda i: i['L'] * i['W'] * i['H'], reverse=True)
    # pack_maximal_rectangles UrunData listesi istiyor:
    base_urunler = [item['urun'] for item in pool_items_sorted]

    # ── AŞAMA 4: Restart döngüsü ─────────────────────────────────────
    n = len(base_urunler)
    N = n_restarts or min(100, max(20, n * 2))

    best_pallets: Optional[List[dict]] = None
    best_score   = (math.inf, math.inf)   # (palet_sayisi, -avg_util) → minimize
    seen_orders: Set[tuple] = set()
    cache_hits = 0
    restarts_tried = 0
    score_history: List[Tuple[int, float, float]] = []

    for i in range(N):
        # Zaman limiti kontrolü
        elapsed_now = (time.perf_counter() - t_start) * 1000
        if elapsed_now > _RESTART_TIME_LIMIT_MS:
            logger.debug("[MR] Time limit hit at restart %d (%.0f ms)", i, elapsed_now)
            break

        # ── Sıralama belirle ─────────────────────────────────────────
        if i == 0:
            urunler_order = base_urunler
        else:
            idx_order = shuffled_variant(list(range(n)), strength=i // 5, seed=i)
            urunler_order = [base_urunler[j] for j in idx_order]

        # ── Tekrar sıralama kontrolü (cache) ─────────────────────────
        order_key = tuple(u.id for u in urunler_order)
        if order_key in seen_orders:
            cache_hits += 1
            continue
        seen_orders.add(order_key)
        restarts_tried += 1

        # ── Packing ──────────────────────────────────────────────────
        try:
            new_pallets = pack_maximal_rectangles_first_fit(
                urunler_order,
                palet_cfg,
                min_support_ratio=min_support_ratio,
            )
        except ValueError as exc:
            logger.warning("[MR] pack_maximal_rectangles_first_fit raised ValueError: %s", exc)
            continue

        # ── Kompaksiyon ───────────────────────────────────────────────
        for p in new_pallets:
            compact_pallet(p, palet_cfg)

        # ── Skor hesapla ─────────────────────────────────────────────
        score = _lex_score(new_pallets, palet_vol)

        if score < best_score:
            best_score   = score
            best_pallets = new_pallets
            avg_u = _avg_util(new_pallets, palet_vol)
            score_history.append((restarts_tried, len(new_pallets), avg_u))
            logger.debug(
                "[MR] restart=%-3d  pallets=%-2d  util=%.2%%  ← new best",
                restarts_tried, len(new_pallets), avg_u * 100,
            )

    metrics.restarts_tried = restarts_tried
    metrics.cache_hits     = cache_hits
    metrics.score_history  = score_history

    # ── AŞAMA 5: Kabul kriteri ────────────────────────────────────────
    if best_pallets is None:
        metrics.fallback_reason = "all_restarts_failed"
        metrics.elapsed_ms = (time.perf_counter() - t_start) * 1000
        return pallets, metrics

    merged_count = len(best_pallets)
    util_after   = _avg_util(best_pallets, palet_vol)

    cond_A = merged_count < original_count                           # palet azaldı
    cond_B = (merged_count == original_count
              and util_after >= util_before + min_util_gain)         # doluluk arttı

    accepted = cond_A or cond_B

    metrics.util_after           = util_after
    metrics.pallets_after        = merged_count
    metrics.util_histogram_after = [round(_pallet_util(p, palet_vol), 4) for p in best_pallets]
    metrics.accepted             = accepted

    elapsed = (time.perf_counter() - t_start) * 1000
    metrics.elapsed_ms = elapsed

    if accepted:
        final = stable_pallets + best_pallets
        metrics.debug_log()
        logger.info("[MR] Accepted. %s", metrics.summary())
        return final, metrics

    # ── Fallback: soft retry (min_util_gain=0) ────────────────────────
    # Eşit palet sayısında herhangi bir iyileşme varsa kabul et
    if merged_count == original_count and util_after > util_before:
        metrics.accepted             = True
        metrics.fallback_reason      = "soft_accept (gain<threshold)"
        final = stable_pallets + best_pallets
        metrics.debug_log()
        logger.info("[MR] Soft-accepted. %s", metrics.summary())
        return final, metrics

    metrics.fallback_reason = (
        f"no_improvement (merged={merged_count} orig={original_count} "
        f"util_before={util_before:.1%} util_after={util_after:.1%})"
    )
    logger.info("[MR] Rejected (fallback). %s", metrics.summary())
    return pallets, metrics


# ====================================================================
# MERGE & REPACK MIX — İteratif BFD Konsolidasyonu
# ====================================================================

@dataclass
class IterMergeLog:
    """Tek bir iterasyonun (1 palet birleştirme denemesinin) kaydı."""
    iteration:        int
    src_pallet_idx:   int    # Kaynak paletin o anki sıra indeksi
    src_util:         float  # Birleştirilmeye çalışılan palet doluluğu
    items_count:      int    # Kaç item taşınmaya çalışıldı
    items_moved:      int    # Kaçı başarıyla taşındı
    success:          bool
    reason:           str = ""


@dataclass
class MixMergeMetrics:
    """
    merge_and_repack_mix() operasyonunun tam istatistiği.

    Hem DEBUG loglama hem de Django islem_adimi_ekle() için kullanılır.
    """
    pallets_before:   int   = 0
    pallets_after:    int   = 0
    avg_util_before:  float = 0.0
    avg_util_after:   float = 0.0
    min_util_before:  float = 0.0
    min_util_after:   float = 0.0
    max_util_before:  float = 0.0
    max_util_after:   float = 0.0
    total_items:      int   = 0    # Toplam item sayısı (kontrol)
    accepted:         bool  = False
    fallback_reason:  str   = ""
    elapsed_ms:       float = 0.0
    iterations: List[IterMergeLog] = field(default_factory=list)

    @classmethod
    def no_op(cls, reason: str = "") -> "MixMergeMetrics":
        m = cls()
        m.fallback_reason = reason
        return m

    def summary(self) -> str:
        if not self.accepted:
            return (
                f"[MixMerge] KABUL EDİLMEDİ — {self.fallback_reason} | "
                f"palet={self.pallets_before} "
                f"avg={self.avg_util_before:.1%} "
                f"süre={self.elapsed_ms:.0f}ms"
            )
        dp = self.pallets_before - self.pallets_after
        du = self.avg_util_after - self.avg_util_before
        return (
            f"[MixMerge] ✅ KABUL — "
            f"Palet: {self.pallets_before}→{self.pallets_after} (Δ{dp:+d}) | "
            f"Avg: {self.avg_util_before:.1%}→{self.avg_util_after:.1%} (Δ{du:+.1%}) | "
            f"Min: {self.min_util_before:.1%}→{self.min_util_after:.1%} | "
            f"Max: {self.max_util_before:.1%}→{self.max_util_after:.1%} | "
            f"süre={self.elapsed_ms:.0f}ms"
        )

    def debug_log(self) -> None:
        logger.debug("[MixMerge] %s", self.summary())
        for log in self.iterations:
            status = "✅ OK" if log.success else "❌ FAIL"
            logger.debug(
                "[MixMerge] iter=%-2d  src_idx=%-2d  util=%.1%%  "
                "items=%d  moved=%d  %s  %s",
                log.iteration, log.src_pallet_idx,
                log.src_util * 100, log.items_count,
                log.items_moved, status, log.reason,
            )


def _pallet_remaining_vol(pallet: dict, palet_vol: float) -> float:
    """Paletin kalan boş hacmi (cm³)."""
    used = sum(i['L'] * i['W'] * i['H'] for i in pallet.get('items', []))
    return max(0.0, palet_vol - used)


def _try_merge_one_pallet(
    src: dict,
    targets: List[dict],
    palet_cfg,
    min_support_ratio: float,
    palet_vol: float,
) -> Tuple[bool, int, List[dict], List[dict]]:
    """
    Kaynak paletteki tüm item'ları, BFD sıralamasıyla hedef paletlere taşımayı dener.

    BFD sıralaması:
        - Item'lar hacme göre büyükten küçüğe (Decreasing)
        - Her item için hedef paletler "kalan boş hacim" artan sıraya göre (Best-Fit)
          → En dolu (ama item'ı sığdırabilecek) palete yerleştir

    Başarılı olursa:
        - Hedef paletleri günceller (yerleşim değişiklikleri korunur).
        - True, taşınan_item_sayısı döndürür.
    Başarısız olursa:
        - Tüm hedef paletleri kaydedilen duruma geri alır (rollback).
        - False, 0 döndürür.

    Returns:
        (success, items_moved, saved_targets_states, updated_targets)
    """
    # ── Hedeflerin durumunu kaydet (rollback için) ──────────────────────
    saved_states = [_save_pallet_state(t) for t in targets]

    # ── free_rects yok ise yeniden inşa et ─────────────────────────────
    for t in targets:
        if 'free_rects' not in t or t['free_rects'] is None:
            _rebuild_pallet_state(t, palet_cfg)

    # ── Hacme göre büyükten küçüğe sırala (BFD) ────────────────────────
    items_to_move = sorted(
        src['items'],
        key=lambda i: i['L'] * i['W'] * i['H'],
        reverse=True,
    )

    items_moved = 0
    all_placed  = True

    for item in items_to_move:
        item_vol = item['L'] * item['W'] * item['H']

        # ── Best-Fit: en az boşluğu olan (en dolu) ama item'ı sığdırabilecek hedef
        # Kaba tarama: kalan hacim ≥ item hacmi olan paletler, kalan hacim artan sıra
        candidates = sorted(
            [t for t in targets
             if _pallet_remaining_vol(t, palet_vol) >= item_vol * 0.8],  # %20 tolerans
            key=lambda t: _pallet_remaining_vol(t, palet_vol),
        )

        placed = False
        for target in candidates:
            if _try_add_item(item, target, palet_cfg, min_support_ratio):
                items_moved += 1
                placed = True
                break

        if not placed:
            all_placed = False
            break   # Rollback gerekiyor

    if all_placed:
        return True, items_moved, saved_states, targets

    # ── Rollback: tüm hedefleri eski haline getir ──────────────────────
    for t, saved in zip(targets, saved_states):
        _restore_pallet_state(t, saved)
    return False, items_moved, saved_states, targets


def merge_and_repack_mix(
    pallets: List[dict],
    palet_cfg,
    min_support_ratio: float = 0.40,
) -> Tuple[List[dict], "MixMergeMetrics"]:
    """
    İteratif BFD (Best-Fit Decreasing) Merge & Repack.

    Algoritma (iteratif):
        1. Paletleri doluluk artan sırayla diz.
        2. En az dolu paleti (src) seç.
        3. src'nin item'larını (hacim desc) diğer paletlere BFD ile yerleştirmeye çalış.
           Her item için hedef = kalan boşluğu minimum olan palet (best-fit).
        4. Tüm item'lar yerleştirildiyse src'yi kaldır (palet sayısı -1), başa dön.
        5. Yerleştirilemezse bu src için dene durduğunu işaretle, sıradaki src'ye geç.
           Hiçbir palet kaldırılamazsa dur.

    Kabul kriteri:
        A) Palet sayısı azaldı  (4→3 gibi)
        B) Palet sayısı aynı AMA avg_util en az 2 puan arttı
        C) Palet sayısı aynı AMA min_util arttı (en kötü palet iyileşti)

    Kısıtlar:
        - Container boyutları (palet_cfg.length/width/height)
        - Max ağırlık (palet_cfg.max_weight)
        - Gravity / support constraints (compute_support_ratio)
        - Corner support (compute_corner_support)
        Bunların hepsi _try_add_item() içinde zaten kontrol edilir.

    Args:
        pallets:           packer dict listesi: [{'items': [...], 'weight': float}, ...]
                           Her item dict: {'urun': UrunData, 'x','y','z','L','W','H'}
        palet_cfg:         PaletConfig nesnesi.
        min_support_ratio: Gravity constraint için minimum destek oranı (varsayılan 0.40).

    Returns:
        (optimized_pallets, MixMergeMetrics)
        - Başarısızsa orijinal liste döner (rollback garantisi).
        - Başarılıysa güncellenmiş palet listesi döner (src paletler silinmiş).
    """
    t_start   = time.perf_counter()
    palet_vol = palet_cfg.volume

    # ── Yeterli palet kontrolü ────────────────────────────────────────
    if len(pallets) < 2:
        return pallets, MixMergeMetrics.no_op("insufficient_pallets (<2)")

    # ── Başlangıç metrikleri ──────────────────────────────────────────
    utils_before   = [_pallet_util(p, palet_vol) for p in pallets]
    avg_u_before   = sum(utils_before) / len(utils_before)
    min_u_before   = min(utils_before)
    max_u_before   = max(utils_before)
    total_items    = sum(len(p.get('items', [])) for p in pallets)

    metrics        = MixMergeMetrics(
        pallets_before  = len(pallets),
        avg_util_before = avg_u_before,
        min_util_before = min_u_before,
        max_util_before = max_u_before,
        total_items     = total_items,
    )

    logger.info(
        "[MixMerge] START — pallets=%d  avg=%.1%%  min=%.1%%  max=%.1%%  items=%d",
        len(pallets), avg_u_before * 100, min_u_before * 100,
        max_u_before * 100, total_items,
    )

    # ── Her palet için free_rects hazırla ─────────────────────────────
    for p in pallets:
        if 'free_rects' not in p or p['free_rects'] is None:
            _rebuild_pallet_state(p, palet_cfg)

    # ── Çalışma kopyası (orijinal listeye dokunma) ────────────────────
    working      = list(pallets)   # liste kopyası; dict referanslar aynı
    iteration    = 0
    any_accepted = False

    # Başarısız olan src paletleri tekrar deneme — sonlanma garantisi için
    failed_src_ids: set = set()   # id(pallet_dict) bazlı

    while True:
        if len(working) < 2:
            break

        # ── Doluluk artan sıra ────────────────────────────────────────
        working.sort(key=lambda p: _pallet_util(p, palet_vol))

        # ── Denenmemiş en az dolu src'yi seç ─────────────────────────
        src = None
        src_idx = -1
        for i, p in enumerate(working):
            if id(p) not in failed_src_ids:
                src = p
                src_idx = i
                break

        if src is None:
            break   # Tüm paletler denendi, iyileştirme yok

        iteration += 1
        src_util   = _pallet_util(src, palet_vol)
        targets    = [p for i, p in enumerate(working) if i != src_idx]

        logger.debug(
            "[MixMerge] iter=%-2d  src_idx=%d  src_util=%.1%%  items=%d  targets=%d",
            iteration, src_idx, src_util * 100,
            len(src.get('items', [])), len(targets),
        )

        success, items_moved, _, _ = _try_merge_one_pallet(
            src, targets, palet_cfg, min_support_ratio, palet_vol,
        )

        log = IterMergeLog(
            iteration      = iteration,
            src_pallet_idx = src_idx,
            src_util       = src_util,
            items_count    = len(src.get('items', [])),
            items_moved    = items_moved,
            success        = success,
            reason         = "all_placed" if success else f"failed_at_item_{items_moved + 1}",
        )
        metrics.iterations.append(log)

        if success:
            logger.info(
                "[MixMerge] iter=%d  ✅ Palet kaldırıldı (src_util=%.1%% → %d item taşındı). "
                "Kalan: %d",
                iteration, src_util * 100, items_moved, len(working) - 1,
            )
            working.pop(src_idx)
            any_accepted = True
            failed_src_ids.clear()   # Yeni durum → eski "failed" bilgisi geçersiz
        else:
            logger.debug(
                "[MixMerge] iter=%d  ❌ Birleştirme başarısız (src_util=%.1%% items_moved=%d/%d).",
                iteration, src_util * 100, items_moved, len(src.get('items', [])),
            )
            failed_src_ids.add(id(src))

    # ── Sonuç metrikleri ──────────────────────────────────────────────
    elapsed = (time.perf_counter() - t_start) * 1000
    metrics.elapsed_ms = elapsed

    if not working:
        # Güvenlik: hiç palet kalmamalı (gerçekte olmaz)
        logger.error("[MixMerge] BUG: working list empty after merge — returning originals")
        return pallets, MixMergeMetrics.no_op("bug_empty_result")

    utils_after  = [_pallet_util(p, palet_vol) for p in working]
    avg_u_after  = sum(utils_after) / len(utils_after)
    min_u_after  = min(utils_after)
    max_u_after  = max(utils_after)

    metrics.pallets_after  = len(working)
    metrics.avg_util_after = avg_u_after
    metrics.min_util_after = min_u_after
    metrics.max_util_after = max_u_after

    # ── Kabul kriteri ─────────────────────────────────────────────────
    cond_A = len(working) < len(pallets)                        # palet azaldı
    cond_B = (len(working) == len(pallets)
              and avg_u_after >= avg_u_before + 0.02)           # %2 doluluk artışı
    cond_C = (len(working) == len(pallets)
              and min_u_after > min_u_before + 1e-4)            # en kötü palet iyileşti
    accepted = any_accepted and (cond_A or cond_B or cond_C)

    metrics.accepted = accepted

    if accepted:
        metrics.debug_log()
        logger.info("[MixMerge] DONE. %s", metrics.summary())
        return working, metrics
    else:
        metrics.fallback_reason = (
            "no_net_improvement "
            f"(pallets={len(pallets)}→{len(working)} "
            f"avg={avg_u_before:.1%}→{avg_u_after:.1%})"
        )
        logger.info("[MixMerge] FALLBACK. %s", metrics.summary())
        return pallets, metrics

