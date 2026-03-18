#!/usr/bin/env python3
"""Streamlit tabanlı GA vs DE karşılaştırma aracı.

Bu araç mevcut 3D Bin Packing motorundaki:
- Genetik Algoritma (GA)
- Differential Evolution (DE)

algoritmalarını **mevcut kütüphane giriş noktaları** üzerinden çağırır ve
aynı JSON girdisi üzerinde, Django UI'ye benzer **single + mix** akışıyla
yan yana karşılaştırır.

Önemli noktalar:
- Mevcut GA/DE algoritma mantığına dokunulmaz.
- Parametreler bu araç tarafından "uydurulmaz";
  - GA için `run_ga` fonksiyonunun kendi varsayılan/adaptif akışı kullanılır.
  - DE için `optimize_with_de` fonksiyonunun kendi varsayılan/adaptif akışı kullanılır.
- Django UI veya ORM akışı kullanılmaz; bu dosya tamamen bağımsızdır.
- JSON yalnızca bir kez parse edilir, algoritmalara kopyaları verilir.
- Single pallet pre-pass sonrası **yalnızca mix pool** üzerindeki ürünler için
    GA/DE çalıştırılır; böylece Django UI'daki gerçek akışa daha yakın, adil
    bir kıyaslama sağlanır.
"""

import os
import sys
import json
import time
import copy
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import streamlit as st

# Proje kökünü sys.path'e ekle (main.py ile aynı yaklaşım)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Mevcut parser ve algoritma giriş noktalarını kullan
from src.utils.parser import parse_json_input  # JSON -> (PaletConfig, List[UrunData])
from src.core.genetic_algorithm import run_ga
from src.core.optimizer_de import optimize_with_de
from src.core.single_pallet import simulate_single_pallet, DEFAULT_SINGLE_THRESHOLD
from src.utils.helpers import group_products_smart
from src.models import PaletConfig, UrunData


def load_and_parse_json(uploaded_file):
    """Streamlit'in yüklediği dosyadan JSON okuyup mevcut parser ile parse eder.

    Dönüş:
        (palet_cfg, all_products, json_data)

    Buradaki json_data yalnızca ÖZET göstermek için kullanılır; algoritma
    tarafına verilen veri yine parse_json_input çıktısıdır.
    """
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8")
    json_data = json.loads(text)

    palet_cfg, all_products = parse_json_input(json_data)
    return palet_cfg, all_products, json_data


def build_json_summary(json_data: Dict[str, Any]) -> Dict[str, int]:
    """JSON özetini mevcut UI mantığına yakın şekilde hazırlar.

    Kurallar:
    - Toplam ürün çeşidi: benzersiz ürün kodu sayısı
    - Toplam paket sayısı: detail satırlarındaki `package_quantity` değerlerinin,
      ürün koduna göre gruplanmış toplamı

    Not:
    - `quantity` alanı burada kullanılmaz; o alan paket adedi değil, birim/kg
      miktarını temsil edebilir.
    """
    details = json_data.get("details", []) if isinstance(json_data, dict) else []

    package_totals_by_code: Dict[str, int] = defaultdict(int)

    for det in details:
        product = det.get("product", {}) or {}
        code = str(product.get("code", "")).strip()
        if not code:
            continue

        raw_package_qty = det.get("package_quantity", 0)
        try:
            package_qty = int(raw_package_qty or 0)
        except Exception:
            package_qty = 0

        if package_qty > 0:
            package_totals_by_code[code] += package_qty

    return {
        "total_product_types": len(package_totals_by_code),
        "total_packages": sum(package_totals_by_code.values()),
    }



def clear_runtime_state():
    """Algoritma çalışmaları arasındaki geçici durumu temizlemeye çalışır."""
    import gc

    gc.collect()


def _get_pallet_volume(palet_cfg: PaletConfig) -> float:
    """Palet hacmini döndürür (cm³).

    PaletConfig.volume özelliği varsa onu kullanır; yoksa L*W*H hesaplar.
    """
    return getattr(palet_cfg, "volume", palet_cfg.length * palet_cfg.width * palet_cfg.height)


def run_single_pallet_prepass(
    palet_cfg: PaletConfig,
    products: List[UrunData],
) -> Tuple[int, List[UrunData], float]:
    """Django'daki single_palet_yerlestirme akışının DB'siz, sade versiyonu.

    - group_products_smart ile ürünleri gruplar
    - her grup için simulate_single_pallet çalıştırır
    - full ve partial single paletleri sayar
    - kalan ürünleri mix pool'a atar

    Dönüş:
        (single_pallet_count, mix_pool_products, single_used_volume)
    """
    groups = group_products_smart(products)

    single_pallet_count = 0
    single_used_volume = 0.0
    mix_pool: List[UrunData] = []

    pallet_volume = _get_pallet_volume(palet_cfg)

    for key, group_items in groups.items():
        total_qty = len(group_items)
        if total_qty == 0:
            continue

        sim_result = simulate_single_pallet(group_items, palet_cfg)

        if not sim_result.get("can_be_single"):
            # Grup single olamıyorsa doğrudan mix pool'a gider
            mix_pool.extend(group_items)
            continue

        capacity = int(sim_result.get("capacity", 0) or 0)
        if capacity <= 0:
            # Güvenlik: geçersiz kapasite durumunda grubu mix'e at
            mix_pool.extend(group_items)
            continue

        item = group_items[0]
        item_volume = float(item.boy) * float(item.en) * float(item.yukseklik)

        if total_qty >= capacity:
            num_full_pallets = total_qty // capacity
            remainder = total_qty % capacity
            remainder_fill_ratio = (
                (remainder * item_volume) / pallet_volume if remainder > 0 else 0.0
            )
            create_partial = remainder > 0 and remainder_fill_ratio >= DEFAULT_SINGLE_THRESHOLD
        else:
            num_full_pallets = 0
            remainder = total_qty
            partial_fill_ratio = (remainder * item_volume) / pallet_volume
            create_partial = partial_fill_ratio >= DEFAULT_SINGLE_THRESHOLD
            if not create_partial:
                # Tamamı mix pool'a gider
                mix_pool.extend(group_items)
                continue

        # Full paletler için kullanılan hacim
        single_pallet_count += num_full_pallets
        single_used_volume += num_full_pallets * capacity * item_volume

        # Partial palet kararı
        if create_partial and remainder > 0:
            single_pallet_count += 1
            single_used_volume += remainder * item_volume
        elif remainder > 0:
            # Partial oluşturulmadıysa kalanlar mix pool'a gider
            leftovers = group_items[-remainder:]
            mix_pool.extend(leftovers)

    return single_pallet_count, mix_pool, single_used_volume


def run_ga_with_defaults(palet_cfg: PaletConfig, products: List[UrunData]):
    """GA'yı **hiçbir parametre override etmeden** çalıştırır.

    Tüm parametreler `run_ga` içindeki mevcut adaptif mantıktan gelir.
    """
    urunler_ga = copy.deepcopy(products)

    start = time.perf_counter()
    best_chromosome, history = run_ga(
        urunler=urunler_ga,
        palet_cfg=palet_cfg,
        # population_size, generations vb. BİLEREK geçmiyoruz
    )
    duration = time.perf_counter() - start

    return best_chromosome, history, duration


def run_de_with_defaults(palet_cfg: PaletConfig, products: List[UrunData]):
    """DE'yi Django UI'daki varsayılan akışa daha yakın şekilde çalıştırır.

    Not:
    - Django UI loglarında DE için jenerasyon değeri 60 görünmektedir.
    - UI tarafında NP_USER=40 bilgisi de loglara yansımaktadır.
    - `optimize_with_de` zaten kendi içinde minimum population kuralını
      uyguladığı için, burada UI ile tutarlı olması adına bu iki değer açıkça
      geçirilir.
    """
    urunler_de = copy.deepcopy(products)

    start = time.perf_counter()
    best_chromosome, history = optimize_with_de(
        urunler=urunler_de,
        palet_cfg=palet_cfg,
        population_size=40,
        generations=60,
    )
    duration = time.perf_counter() - start

    return best_chromosome, history, duration


def summarize_chromosome(chromosome) -> Optional[Dict[str, Any]]:
    """GA/DE dönüşü olan kromozomdan ortak metrikleri çıkarır.

    Beklenen arayüz (her iki motor için de uyumlu):
    - chromosome.fitness
    - chromosome.palet_sayisi
    - chromosome.ortalama_doluluk (0-1 arası oran)
    """
    if chromosome is None:
        return None

    utilization = getattr(chromosome, "ortalama_doluluk", None)
    pallets = getattr(chromosome, "palet_sayisi", None)
    fitness = getattr(chromosome, "fitness", None)

    if utilization is None or pallets is None:
        return None

    return {
        "fitness": fitness,
        "utilization": float(utilization),  # 0-1
        "pallets": int(pallets),
    }


def combine_with_single_prepass(
    mix_summary: Optional[Dict[str, Any]],
    palet_cfg: PaletConfig,
    single_pallet_count: int,
    single_used_volume: float,
) -> Dict[str, Any]:
    """Single pre-pass + mix sonucu birleştirerek toplam özet üretir.

    Çıktı sözlüğü en az şu alanları içerir:
        - single_pallet_count
        - mix_pallet_count
        - pallet_count
        - avg_utilization
        - utilization
        - pallets (geriye dönük uyumluluk için pallet_count ile aynı)
    """
    pallet_volume = _get_pallet_volume(palet_cfg)

    if mix_summary is None:
        total_pallets = single_pallet_count
        if total_pallets > 0 and pallet_volume > 0:
            combined_util = single_used_volume / (total_pallets * pallet_volume)
        else:
            combined_util = 0.0

        return {
            "single_pallet_count": single_pallet_count,
            "mix_pallet_count": 0,
            "pallet_count": total_pallets,
            "pallets": total_pallets,
            "avg_utilization": combined_util,
            "utilization": combined_util,
            "fitness": None,
        }

    mix_pallets = int(mix_summary.get("pallets", 0) or 0)
    mix_util = float(mix_summary.get("utilization", 0.0) or 0.0)

    mix_used_volume = mix_util * mix_pallets * pallet_volume

    total_pallets = single_pallet_count + mix_pallets
    if total_pallets > 0 and pallet_volume > 0:
        combined_util = (single_used_volume + mix_used_volume) / (total_pallets * pallet_volume)
    else:
        combined_util = 0.0

    combined = dict(mix_summary)
    combined.update(
        {
            "single_pallet_count": single_pallet_count,
            "mix_pallet_count": mix_pallets,
            "pallet_count": total_pallets,
            "pallets": total_pallets,
            "avg_utilization": combined_util,
            "utilization": combined_util,
        }
    )

    return combined


def compare_algorithms(palet_cfg: PaletConfig, products: List[UrunData]):
    """GA ve DE'yi Django UI'ye benzer single + mix akışıyla karşılaştırır.

    Akış:
        1) Single pallet pre-pass → single_pallet_count, mix_pool, single_used_volume
        2) GA/DE yalnızca mix_pool üzerinde çalışır (varsa)
        3) Sonuçlar combine_with_single_prepass ile birleştirilir
        4) Süre metrikleri single pre-pass + algoritma süresinin toplamıdır
    """
    results: Dict[str, Any] = {}

    # --- GA ---
    ga_products = copy.deepcopy(products)

    ga_pre_start = time.perf_counter()
    ga_single_pallet_count, ga_mix_pool, ga_single_used_volume = run_single_pallet_prepass(
        palet_cfg, ga_products
    )
    ga_pre_time = time.perf_counter() - ga_pre_start

    ga_chrom = None
    ga_hist = None
    ga_algo_time = 0.0
    ga_mix_summary = None

    if ga_mix_pool:
        ga_chrom, ga_hist, ga_algo_time = run_ga_with_defaults(palet_cfg, ga_mix_pool)
        ga_mix_summary = summarize_chromosome(ga_chrom)

    ga_total_time = ga_pre_time + ga_algo_time
    ga_combined_summary = combine_with_single_prepass(
        ga_mix_summary,
        palet_cfg,
        ga_single_pallet_count,
        ga_single_used_volume,
    )

    results["ga"] = {
        "name": "Genetic Algorithm (GA)",
        "chromosome": ga_chrom,
        "history": ga_hist,
        "time": ga_total_time,
        "summary": ga_combined_summary,
        "prepass": {
            "single_pallet_count": ga_single_pallet_count,
            "single_used_volume": ga_single_used_volume,
            "mix_pool_size": len(ga_mix_pool),
            "time": ga_pre_time,
        },
        "mix_summary": ga_mix_summary,
    }

    clear_runtime_state()

    # --- DE ---
    de_products = copy.deepcopy(products)

    de_pre_start = time.perf_counter()
    de_single_pallet_count, de_mix_pool, de_single_used_volume = run_single_pallet_prepass(
        palet_cfg, de_products
    )
    de_pre_time = time.perf_counter() - de_pre_start

    de_chrom = None
    de_hist = None
    de_algo_time = 0.0
    de_mix_summary = None

    if de_mix_pool:
        de_chrom, de_hist, de_algo_time = run_de_with_defaults(palet_cfg, de_mix_pool)
        de_mix_summary = summarize_chromosome(de_chrom)

    de_total_time = de_pre_time + de_algo_time
    de_combined_summary = combine_with_single_prepass(
        de_mix_summary,
        palet_cfg,
        de_single_pallet_count,
        de_single_used_volume,
    )

    results["de"] = {
        "name": "Differential Evolution (DE)",
        "chromosome": de_chrom,
        "history": de_hist,
        "time": de_total_time,
        "summary": de_combined_summary,
        "prepass": {
            "single_pallet_count": de_single_pallet_count,
            "single_used_volume": de_single_used_volume,
            "mix_pool_size": len(de_mix_pool),
            "time": de_pre_time,
        },
        "mix_summary": de_mix_summary,
    }

    clear_runtime_state()

    return results


def render_results(results: Dict[str, Any]):
    """Streamlit arayüzünde karşılaştırma sonuçlarını gösterir."""
    ga = results.get("ga")
    de = results.get("de")

    ga_sum = ga.get("summary") if ga else None
    de_sum = de.get("summary") if de else None

    st.subheader("Karşılaştırma Sonuçları")

    col1, col2 = st.columns(2)

    # GA kartı
    with col1:
        st.markdown("### Genetic Algorithm (GA)")
        if ga_sum:
            st.metric(
                label="Çalışma süresi (sn)",
                value=f"{ga['time']:.3f}",
            )
            st.metric(
                label="Toplam doluluk (%)",
                value=f"{ga_sum['utilization'] * 100:.2f}",
            )
            st.metric(
                label="Kullanılan palet sayısı",
                value=str(ga_sum["pallets"]),
            )
            pre = ga.get("prepass", {})
            st.markdown("---")
            st.markdown("**Palet Dağılımı (GA)**")
            st.write(f"Single Palet: {pre.get('single_pallet_count', 0)}")
            st.write(f"Mix Pool Ürün Sayısı: {pre.get('mix_pool_size', 0)}")
            st.write(f"Mix Palet: {ga_sum.get('mix_pallet_count', 0)}")
        else:
            st.error("GA herhangi bir çözüm üretemedi veya özet metrikler okunamadı.")

    # DE kartı
    with col2:
        st.markdown("### Differential Evolution (DE)")
        if de_sum:
            st.metric(
                label="Çalışma süresi (sn)",
                value=f"{de['time']:.3f}",
            )
            st.metric(
                label="Toplam doluluk (%)",
                value=f"{de_sum['utilization'] * 100:.2f}",
            )
            st.metric(
                label="Kullanılan palet sayısı",
                value=str(de_sum["pallets"]),
            )
            pre = de.get("prepass", {})
            st.markdown("---")
            st.markdown("**Palet Dağılımı (DE)**")
            st.write(f"Single Palet: {pre.get('single_pallet_count', 0)}")
            st.write(f"Mix Pool Ürün Sayısı: {pre.get('mix_pool_size', 0)}")
            st.write(f"Mix Palet: {de_sum.get('mix_pallet_count', 0)}")
        else:
            st.error("DE herhangi bir çözüm üretemedi veya özet metrikler okunamadı.")

    # İsteğe bağlı özet karşılaştırmalar
    if ga_sum and de_sum:
        st.markdown("---")
        st.subheader("Özet Karşılaştırma")

        faster = None
        if ga["time"] < de["time"]:
            faster = "GA"
        elif de["time"] < ga["time"]:
            faster = "DE"

        better_util = None
        if ga_sum["utilization"] > de_sum["utilization"]:
            better_util = "GA"
        elif de_sum["utilization"] > ga_sum["utilization"]:
            better_util = "DE"

        fewer_pallets = None
        if ga_sum["pallets"] < de_sum["pallets"]:
            fewer_pallets = "GA"
        elif de_sum["pallets"] < ga_sum["pallets"]:
            fewer_pallets = "DE"

        cols = st.columns(3)

        with cols[0]:
            st.markdown("**Daha hızlı algoritma**")
            st.write(faster or "Berabere / Belirsiz")

        with cols[1]:
            st.markdown("**Daha yüksek doluluk**")
            st.write(better_util or "Berabere / Belirsiz")

        with cols[2]:
            st.markdown("**Daha az palet kullanan**")
            st.write(fewer_pallets or "Berabere / Belirsiz")


def main():
    st.set_page_config(page_title="GA vs DE Karşılaştırma", layout="wide")

    st.title("GA vs DE Karşılaştırma Aracı")
    st.markdown(
        """
                Bu ekran, mevcut 3D Bin Packing motorundaki **Genetik Algoritma (GA)** ve
                **Differential Evolution (DE)** çözücülerini **aynı JSON girdisi** üzerinde,
                Django arayüzündeki akışa benzer şekilde karşılaştırmak için tasarlanmıştır.

                - Mevcut algoritma modülleri doğrudan kullanılır.
                - Algoritma içi parametre mantığı **değiştirilmez**.
                - JSON veri sadece **bir kez** parse edilir ve algoritmalara kopyaları verilir.
                - Akış: önce single pallet pre-pass ile tek ürün paletleri çıkarılır,
                    kalan ürünler mix pool'a alınır ve GA/DE sadece bu mix pool üzerinde
                    çalıştırılır.
                - Sonuç kartlarında gösterilen süre, single pre-pass + mix optimizasyon
                    süresinin toplamıdır.
        """
    )

    uploaded_file = st.file_uploader("JSON dosyası yükleyin", type=["json"])

    if uploaded_file is not None:
        # Aynı dosya için JSON'u sadece bir kez parse et
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if (
            "parsed_json_id" not in st.session_state
            or st.session_state.get("parsed_json_id") != file_id
        ):
            try:
                palet_cfg, products, json_data = load_and_parse_json(uploaded_file)
                json_summary = build_json_summary(json_data)
            except Exception as e:
                st.error(f"JSON dosyası okunurken/parselenirken hata oluştu: {e}")
                return

            st.session_state["parsed_json_id"] = file_id
            st.session_state["palet_cfg"] = palet_cfg
            st.session_state["products"] = products
            st.session_state["raw_json"] = json_data
            st.session_state["json_summary"] = json_summary

        palet_cfg = st.session_state.get("palet_cfg")
        products = st.session_state.get("products", [])
        json_data = st.session_state.get("raw_json", {})
        json_summary = st.session_state.get("json_summary", {})

        # JSON özet bilgisi
        if palet_cfg is not None and products:
            st.subheader("JSON Özeti")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Konteyner Boyutları (LxWxH)**")
                st.write(f"{palet_cfg.length} x {palet_cfg.width} x {palet_cfg.height} cm")
            with col_b:
                st.markdown("**Max Ağırlık**")
                st.write(f"{palet_cfg.max_weight} kg")
            with col_c:
                st.markdown("**Toplam Paket Sayısı**")
                st.write(json_summary.get("total_packages", 0))

            st.markdown(
                f"Toplam ürün çeşidi: **{json_summary.get('total_product_types', 0)}**"
            )

        if st.button("Karşılaştırmayı Başlat"):
            palet_cfg = st.session_state.get("palet_cfg")
            products = st.session_state.get("products", [])

            if palet_cfg is None or not products:
                st.warning("Önce geçerli bir JSON dosyası yükleyin.")
                return

            with st.spinner("Algoritmalar çalıştırılıyor, lütfen bekleyin..."):
                try:
                    results = compare_algorithms(palet_cfg, products)
                except Exception as e:
                    st.error(f"Algoritmalar çalıştırılırken hata oluştu: {e}")
                    return

            render_results(results)
    else:
        st.info("Lütfen önce bir JSON dosyası yükleyin.")


if __name__ == "__main__":
    main()
