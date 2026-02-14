#!/usr/bin/env python3
"""
3D Bin Packing Optimization - Ana Giriş Noktası
=================================================

Django'dan bağımsız olarak algoritmaları doğrudan çalıştırır.
Test verileriyle hızlı deney yapmak için kullanılır.

Kullanım:
    python main.py                          # Varsayılan test dosyası
    python main.py data/samples/0109.json   # Belirli dosya
    python main.py --algorithm greedy       # Greedy mod
    python main.py --algorithm genetic      # GA mod (varsayılan)
"""

import sys
import os
import json
import argparse
import time

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import PaletConfig, UrunData
from src.utils.parser import parse_json_input, load_json_file
from src.utils.helpers import urun_hacmi, group_products_smart
from src.core.single_pallet import simulate_single_pallet, generate_grid_placement
from src.core.mix_pallet import mix_palet_yerlestirme_main
from src.core.packing import pack_maximal_rectangles
from src.core.genetic_algorithm import run_ga
from src.utils.visualization import render_pallet_3d


def run_optimization_standalone(json_path, algorithm='genetic', output_dir='output'):
    """
    Django olmadan tam optimizasyon çalıştırır.
    
    Args:
        json_path: Girdi JSON dosya yolu
        algorithm: 'genetic' veya 'greedy'
        output_dir: Çıktı klasörü
    """
    print("=" * 70)
    print("  3D BIN PACKING OPTIMIZATION ENGINE")
    print(f"  Algoritma: {algorithm.upper()}")
    print(f"  Girdi: {json_path}")
    print("=" * 70)
    
    # 1. Veriyi yükle
    json_data = load_json_file(json_path)
    palet_cfg, all_products = parse_json_input(json_data)
    
    print(f"\n📦 Konteyner: {palet_cfg.length}×{palet_cfg.width}×{palet_cfg.height} cm")
    print(f"   Max Ağırlık: {palet_cfg.max_weight} kg")
    print(f"   Toplam Ürün: {len(all_products)}")
    
    # Toplam hacim analizi
    total_vol = sum(urun_hacmi(u) for u in all_products)
    theo_min = max(1, int(total_vol / palet_cfg.volume) + 1)
    print(f"   Toplam Hacim: {total_vol:,.0f} cm³")
    print(f"   Teorik Min. Palet: {theo_min}")
    
    start_time = time.time()
    
    # 2. Single Palet Analizi
    print("\n" + "=" * 50)
    print("  AŞAMA 1: SINGLE PALET ANALİZİ")
    print("=" * 50)
    
    groups = group_products_smart(all_products)
    single_pallets = []
    mix_pool = []
    
    for key, group_items in groups.items():
        urun_kodu = key[0]
        total_qty = len(group_items)
        
        sim_result = simulate_single_pallet(group_items, palet_cfg)
        
        if sim_result["can_be_single"]:
            capacity = sim_result["capacity"]
            item_volume = urun_hacmi(group_items[0])
            
            num_full = total_qty // capacity
            remainder = total_qty % capacity
            
            for i in range(num_full):
                palet_items = group_items[i * capacity:(i + 1) * capacity]
                placements = generate_grid_placement(palet_items, palet_cfg)
                single_pallets.append({
                    'type': 'SINGLE',
                    'urun_kodu': urun_kodu,
                    'items': placements,
                    'item_count': len(placements)
                })
            
            # Partial palet (>= %90)
            if remainder > 0:
                partial_fill = (remainder * item_volume) / palet_cfg.volume
                if partial_fill >= 0.90:
                    palet_items = group_items[-remainder:]
                    placements = generate_grid_placement(palet_items, palet_cfg)
                    single_pallets.append({
                        'type': 'SINGLE',
                        'urun_kodu': urun_kodu,
                        'items': placements,
                        'item_count': len(placements)
                    })
                else:
                    mix_pool.extend(group_items[-remainder:])
        else:
            mix_pool.extend(group_items)
    
    print(f"\n  Single Paletler: {len(single_pallets)}")
    print(f"  Mix Havuzuna Kalan: {len(mix_pool)} ürün")
    
    # 3. Mix Palet
    mix_pallets = []
    if mix_pool:
        print("\n" + "=" * 50)
        print("  AŞAMA 2: MIX PALET OPTİMİZASYONU")
        print("=" * 50)
        
        if algorithm == 'genetic':
            best_solution, history = run_ga(
                urunler=mix_pool,
                palet_cfg=palet_cfg,
                population_size=50,
                generations=50,
                mutation_rate=0.3
            )
            
            if best_solution:
                siralanmis = [best_solution.urunler[i] for i in best_solution.sira_gen]
                pallets_data = pack_maximal_rectangles(siralanmis, palet_cfg)
            else:
                pallets_data = pack_maximal_rectangles(mix_pool, palet_cfg)
        else:
            pallets_data = pack_maximal_rectangles(mix_pool, palet_cfg)
        
        for p_data in pallets_data:
            mix_pallets.append({
                'type': 'MIX',
                'items': p_data['items'],
                'item_count': len(p_data['items']),
                'weight': p_data['weight']
            })
    
    elapsed = time.time() - start_time
    
    # 4. Sonuç Raporu
    total_pallets = len(single_pallets) + len(mix_pallets)
    
    print("\n" + "=" * 70)
    print("  SONUÇ RAPORU")
    print("=" * 70)
    print(f"  Toplam Palet: {total_pallets}")
    print(f"    - Single: {len(single_pallets)}")
    print(f"    - Mix:    {len(mix_pallets)}")
    print(f"  Teorik Min:  {theo_min}")
    print(f"  Süre:        {elapsed:.2f} saniye")
    print(f"  Algoritma:   {algorithm.upper()}")
    
    # Doluluk oranları
    all_pallets = single_pallets + mix_pallets
    if all_pallets:
        fill_ratios = []
        for p in all_pallets:
            items = p['items']
            used_vol = sum(i['L'] * i['W'] * i['H'] for i in items)
            fill_ratios.append(used_vol / palet_cfg.volume * 100)
        
        avg_fill = sum(fill_ratios) / len(fill_ratios)
        print(f"  Ort. Doluluk: %{avg_fill:.1f}")
        print(f"  Min Doluluk:  %{min(fill_ratios):.1f}")
        print(f"  Max Doluluk:  %{max(fill_ratios):.1f}")
    
    # 5. Görselleri kaydet
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    
    for idx, palet in enumerate(all_pallets):
        items_for_viz = []
        for item in palet['items']:
            items_for_viz.append({
                'urun_kodu': item['urun'].urun_kodu,
                'x': item['x'], 'y': item['y'], 'z': item['z'],
                'L': item['L'], 'W': item['W'], 'H': item['H']
            })
        
        title = f"Palet {idx + 1} - {palet['type']}\nÜrün: {palet['item_count']}"
        buf = render_pallet_3d(palet_cfg.length, palet_cfg.width, palet_cfg.height,
                               items_for_viz, title=title)
        
        img_path = os.path.join(output_dir, 'images', f'palet_{idx + 1}.png')
        with open(img_path, 'wb') as f:
            f.write(buf.read())
        print(f"  📸 {img_path}")
    
    # JSON rapor
    report = {
        'input_file': json_path,
        'algorithm': algorithm,
        'elapsed_seconds': round(elapsed, 2),
        'container': {
            'length': palet_cfg.length,
            'width': palet_cfg.width,
            'height': palet_cfg.height,
            'max_weight': palet_cfg.max_weight
        },
        'total_products': len(all_products),
        'theoretical_min_pallets': theo_min,
        'result': {
            'total_pallets': total_pallets,
            'single_pallets': len(single_pallets),
            'mix_pallets': len(mix_pallets),
            'avg_fill_ratio': round(avg_fill, 2) if all_pallets else 0
        }
    }
    
    report_path = os.path.join(output_dir, 'reports', 'optimization_result.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📊 {report_path}")
    
    print("\n✅ Optimizasyon tamamlandı.")
    return report


def main():
    parser = argparse.ArgumentParser(
        description='3D Bin Packing Optimization Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py data/samples/0109.json
  python main.py data/samples/0109.json --algorithm greedy
  python main.py data/samples/0109.json --output results/
        """
    )
    parser.add_argument('input', nargs='?', default=None,
                        help='Girdi JSON dosya yolu')
    parser.add_argument('--algorithm', '-a', choices=['genetic', 'greedy'],
                        default='genetic', help='Kullanılacak algoritma (varsayılan: genetic)')
    parser.add_argument('--output', '-o', default='output',
                        help='Çıktı klasörü (varsayılan: output/)')
    
    args = parser.parse_args()
    
    # Girdi dosyası kontrolü
    if args.input is None:
        # data/samples/ altındaki ilk JSON dosyasını bul
        samples_dir = os.path.join(os.path.dirname(__file__), 'data', 'samples')
        if os.path.exists(samples_dir):
            json_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.json')])
            if json_files:
                args.input = os.path.join(samples_dir, json_files[0])
                print(f"Varsayılan dosya: {args.input}")
            else:
                print("HATA: data/samples/ klasöründe JSON dosyası bulunamadı.")
                sys.exit(1)
        else:
            print("HATA: data/samples/ klasörü bulunamadı.")
            print("Kullanım: python main.py <girdi.json>")
            sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"HATA: Dosya bulunamadı: {args.input}")
        sys.exit(1)
    
    run_optimization_standalone(args.input, args.algorithm, args.output)


if __name__ == '__main__':
    main()
