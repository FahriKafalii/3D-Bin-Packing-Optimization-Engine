import json
import math
import os
import tempfile
import threading
import time
import uuid
from threading import Thread
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.urls import reverse
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
from .models import Urun, Palet, Optimization, PHASE_RANGES
from .services import (
    single_palet_yerlestirme,
    chromosome_to_palets,
    mix_palet_data_to_django,
    merge_repack_service,
    merge_repack_mix_service,
    palet_gorsellestir,
    ozet_grafikler_olustur,
)
from src.models import PaletConfig, UrunData
from src.core.packing_first_fit import pack_maximal_rectangles_first_fit
from src.utils.visualization import renk_uret


# ====================================================================
# CANCEL REGISTRY — kullanıcı sayfadan ayrılırsa iş iptal edilsin.
# Process-içi set; tek-worker dev sunucusunda yeterli.
# ====================================================================

_CANCEL_LOCK = threading.Lock()
_CANCELLED_OPTS = set()       # opt_id'ler
_CANCELLED_GROUPS = set()     # benchmark group_id'ler


def _cancel_opt(opt_id):
    with _CANCEL_LOCK:
        _CANCELLED_OPTS.add(int(opt_id))


def _cancel_group(group_id):
    if not group_id:
        return
    with _CANCEL_LOCK:
        _CANCELLED_GROUPS.add(str(group_id))


def _is_cancelled(opt_id=None, group_id=None):
    with _CANCEL_LOCK:
        if opt_id is not None and int(opt_id) in _CANCELLED_OPTS:
            return True
        if group_id is not None and str(group_id) in _CANCELLED_GROUPS:
            return True
    return False


class OptimizationCancelled(Exception):
    """Worker'a iptal sinyali taşır."""
    pass


def _check_cancel(opt_id, group_id=None):
    if _is_cancelled(opt_id=opt_id, group_id=group_id):
        raise OptimizationCancelled()


# ====================================================================
# PROGRESS HELPER (thread-safe, no external deps)
# ====================================================================

def _run_greedy_mix(urun_data_listesi, palet_cfg, start_id):
    """Saf greedy mix palet yerleştirme (First-Fit + Maximal Rectangles).
    mix_palet_data_to_django formatında sonuç döndürür.
    """
    pallets = pack_maximal_rectangles_first_fit(urun_data_listesi, palet_cfg)
    result = []
    pid = start_id
    for p in pallets:
        result.append({
            'id': pid,
            'type': 'MIX',
            'quantity': len(p['items']),
            'items': p['items'],
            'weight': p.get('weight', 0),
        })
        pid += 1
    return result


def _estimate_mix_sec(algoritma, urun_sayisi):
    """Mix aşamasının tahmini süresi (saniye). Kaba — ease-out zaten tolerant."""
    n = max(1, int(urun_sayisi))
    if algoritma == 'greedy':
        return max(3.0, 2.0 + n * 0.05)
    if algoritma == 'differential_evolution':
        return max(15.0, 15.0 + n * 0.6)
    # genetic ve diğerleri
    return max(10.0, 10.0 + n * 0.8)


def _phase_progress(durum, completed=False, error=False):
    """
    Zaman tabanlı ease-out ile faza göre ilerleme hesaplar.

    - Her fazın PHASE_RANGES içinde (start, end, label) değeri vardır.
    - Faza girildiğinde yüzde = start; fazın tahmini süresi boyunca
      ease-out (1 - e^(-t/tau)) eğrisiyle (end - 1)'e yaklaşır.
    - Asla end'e ulaşmaz; bir sonraki set_phase() ile yüzde start'a çıkar.
    - completed=True → 100. error → mevcut yüzde korunur (kırmızı bar).
    - Monotonluk garantisi: durum['last_pct'] ile yüzde asla geri gitmez.
    """
    if completed:
        return 100, 'Tamamlandı'

    phase = durum.get('phase')
    last_pct = float(durum.get('last_pct', 0))

    if not phase or phase not in PHASE_RANGES:
        return int(round(last_pct)), durum.get('phase_label', 'Hazırlanıyor')

    start, end, label = PHASE_RANGES[phase]
    start = float(start)
    end   = float(end)
    span  = max(0.0, end - start - 1.0)  # end'e asla dokunmasın

    expected = max(0.5, float(durum.get('phase_expected_sec', 5.0)))
    elapsed  = max(0.0, time.time() - float(durum.get('phase_start', time.time())))
    tau = expected / 3.0  # ~3*tau sonunda %95 doygunluk

    k = 1.0 - math.exp(-elapsed / tau) if tau > 0 else 1.0
    pct = start + span * k

    pct = max(pct, last_pct)  # geri gitme
    pct = max(0.0, min(99.0, pct))
    return int(round(pct)), label


def _normalize_progress(current_step, total_steps, completed=False, durum=None, error=False):
    """current_step/total_steps + faz tabanlı yüzde + label.
    durum verilirse faz bilgisi kullanılır; son yüzde durum['last_pct']'e yazılır.
    """
    total = max(1, int(total_steps))
    cur   = max(0, min(int(current_step), total))
    if durum is None:
        pct = 100 if completed else max(0, min(95, int(round(100 * cur / total))))
        return cur, total, pct
    pct, _label = _phase_progress(durum, completed=completed, error=error)
    return cur, total, pct


def upload_result(request):
    """AJAX ile yüklenen JSON dosyasını işler"""
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'Dosya yüklenemedi.'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    # Dosyanın JSON olduğunu kontrol et
    if not uploaded_file.name.lower().endswith('.json'):
        return JsonResponse({'success': False, 'error': 'Yalnızca JSON dosyaları kabul edilir.'}, status=400)
    
    # Dosyayı geçici olarak kaydet
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    
    with open(temp_file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    # JSON dosyasını valide et
    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            yuklenen_veri = json.load(f)
        
        # Geçici dosyayı sil
        os.remove(temp_file_path)
        
        # Yeni JSON formatını parse et
        urun_verileri = []
        
        # Yeni format kontrolü: {"id": ..., "container": {...}, "details": [...]}
        if isinstance(yuklenen_veri, dict) and 'details' in yuklenen_veri:
            detaylar = yuklenen_veri.get('details', [])
            container_info = yuklenen_veri.get('container', {})
            # JSON üst düzeyindeki id bilgisini palet_id olarak ekle
            try:
                palet_id = yuklenen_veri.get('id')
                if palet_id is not None:
                    container_info['palet_id'] = palet_id
            except Exception:
                pass

            # Container bilgilerini session'a kaydet (ileride kullanmak için)
            request.session['container_info'] = container_info
            
            def to_float(x, default=0.0):
                try:
                    return float(x) if x is not None else default
                except (TypeError, ValueError):
                    return default
            
            # Her bir detail kaydını işle
            for detail in detaylar:
                product = detail.get('product', {})
                package_quantity = detail.get('package_quantity')
                quantity = detail.get('quantity', 0)
                unit_id = detail.get('unit_id', 'ADET')
                
                # Ürün kodunu al
                code = product.get('code', product.get('id', 'UNKNOWN'))
                
                # Paket boyutlarını al
                package_length = to_float(product.get('package_length'))
                package_width = to_float(product.get('package_width'))
                package_height = to_float(product.get('package_height'))
                package_weight = to_float(product.get('package_weight'))
                
                # Birim boyutlarını al
                unit_length = to_float(product.get('unit_length'))
                unit_width = to_float(product.get('unit_width'))
                unit_height = to_float(product.get('unit_height'))
                unit_weight = to_float(product.get('unit_weight'))
                
                # Mukavemet bilgisi
                mukavemet = to_float(product.get('package_max_stack_weight'), default=100000)
                if mukavemet == 0:
                    mukavemet = 100000
                
                # package_quantity null ise: unit (ürün) bazında işle
                if package_quantity is None or package_quantity <= 0:
                    # KG cinsinden ise kaç adet ürün olduğunu hesapla
                    if unit_id == 'KG' and unit_weight > 0:
                        adet_urun = int(quantity / unit_weight)
                    else:
                        adet_urun = int(quantity)
                    
                    # Her bir ürün için ayrı kayıt oluştur
                    for i in range(adet_urun):
                        urun_listesi_item = {
                            'urun_kodu': str(code),
                            'urun_adi': f"{code}",
                            'boy': unit_length,
                            'en': unit_width,
                            'yukseklik': unit_height,
                            'agirlik': unit_weight,
                            'mukavemet': mukavemet,
                            'donus_serbest': True,
                            'istiflenebilir': True,
                            'package_quantity': None,
                            'quantity': to_float(quantity),
                            'unit_length': unit_length,
                            'unit_width': unit_width,
                            'unit_height': unit_height,
                            'unit_weight': unit_weight
                        }
                        urun_verileri.append(urun_listesi_item)
                else:
                    # package_quantity var ise: paket bazında işle
                    for i in range(package_quantity):
                        urun_listesi_item = {
                            'urun_kodu': str(code),
                            'urun_adi': f"{code}",
                            'boy': package_length,
                            'en': package_width,
                            'yukseklik': package_height,
                            'agirlik': package_weight,
                            'mukavemet': mukavemet,
                            'donus_serbest': True,
                            'istiflenebilir': True,
                            'package_quantity': package_quantity,
                            'quantity': to_float(quantity),
                            'unit_length': unit_length,
                            'unit_width': unit_width,
                            'unit_height': unit_height,
                            'unit_weight': unit_weight
                        }
                        urun_verileri.append(urun_listesi_item)
        
        # Eski format kontrolü (geriye dönük uyumluluk)
        elif isinstance(yuklenen_veri, list):
            urun_verileri = yuklenen_veri
        else:
            return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı. Desteklenen format: {"details": [...]}'}, status=400)

        # Verileri doğrula
        if not isinstance(urun_verileri, list) or len(urun_verileri) == 0:
            return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı. Ürün listesi boş veya hatalı.'}, status=400)
        
        # Gerekli alanları kontrol et
        required_fields = ['urun_kodu', 'urun_adi', 'boy', 'en', 'yukseklik', 'agirlik']
        for urun in urun_verileri:
            for field in required_fields:
                if field not in urun:
                    return JsonResponse({'success': False, 'error': f'Eksik alan: {field}'}, status=400)
        
        # Verileri session'a kaydet
        request.session['urun_verileri'] = urun_verileri
        
        # Başarılı sonuç dön
        return JsonResponse({
            'success': True, 
            'message': f'Toplam {len(urun_verileri)} ürün yüklendi.',
            'next_url': reverse('palet_app:urun_listesi')
        })
        
    except json.JSONDecodeError:
        # Geçici dosyayı sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı.'}, status=400)
    except Exception as e:
        # Geçici dosyayı sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JsonResponse({'success': False, 'error': f'Hata: {str(e)}'}, status=400)

# Palet seçim sayfası - KALDIRILDI (Container bilgisi JSON'dan alınıyor)

# Ürün listesi sayfası
def urun_listesi(request):
    """Yüklenen ürünleri listeler"""
    # Verilerin session'da olup olmadığını kontrol et
    if 'urun_verileri' not in request.session:
        return redirect('palet_app:home')
    
    urun_verileri = request.session.get('urun_verileri', [])
    container_info = request.session.get('container_info', {})
    
    # Ürünleri grupla (aynı ürün koduna sahip olanlar)
    urun_gruplari = {}
    for urun in urun_verileri:
        kod = urun['urun_kodu']
        if kod not in urun_gruplari:
            urun_gruplari[kod] = {
                'urun_kodu': kod,
                'urun_adi': urun['urun_adi'],
                'boy': urun['boy'],
                'en': urun['en'],
                'yukseklik': urun['yukseklik'],
                'agirlik': urun['agirlik'],
                'mukavemet': urun.get('mukavemet', 'N/A'),
                'adet': 0,
                'toplam_agirlik': 0,
                'toplam_hacim': 0
            }
        urun_gruplari[kod]['adet'] += 1
        urun_gruplari[kod]['toplam_agirlik'] += urun['agirlik']
        urun_gruplari[kod]['toplam_hacim'] += (urun['boy'] * urun['en'] * urun['yukseklik'])
    
    # Listeye çevir ve sırala
    urun_listesi = sorted(urun_gruplari.values(), key=lambda x: x['urun_kodu'])
    
    context = {
        'urun_listesi': urun_listesi,
        'toplam_urun_cesidi': len(urun_listesi),
        'toplam_paket': len(urun_verileri),
        'container_info': container_info
    }
    
    return render(request, 'palet_app/urun_listesi.html', context)

# Arka planda çalışacak optimizasyon işlemi
def run_optimization(urun_verileri, container_info, optimization_id, algoritma='greedy', ga_mode='balanced', group_id=None):
    """
    Arka planda çalışacak optimizasyon işlemi. Bu fonksiyon bir thread içinde çalışır.
    ga_mode: sadece genetic için kullanılır — 'fast', 'balanced', 'quality'.
    group_id: Benchmark grubu; sayfadan çıkılınca toplu iptal için.
    """
    try:
        _check_cancel(optimization_id, group_id)
        optimization = Optimization.objects.get(id=optimization_id)
        # Tek dogru kaynak: baslatma aninda kaydedilen algoritma (DB)
        algoritma = (optimization.algoritma or 'greedy').strip().lower()
        if algoritma not in ('genetic', 'differential_evolution', 'greedy'):
            algoritma = 'greedy'
        _thread_t0 = time.time()
        # Her algoritma kendi süresini t=0'dan başlasın (sıralı çalıştırmada
        # önceki algoritmaların beklemesi bu süreye yansımasın).
        from django.utils import timezone as _tz
        optimization.baslangic_zamani = _tz.now()
        optimization.save(update_fields=['baslangic_zamani'])
        print(f"\nrun_optimization() basladi (ID: {optimization_id}, Algoritma DB'den: {algoritma}, t0={_thread_t0:.3f})")
        print(f"Optimization objesi bulundu (ID: {optimization_id})")

        # Hangi algoritmanin calistigini net yaz (yanlis baglanti kontrolu icin)
        _check_cancel(optimization_id, group_id)
        optimization.set_phase('baslangic')
        optimization.islem_adimi_ekle(f"[MOTOR] Algoritma: {algoritma.upper()}")
        # Adım 1: Ürünleri veritabanına kaydet
        _check_cancel(optimization_id, group_id)
        optimization.set_phase('urunler')
        optimization.islem_adimi_ekle("Ürün verileri yükleniyor...")
        
        urunler = []
        for veri in urun_verileri:
            urun = Urun(
                urun_kodu=veri["urun_kodu"],
                urun_adi=veri["urun_adi"],
                boy=veri["boy"],
                en=veri["en"],
                yukseklik=veri["yukseklik"],
                agirlik=veri["agirlik"],
                mukavemet=veri.get("mukavemet", 100000),
                donus_serbest=veri.get("donus_serbest", True),
                istiflenebilir=veri.get("istiflenebilir", True)
            )
            urun.save()
            urunler.append(urun)
        
        # Adım 2: Single palet yerleştirme
        _check_cancel(optimization_id, group_id)
        optimization.set_phase('single', expected_sec=max(3.0, len(urunler) * 0.04))
        optimization.islem_adimi_ekle("Single paletler oluşturuluyor...")
        single_paletler, yerlesmemis_urunler = single_palet_yerlestirme(urunler, container_info, optimization)
        
        # Palet konfigürasyonu oluştur (hem genetic hem greedy için)
        palet_cfg = PaletConfig(
            length=container_info['length'],
            width=container_info['width'],
            height=container_info['height'],
            max_weight=container_info['weight']
        )
        
        # Django modellerini UrunData'ya çevir (hem genetic hem greedy için)
        urun_data_listesi = []
        for urun in yerlesmemis_urunler:
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
            urun_data_listesi.append(urun_data)
        
        # Adım 3: Mix palet yerleştirme
        _check_cancel(optimization_id, group_id)
        mix_motor_start = time.time()
        optimization.set_phase('mix', expected_sec=_estimate_mix_sec(algoritma, len(urun_data_listesi)))
        optimization.islem_adimi_ekle(f"[TANI] Mix motoru başlıyor — yerleşmeyen ürün sayısı: {len(urun_data_listesi)}")
        if algoritma == 'genetic':
            from src.core.genetic_algorithm import run_ga
            
            urun_sayisi = len(urun_data_listesi)
            ga_mode = (ga_mode or 'balanced').strip().lower()
            if ga_mode not in ('fast', 'balanced', 'quality'):
                ga_mode = 'balanced'

            if ga_mode == 'fast':
                optimization.islem_adimi_ekle("Genetik Algoritma (Hızlı mod) ile mix paletler olusturuluyor...")
                pop_size = 50 if urun_sayisi > 300 else 40
                generations = 80 if urun_sayisi > 300 else 60
                elitism = 2
                mutation_rate = 0.24
                tournament_k = 3
            elif ga_mode == 'quality':
                optimization.islem_adimi_ekle("Genetik Algoritma (Kaliteli mod) ile mix paletler olusturuluyor...")
                pop_size = 150 if urun_sayisi > 500 else 120
                generations = 280 if urun_sayisi > 500 else 220
                elitism = max(4, min(10, pop_size // 12))
                mutation_rate = 0.16
                tournament_k = 4
            else:
                optimization.islem_adimi_ekle("Genetik Algoritma (Dengeli mod) ile mix paletler olusturuluyor...")
                pop_size = 100 if urun_sayisi > 500 else (85 if urun_sayisi > 200 else 70)
                generations = min(200, 120 + urun_sayisi // 10)
                elitism = max(2, min(5, pop_size // 20))
                mutation_rate = 0.20
                tournament_k = 3

            optimization.islem_adimi_ekle(
                f"Mod: {ga_mode.upper()} | Pop={pop_size}, Nesil={generations}, Elit={elitism}, Mut={mutation_rate}, K={tournament_k}, Ürün={urun_sayisi}"
            )

            best_chromosome, history = run_ga(
                urunler=urun_data_listesi,
                palet_cfg=palet_cfg,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                tournament_k=tournament_k,
                elitism=elitism
            )
            
            if best_chromosome:
                optimization.islem_adimi_ekle(
                    f"En iyi çözüm: Fitness={best_chromosome.fitness:.2f}, "
                    f"Palet={best_chromosome.palet_sayisi}, "
                    f"Doluluk={best_chromosome.ortalama_doluluk:.2%}"
                )
                
                # En iyi kromozomdan paletleri oluştur
                mix_paletler = chromosome_to_palets(
                    best_chromosome, 
                    palet_cfg, 
                    optimization, 
                    len(single_paletler) + 1
                )
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (GA).")
            else:
                optimization.islem_adimi_ekle("GA çözüm üretemedi, Greedy yönteme geçiliyor...")
                mix_palet_data = _run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
                mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")

        elif algoritma == 'differential_evolution':
            from src.core.optimizer_de import optimize_with_de
            
            optimization.islem_adimi_ekle("Differential Evolution (DE) Motoru ile mix paletler olusturuluyor...")
            optimization.islem_adimi_ekle("İleri seviye optimizasyon teknikleri kullanılıyor...")
            
            # Ürün sayısına göre dinamik parametreler
            urun_sayisi = len(urun_data_listesi)
            
            # DE için optimize edilmiş parametreler
            pop_size = 50 if urun_sayisi > 100 else 40
            generations = 100 if urun_sayisi > 100 else 60
            
            optimization.islem_adimi_ekle(
                f"DE Parametreler: Pop={pop_size}, Nesil={generations}, "
                f"Ürün={urun_sayisi}, Fitness Önbellek: Aktif"
            )
            
            # DE motorunu çalıştır
            best_chromosome, history = optimize_with_de(
                urunler=urun_data_listesi,
                palet_cfg=palet_cfg,
                population_size=pop_size,
                generations=generations,
                F=0.8,
                CR_p=0.9
            )
            
            if best_chromosome:
                optimization.islem_adimi_ekle(
                    f"DE En iyi cozum: Fitness={best_chromosome.fitness:.2f}, "
                    f"Palet={best_chromosome.palet_sayisi}, "
                    f"Doluluk={best_chromosome.ortalama_doluluk:.2%}"
                )
                
                # En iyi kromozomdan paletleri oluştur
                mix_paletler = chromosome_to_palets(
                    best_chromosome, 
                    palet_cfg, 
                    optimization, 
                    len(single_paletler) + 1
                )
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (DE).")
            else:
                optimization.islem_adimi_ekle("DE çözüm üretemedi, Greedy yönteme geçiliyor...")
                mix_palet_data = _run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
                mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")

        else:
            optimization.islem_adimi_ekle("Mix paletler oluşturuluyor (Greedy - First-Fit + Maximal Rectangles)...")
            mix_palet_data = _run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
            mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
            optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")
        
        mix_motor_sec = round(time.time() - mix_motor_start, 2)
        optimization.islem_adimi_ekle(
            f"[TANI] Mix motoru bitti ({algoritma.upper()}): {mix_motor_sec} sn, "
            f"{len(mix_paletler)} mix palet üretildi"
        )

        # ── Merge & Repack post-optimizasyon ─────────────────────────────────
        merge_start = time.time()
        if len(mix_paletler) >= 2:
            # Aşama 1 — İteratif BFD (belirlenimsel, hızlı)
            _check_cancel(optimization_id, group_id)
            optimization.set_phase('merge1', expected_sec=max(4.0, len(mix_paletler) * 0.8))
            optimization.islem_adimi_ekle(
                f"[1/2] Merge & Repack BFD basliyor "
                f"({len(mix_paletler)} mix palet → iteratif birleştirme)..."
            )
            mix_paletler, mix_metrics = merge_repack_mix_service(
                mix_paletler,
                palet_cfg,
                optimization,
                baslangic_id=len(single_paletler) + 1,
                urun_data_listesi=urun_data_listesi,
            )
            optimization.islem_adimi_ekle(mix_metrics.summary())

            # Aşama 2 — Random Restart (stokastik, BFD sonrası kalan fırsatı karolar)
            if len(mix_paletler) >= 2:
                _check_cancel(optimization_id, group_id)
                optimization.set_phase('merge2', expected_sec=max(3.0, len(mix_paletler) * 0.5))
                optimization.islem_adimi_ekle(
                    f"[2/2] Merge & Repack Random Restart basliyor "
                    f"({len(mix_paletler)} palet kaldı)..."
                )
                mix_paletler, mr_metrics = merge_repack_service(
                    mix_paletler,
                    palet_cfg,
                    optimization,
                    baslangic_id=len(single_paletler) + 1,
                    urun_data_listesi=urun_data_listesi,
                )
                optimization.islem_adimi_ekle(mr_metrics.summary())

        merge_sec = round(time.time() - merge_start, 2)
        optimization.islem_adimi_ekle(
            f"[TANI] Merge & Repack bitti ({algoritma.upper()}): {merge_sec} sn, "
            f"son {len(mix_paletler)} mix palet"
        )

        # Adım 4: İstatistikleri güncelle (Görselleştirme artık on-the-fly yapılıyor)
        optimization.islem_adimi_ekle("İstatistikler hesaplanıyor...")
        
        # Tüm paletleri birleştir
        tum_paletler = list(single_paletler) + list(mix_paletler)
        
        # Palet istatistiklerini güncelle
        from .models import Palet
        paletler = Palet.objects.filter(optimization=optimization)
        single = paletler.filter(palet_turu='single').count()
        mix = paletler.filter(palet_turu='mix').count()
        optimization.single_palet = single
        optimization.mix_palet = mix
        optimization.toplam_palet = single + mix
        optimization.save()
        
        # Yerleştirilemeyen ürünleri kaydet
        son_yerlesmeyen_urunler = []
        for urun in urunler:
            yerlestirilmis = False
            for palet in tum_paletler:
                urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
                if str(urun.id) in urun_konumlari:
                    yerlestirilmis = True
                    break
            
            if not yerlestirilmis:
                son_yerlesmeyen_urunler.append({
                    'id': urun.id,
                    'urun_kodu': urun.urun_kodu,
                    'urun_adi': urun.urun_adi,
                    'boy': urun.boy,
                    'en': urun.en,
                    'yukseklik': urun.yukseklik,
                    'agirlik': urun.agirlik
                })
        
        optimization.yerlesmemis_urunler = son_yerlesmeyen_urunler
        
        # Tum paletler icin gorselleri olustur
        _check_cancel(optimization_id, group_id)
        optimization.set_phase('gorsel', expected_sec=max(2.0, len(tum_paletler) * 0.25))
        optimization.islem_adimi_ekle("Palet görselleri oluşturuluyor...")
        for palet in tum_paletler:
            _check_cancel(optimization_id, group_id)
            if not palet.gorsel:  # Henüz görsel yoksa
                try:
                    urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
                    urun_ids = [int(id) for id in urun_konumlari.keys()]
                    palet_urunleri = list(Urun.objects.filter(id__in=urun_ids))
                    
                    png_content = palet_gorsellestir(palet, palet_urunleri, save_to_file=True)
                    palet.gorsel.save(f'palet_{palet.palet_id}.png', png_content, save=True)
                    print(f"Palet {palet.palet_id} gorseli olusturuldu")
                except Exception as e:
                    print(f"UYARI: Palet {palet.palet_id} gorseli olusturulamadi: {str(e)}")
        
        # Optimizasyonu tamamla
        optimization.islem_adimi_ekle("Optimizasyon tamamlandı.")
        optimization.tamamla()
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZASYON TAMAMLANDI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization_id}")
        print(f"Toplam Palet: {optimization.toplam_palet}")
        print(f"Single Palet: {optimization.single_palet}")
        print(f"Mix Palet: {optimization.mix_palet}")
        print(f"Yerleşemeyen: {len(son_yerlesmeyen_urunler)}")
        print(f"{'='*60}\n")
        
    except OptimizationCancelled:
        print(f"[IPTAL] Optimization {optimization_id} ({algoritma}) kullanıcı tarafından iptal edildi.")
        try:
            optimization = Optimization.objects.get(id=optimization_id)
            optimization.islem_adimi_ekle("İşlem kullanıcı tarafından iptal edildi.")
            durum = optimization.get_islem_durumu()
            durum['current_step'] = -1
            durum['cancelled'] = True
            optimization.islem_durumu = json.dumps(durum)
            optimization.save()
        except Exception as inner_e:
            print(f"Cancel temizlik hatasi: {inner_e}")
        return

    except Exception as e:
        # Hata durumunda
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"HATA: OPTIMIZASYON HATASI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization_id}")
        print(f"HATA: {str(e)}")
        print(f"DETAY:\n{error_detail}")
        print(f"{'='*60}\n")

        try:
            optimization = Optimization.objects.get(id=optimization_id)
            optimization.islem_adimi_ekle(f"Hata: {str(e)}")
            # Tamamen hatalı olduğunu belirt
            durum = optimization.get_islem_durumu()
            durum['current_step'] = -1  # Hata durumu
            optimization.islem_durumu = json.dumps(durum)
            optimization.save()
        except Exception as inner_e:
            print(f"Inner exception: {str(inner_e)}")

# İşleniyor sayfası
def processing(request):
    """İşlem simülasyonu sayfası"""
    # Verilerin session'da olup olmadığını kontrol et
    if 'urun_verileri' not in request.session:
        print(f"UYARI: processing: Session'da urun_verileri yok!")
        return redirect('palet_app:home')
    
    # Container bilgisi var mı kontrol et
    container_info = request.session.get('container_info')
    if not container_info:
        print(f"UYARI: processing: Session'da container_info yok!")
        return redirect('palet_app:home')
    
    optimization_id = request.session.get('optimization_id')
    print(f"processing sayfasi yuklendi (Optimization ID: {optimization_id})")
    
    return render(request, 'palet_app/processing.html')

# Yerleştirme başlatma API'si
def start_placement(request):
    """Yerleştirme işlemini başlatır"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Yalnızca POST istekleri kabul edilir.'}, status=400)
    
    # Gerekli verileri kontrol et
    if 'urun_verileri' not in request.session:
        return JsonResponse({'success': False, 'error': 'Ürün verileri bulunamadı.'}, status=400)
    
    # Container bilgisini al (JSON'dan gelen)
    container_info = request.session.get('container_info')
    if not container_info:
        return JsonResponse({'success': False, 'error': 'Container bilgisi bulunamadı.'}, status=400)
    
    # Frontend'den gelen algoritma secimi (genetic | differential_evolution | greedy)
    import json as json_module
    algoritma_raw = None
    ga_mode = 'balanced'
    try:
        body = {}
        if request.body:
            body = json_module.loads(request.body)
            algoritma_raw = body.get('algoritma')
        if algoritma_raw is None:
            algoritma_raw = request.POST.get('algoritma')
        if algoritma_raw is not None:
            algoritma = str(algoritma_raw).strip().lower()
        else:
            algoritma = 'genetic'
        if algoritma not in ('genetic', 'differential_evolution', 'greedy'):
            print(f"[start_placement] Gecersiz algoritma '{algoritma_raw}' -> genetic kullaniliyor")
            algoritma = 'genetic'
        if algoritma == 'genetic':
            gm = (body.get('ga_mode') or 'balanced').strip().lower()
            if gm in ('fast', 'balanced', 'quality'):
                ga_mode = gm
        print(f"[start_placement] Algoritma: {algoritma!r}, GA mod: {ga_mode!r}")
    except Exception as e:
        print(f"[start_placement] Body parse hatasi, varsayilan genetic kullaniliyor: {e}")
        algoritma = 'genetic'
    
    # Container bilgilerini al
    container_length = container_info.get('length', 120)
    container_width = container_info.get('width', 100)
    container_height = container_info.get('height', 180)
    container_weight = container_info.get('weight', 1250)
    
    with transaction.atomic():
        # Optimizasyon objesi oluştur (dinamik container bilgileriyle)
        optimization = Optimization.objects.create(
            palet_tipi=None,  # Artık sabit palet tipi kullanmıyoruz
            container_length=container_length,
            container_width=container_width,
            container_height=container_height,
            container_weight=container_weight,
            algoritma=algoritma,  # Algoritmayı kaydet
            islem_durumu=json.dumps({
                "current_step": 0,
                "total_steps": 5,
                "messages": []
            })
        )
        
        # Optimizasyon ID'sini session'a kaydet
        request.session['optimization_id'] = optimization.id
        request.session['algoritma'] = algoritma  # Algoritma bilgisini kaydet
        request.session.modified = True  # Session'ın güncellendiğini işaretle
        
        print(f"\n{'='*60}")
        print(f"YENI OPTIMIZASYON BASLATILDI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization.id}")
        print(f"Algoritma: {algoritma}")
        print(f"Container: {container_length}x{container_width}x{container_height} cm")
        print(f"Max Ağırlık: {container_weight} kg")
        print(f"Ürün Sayısı: {len(request.session['urun_verileri'])}")
        print(f"{'='*60}\n")
        
        # Container bilgilerini dict olarak hazırla
        container_dict = {
            'length': container_length,
            'width': container_width,
            'height': container_height,
            'weight': container_weight
        }
        
        # İşlemi background thread'de başlat
        try:
            thread = Thread(target=run_optimization, args=(request.session['urun_verileri'], container_dict, optimization.id, algoritma, ga_mode))
            thread.daemon = True
            thread.start()
            print(f"Thread baslatildi (ID: {optimization.id})")
        except Exception as e:
            print(f"HATA: Thread baslatma hatasi: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Thread başlatılamadı: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'success': True,
        'message': 'Optimizasyon başlatıldı.',
        'optimization_id': optimization.id,
        'status_url': reverse('palet_app:optimization_status')
    })

# Optimizasyon durumu API
def optimization_status(request):
    """Optimizasyon durumunu döndürür"""
    # Optimizasyon ID'sini al
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        print(f"UYARI: optimization_status: Session'da optimization_id yok!")
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)
    
    try:
        # Optimizasyon durumunu kontrol et
        optimization = Optimization.objects.get(id=optimization_id)
        durum = optimization.get_islem_durumu()
        
        print(f"Status check (ID: {optimization_id}): Completed={optimization.tamamlandi}, Step={durum.get('current_step', 0)}/{durum.get('total_steps', 5)}")
        
        # Eğer işlem tamamlandıysa, analiz sayfasına yönlendir
        if optimization.tamamlandi:
            print(f"Optimization tamamlandi, yonlendirme yapiliyor...")
            return JsonResponse({
                'success': True,
                'completed': True,
                'next_url': reverse('palet_app:analysis')
            })
        
        is_error = durum.get('current_step', 0) == -1
        cur, tot, pct = _normalize_progress(
            durum.get('current_step', 0),
            durum.get('total_steps', 5),
            completed=False,
            durum=durum,
            error=is_error,
        )
        phase_label = PHASE_RANGES.get(durum.get('phase', ''), (0, 0, 'Hazırlanıyor'))[2]
        return JsonResponse({
            'success': True,
            'completed': False,
            'current_step': cur,
            'total_steps': tot,
            'percent': pct,
            'phase_label': phase_label,
            'messages': durum.get('messages', [])
        })
        
    except Optimization.DoesNotExist:
        print(f"Optimization bulunamadi (ID: {optimization_id})")
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)

# Analiz sayfası
def analysis(request):
    """Optimizasyon sonuçlarını gösterir"""
    # Optimizasyon ID'sini al
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        print(f"UYARI: analysis: Session'da optimization_id yok!")
        return redirect('palet_app:home')
    
    print(f"analysis view cagrildi (ID: {optimization_id})")
    
    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)
        
        print(f"   Tamamlandı: {optimization.tamamlandi}")
        print(f"   Toplam Palet: {optimization.toplam_palet}")
        
        if not optimization.tamamlandi:
            print(f"UYARI: Optimizasyon henuz tamamlanmamis, processing'e yonlendiriliyor...")
            return redirect('palet_app:processing')
        
        paletler = Palet.objects.filter(optimization=optimization).order_by('palet_id')
        print(f"   Bulunan palet sayısı: {paletler.count()}")
        
        # Interaktif grafikleri on-the-fly oluştur
        pie_chart_html, bar_chart_html = ozet_grafikler_olustur(optimization)
        
        context = {
            'optimization': optimization,
            'paletler': paletler,
            'single_oran': optimization.single_palet / optimization.toplam_palet * 100 if optimization.toplam_palet > 0 else 0,
            'mix_oran': optimization.mix_palet / optimization.toplam_palet * 100 if optimization.toplam_palet > 0 else 0,
            'yerlesmemis_urunler': optimization.yerlesmemis_urunler,
            'pie_chart_html': pie_chart_html,
            'bar_chart_html': bar_chart_html
        }
        
        print(f"Analysis sayfasi render ediliyor...")
        return render(request, 'palet_app/analysis.html', context)
        
    except Optimization.DoesNotExist:
        print(f"Optimization bulunamadi (ID: {optimization_id})")
        return redirect('palet_app:home')

# Palet detay sayfası
def palet_detail(request, palet_id):
    """Tek bir palet detayını gösterir"""
    # Optimizasyon ID'sini al
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        return redirect('palet_app:home')
    
    try:
        # Optimizasyon objesi
        optimization = get_object_or_404(Optimization, id=optimization_id)
        
        # Eğer optimizasyon henüz tamamlanmadıysa, işleniyor sayfasına yönlendir
        if not optimization.tamamlandi:
            return redirect('palet_app:processing')
        
        # Paleti al
        palet = get_object_or_404(Palet, optimization=optimization, palet_id=palet_id)
        
        # Tüm paletleri al (önceki/sonraki navigasyonu için)
        tum_paletler = Palet.objects.filter(optimization=optimization).order_by('palet_id')
        palet_ids = list(tum_paletler.values_list('palet_id', flat=True))
        
        # Önceki/sonraki palet ID'lerini belirle
        current_index = palet_ids.index(palet_id)
        prev_id = palet_ids[current_index - 1] if current_index > 0 else None
        next_id = palet_ids[current_index + 1] if current_index < len(palet_ids) - 1 else None
        
        # Bu palette hangi ürünlerin olduğunu bul
        urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
        urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)
        
        urun_ids = [int(id) for id in urun_konumlari.keys()]
        urunler = list(Urun.objects.filter(id__in=urun_ids))
        
        # Ürün kodlarına göre renk sözlüğü oluştur (görselleştirme ile aynı mantık)
        urun_renkleri = {}
        for urun in urunler:
            if urun.urun_kodu not in urun_renkleri:
                urun_renkleri[urun.urun_kodu] = renk_uret(urun.urun_kodu)
        
        # Ürün detaylarını hazırla
        urun_detaylari = []
        for urun in urunler:
            konum = urun_konumlari.get(str(urun.id), [0, 0, 0])
            boyut = urun_boyutlari.get(str(urun.id), [0, 0, 0])
            
            # Liste ise tuple'a dönüştür
            if isinstance(konum, list):
                konum = tuple(konum)
            if isinstance(boyut, list):
                boyut = tuple(boyut)
            
            # Renk bilgisini al (RGB 0-1 aralığında)
            renk_rgb = urun_renkleri.get(urun.urun_kodu, (0.5, 0.5, 0.5))
            # RGB'yi 0-255 aralığına çevir
            renk_rgb_255 = (int(renk_rgb[0] * 255), int(renk_rgb[1] * 255), int(renk_rgb[2] * 255))
                
            urun_detaylari.append({
                'urun': urun,
                'konum': konum,
                'boyut': boyut,
                'renk_rgb': renk_rgb_255
            })
        
        context = {
            'palet': palet,
            'urun_detaylari': urun_detaylari,
            'prev_id': prev_id,
            'next_id': next_id,
            'total_palets': len(palet_ids),
        }
        
        return render(request, 'palet_app/palet_detail.html', context)
        
    except Exception as e:
        return HttpResponseBadRequest(f"Hata: {str(e)}")


def palet_3d_data(request, palet_id):
    """3D görselleştirme için palet verisini JSON formatında döndürür"""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        return JsonResponse({'error': 'Optimizasyon bulunamadı'}, status=400)
    
    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)
        palet = get_object_or_404(Palet, optimization=optimization, palet_id=palet_id)
        
        # Palet boyutları
        palet_data = {
            'palet_id': palet.palet_id,
            'boy': palet.boy,
            'en': palet.en,
            'yukseklik': palet.max_yukseklik,
            'doluluk': palet.doluluk_orani(),
            'agirlik': float(palet.toplam_agirlik),
            'items': []
        }
        
        # Ürünleri al
        urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
        urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)
        urun_ids = [int(id) for id in urun_konumlari.keys()]
        urunler = {urun.id: urun for urun in Urun.objects.filter(id__in=urun_ids)}
        
        # Her ürün için veri hazırla
        for urun_id_str, konum in urun_konumlari.items():
            urun_id = int(urun_id_str)
            if urun_id not in urunler:
                continue
                
            urun = urunler[urun_id]
            boyut = urun_boyutlari.get(urun_id_str, [0, 0, 0])
            
            # Renk oluştur (RGB 0-1 aralığı)
            renk_rgb = renk_uret(urun.urun_kodu)
            
            palet_data['items'].append({
                'urun_kodu': urun.urun_kodu,
                'urun_adi': urun.urun_adi,
                'x': konum[0],
                'y': konum[1],
                'z': konum[2],
                'boy': boyut[0],
                'en': boyut[1],
                'yukseklik': boyut[2],
                'agirlik': float(urun.agirlik),
                'renk': {
                    'r': renk_rgb[0],
                    'g': renk_rgb[1],
                    'b': renk_rgb[2]
                }
            })
        
        return JsonResponse(palet_data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



# Ana sayfa
def home_view(request):
    return render(request, 'palet_app/home.html')  # Ana sayfa şablonunu render et


# ====================================================================
# BENCHMARK (TOPLU TEST) — 3 algoritmayı paralel çalıştır ve karşılaştır
# ====================================================================

BENCHMARK_ALGORITHMS = [
    {'key': 'greedy', 'label': 'Greedy', 'ga_mode': None},
    {'key': 'genetic', 'label': 'Genetik Algoritma', 'ga_mode': 'balanced'},
    {'key': 'differential_evolution', 'label': 'Differential Evolution', 'ga_mode': None},
]


def start_benchmark(request):
    """3 algoritmayı aynı ürün verisi üzerinde paralel çalıştırır."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Yalnızca POST istekleri kabul edilir.'}, status=400)

    if 'urun_verileri' not in request.session:
        return JsonResponse({'success': False, 'error': 'Ürün verileri bulunamadı.'}, status=400)

    container_info = request.session.get('container_info')
    if not container_info:
        return JsonResponse({'success': False, 'error': 'Container bilgisi bulunamadı.'}, status=400)

    container_length = container_info.get('length', 120)
    container_width = container_info.get('width', 100)
    container_height = container_info.get('height', 180)
    container_weight = container_info.get('weight', 1250)

    group_id = str(uuid.uuid4())
    created_ids = []

    with transaction.atomic():
        for algo in BENCHMARK_ALGORITHMS:
            optimization = Optimization.objects.create(
                palet_tipi=None,
                container_length=container_length,
                container_width=container_width,
                container_height=container_height,
                container_weight=container_weight,
                algoritma=algo['key'],
                benchmark_group_id=group_id,
                islem_durumu=json.dumps({
                    "current_step": 0,
                    "total_steps": 5,
                    "messages": []
                })
            )
            created_ids.append((optimization.id, algo['key'], algo['ga_mode']))

    request.session['benchmark_group_id'] = group_id
    request.session.modified = True

    print(f"\n{'='*60}\nBENCHMARK BASLATILDI (Group: {group_id})\n{'='*60}")

    container_dict = {
        'length': container_length,
        'width': container_width,
        'height': container_height,
        'weight': container_weight,
    }

    # Tek thread içinde SIRAYLA çalıştır — paralel DB kilitlenmesini ve
    # yanıltıcı aynı-süre sonuçlarını engeller. Her algoritma kendi 0→100'ünü
    # UI'da tamamlayıp bir sonrakine devreder.
    urun_verileri_snapshot = list(request.session['urun_verileri'])

    def _run_benchmark_sequence():
        for opt_id, algo_key, ga_mode in created_ids:
            if _is_cancelled(group_id=group_id):
                print(f"[Benchmark] Grup iptal edildi, kalanlar atlaniyor.")
                # Kalan (başlamamış) optimizasyonları da iptal işaretle
                try:
                    remaining = Optimization.objects.filter(
                        benchmark_group_id=group_id, tamamlandi=False
                    )
                    for r in remaining:
                        d = r.get_islem_durumu()
                        if d.get('current_step', 0) != -1:
                            r.islem_adimi_ekle("İşlem kullanıcı tarafından iptal edildi.")
                            d = r.get_islem_durumu()
                            d['current_step'] = -1
                            d['cancelled'] = True
                            r.islem_durumu = json.dumps(d)
                            r.save()
                except Exception as _e:
                    print(f"Iptal temizlik hatasi: {_e}")
                break
            print(f"[Benchmark] BASLIYOR: {algo_key} (ID: {opt_id})")
            try:
                run_optimization(
                    urun_verileri_snapshot,
                    container_dict,
                    opt_id,
                    algo_key,
                    ga_mode or 'balanced',
                    group_id,
                )
                print(f"[Benchmark] BITTI: {algo_key} (ID: {opt_id})")
            except Exception as e:
                print(f"HATA: Benchmark ({algo_key}): {e}")

    try:
        t = Thread(target=_run_benchmark_sequence)
        t.daemon = True
        t.start()
        print(f"[Benchmark] Sirali dispatcher thread baslatildi.")
    except Exception as e:
        print(f"HATA: Benchmark dispatcher baslatma: {e}")

    return JsonResponse({
        'success': True,
        'benchmark_group_id': group_id,
        'status_url': reverse('palet_app:benchmark_status'),
        'processing_url': reverse('palet_app:benchmark_processing'),
    })


def benchmark_processing(request):
    """Benchmark ilerleme sayfası — 3 algoritma progress bar'larını gösterir."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return redirect('palet_app:home')

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')
    if not optimizations.exists():
        return redirect('palet_app:home')

    return render(request, 'palet_app/benchmark_processing.html', {
        'benchmark_group_id': group_id,
        'optimizations': optimizations,
    })


def benchmark_status(request):
    """Grup içindeki tüm optimizasyonların durumunu tek JSON'da döndürür."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return JsonResponse({'success': False, 'error': 'Benchmark grubu bulunamadı.'}, status=400)

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')

    items = []
    all_done = True
    any_error = False
    for opt in optimizations:
        durum = opt.get_islem_durumu()
        is_error = durum.get('current_step', 0) == -1
        cur, tot, pct = _normalize_progress(
            durum.get('current_step', 0),
            durum.get('total_steps', 5),
            completed=opt.tamamlandi,
            durum=durum,
            error=is_error,
        )
        if is_error:
            any_error = True
        if not opt.tamamlandi and not is_error:
            all_done = False

        messages = durum.get('messages', []) or []
        elapsed = None
        if opt.bitis_zamani and opt.baslangic_zamani:
            elapsed = round((opt.bitis_zamani - opt.baslangic_zamani).total_seconds(), 2)

        items.append({
            'id': opt.id,
            'algoritma': opt.algoritma,
            'completed': opt.tamamlandi,
            'error': is_error,
            'current_step': cur,
            'total_steps': tot,
            'percent': pct,
            'last_message': messages[-1] if messages else '',
            'messages': messages,
            'elapsed': elapsed,
        })

    return JsonResponse({
        'success': True,
        'all_done': all_done,
        'any_error': any_error,
        'items': items,
        'result_url': reverse('palet_app:benchmark_result'),
    })


def benchmark_result(request):
    """3 algoritmanın sonuçlarını yan yana karşılaştırır."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return redirect('palet_app:home')

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')
    if not optimizations.exists():
        return redirect('palet_app:home')

    palet_cfg_dims = None
    results = []
    for opt in optimizations:
        paletler = Palet.objects.filter(optimization=opt)

        # Doluluk & toplam hacim hesabı
        fill_ratios = []
        total_used_vol = 0.0
        container_vol = max(1.0, (opt.container_length or 1) * (opt.container_width or 1) * (opt.container_height or 1))

        for p in paletler:
            boyutlar = p.json_to_dict(p.urun_boyutlari)
            used = 0.0
            for _id, dims in boyutlar.items():
                if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                    used += float(dims[0]) * float(dims[1]) * float(dims[2])
            total_used_vol += used
            fill_ratios.append(used / container_vol * 100.0)

        avg_fill = (sum(fill_ratios) / len(fill_ratios)) if fill_ratios else 0.0
        elapsed = None
        if opt.bitis_zamani and opt.baslangic_zamani:
            elapsed = (opt.bitis_zamani - opt.baslangic_zamani).total_seconds()

        if palet_cfg_dims is None:
            palet_cfg_dims = {
                'length': opt.container_length,
                'width': opt.container_width,
                'height': opt.container_height,
                'weight': opt.container_weight,
            }

        results.append({
            'optimization': opt,
            'algoritma': opt.algoritma,
            'algoritma_label': _algoritma_label(opt.algoritma),
            'completed': opt.tamamlandi,
            'toplam_palet': opt.toplam_palet,
            'single_palet': opt.single_palet,
            'mix_palet': opt.mix_palet,
            'avg_fill': round(avg_fill, 2),
            'elapsed': round(elapsed, 2) if elapsed is not None else None,
            'yerlesmemis_sayi': len(opt.yerlesmemis_urunler or []),
        })

    # En iyi sonucu işaretle: önce palet sayısı (düşük iyi), sonra doluluk (yüksek iyi)
    completed_results = [r for r in results if r['completed'] and r['toplam_palet'] > 0]
    best_id = None
    if completed_results:
        best = min(completed_results, key=lambda r: (r['toplam_palet'], -r['avg_fill']))
        best_id = best['optimization'].id
    for r in results:
        r['is_best'] = (r['optimization'].id == best_id)

    return render(request, 'palet_app/benchmark_result.html', {
        'results': results,
        'container': palet_cfg_dims,
        'benchmark_group_id': group_id,
    })


def benchmark_select(request, optimization_id):
    """Benchmark sonuçlarından birini seç → mevcut analysis sayfasına yönlendir."""
    group_id = request.session.get('benchmark_group_id')
    try:
        opt = Optimization.objects.get(id=optimization_id, benchmark_group_id=group_id)
    except Optimization.DoesNotExist:
        return redirect('palet_app:home')

    request.session['optimization_id'] = opt.id
    request.session['algoritma'] = opt.algoritma
    request.session.modified = True
    return redirect('palet_app:analysis')


def _algoritma_label(key):
    return {
        'greedy': 'Greedy',
        'genetic': 'Genetik Algoritma',
        'differential_evolution': 'Differential Evolution',
    }.get(key, key)


# ====================================================================
# CANCEL ENDPOINTS — kullanıcı sayfadan ayrılırsa iptal
# ====================================================================

@csrf_exempt
def cancel_optimization(request):
    """Tekil optimizasyon iptali (processing sayfasından ayrılma)."""
    opt_id = request.session.get('optimization_id')
    if opt_id:
        _cancel_opt(opt_id)
        print(f"[IPTAL] Tekil optimization iptali istendi: {opt_id}")
    return JsonResponse({'success': True})


@csrf_exempt
def cancel_benchmark(request):
    """Benchmark grubu iptali (benchmark sayfasından ayrılma)."""
    group_id = request.session.get('benchmark_group_id')
    if group_id:
        _cancel_group(group_id)
        print(f"[IPTAL] Benchmark grup iptali istendi: {group_id}")
    return JsonResponse({'success': True})
