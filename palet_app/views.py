import json
import os
import tempfile
from threading import Thread
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.urls import reverse
from django.db import transaction
from .models import Urun, Palet, Optimization
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
from src.core.mix_pallet import mix_palet_yerlestirme_main as mix_palet_yerlestirme
from src.utils.visualization import renk_uret


# ====================================================================
# PROGRESS HELPER (thread-safe, no external deps)
# ====================================================================

def _normalize_progress(current_step, total_steps, completed=False):
    """
    Returns safe (current_step, total_steps, percent) tuple.
    - total_steps is always >= 1
    - current_step is always in [0, total_steps]
    - percent is always in [0, 100]
    - If completed=True, percent is forced to 100
    """
    total = max(1, int(total_steps))
    cur   = max(0, min(int(current_step), total))
    if completed:
        return total, total, 100
    pct = int(round(100 * cur / total))
    pct = max(0, min(100, pct))
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
def run_optimization(urun_verileri, container_info, optimization_id, algoritma='greedy'):
    """
    Arka planda çalışacak optimizasyon işlemi. Bu fonksiyon bir thread içinde çalışır.
    
    Args:
        urun_verileri: Ürün verileri listesi
        container_info: Container bilgileri dict (length, width, height, weight)
        optimization_id: Optimizasyon ID'si
        algoritma: 'greedy', 'genetic' veya 'differential_evolution'
    """
    print(f"\nrun_optimization() basladi (ID: {optimization_id}, Algoritma: {algoritma})")
    
    try:
        # Optimizasyon objesi
        optimization = Optimization.objects.get(id=optimization_id)
        print(f"Optimization objesi bulundu (ID: {optimization_id})")
        
        # Hangi algoritmanin calistigini net yaz (yanlis baglanti kontrolu icin)
        optimization.islem_adimi_ekle(f"[MOTOR] Algoritma: {algoritma.upper()}")
        # Adım 1: Ürünleri veritabanına kaydet
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
        if algoritma == 'genetic':
            from src.core.genetic_algorithm import run_ga
            
            optimization.islem_adimi_ekle("Genetik Algoritma Motoru ile mix paletler olusturuluyor...")
            optimization.islem_adimi_ekle("Bu işlem ürün sayısına göre 1-3 dakika sürebilir...")
            
            # Ürün sayısına göre dinamik parametreler
            urun_sayisi = len(urun_data_listesi)
            
            # Optimize edilmiş parametreler
            pop_size = 100 if urun_sayisi > 500 else 80
            generations = 200 if urun_sayisi > 500 else 150
            
            optimization.islem_adimi_ekle(f"Parametreler: Pop={pop_size}, Nesil={generations}, Ürün={urun_sayisi}")
            
            # GA motorunu çalıştır
            best_chromosome, history = run_ga(
                urunler=urun_data_listesi,
                palet_cfg=palet_cfg,
                population_size=pop_size,
                generations=generations,
                mutation_rate=0.2,
                tournament_k=4,
                elitism=3
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
                mix_palet_data = mix_palet_yerlestirme(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
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
                mix_palet_data = mix_palet_yerlestirme(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
                mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")
        
        else:
            optimization.islem_adimi_ekle("Mix paletler oluşturuluyor (Greedy)...")
            mix_palet_data = mix_palet_yerlestirme(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
            mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
            optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu.")
        
        # ── Merge & Repack post-optimizasyon ─────────────────────────────────
        if len(mix_paletler) >= 2:
            # Aşama 1 — İteratif BFD (belirlenimsel, hızlı)
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
        optimization.islem_adimi_ekle("Palet görselleri oluşturuluyor...")
        for palet in tum_paletler:
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
    try:
        body = json_module.loads(request.body) if request.body else {}
        algoritma = body.get('algoritma', 'genetic')
        if algoritma not in ('genetic', 'differential_evolution', 'greedy'):
            algoritma = 'genetic'
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
            thread = Thread(target=run_optimization, args=(request.session['urun_verileri'], container_dict, optimization.id, algoritma))
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
        
        cur, tot, pct = _normalize_progress(
            durum.get('current_step', 0),
            durum.get('total_steps', 5),
            completed=False
        )
        return JsonResponse({
            'success': True,
            'completed': False,
            'current_step': cur,
            'total_steps': tot,
            'percent': pct,
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
        # Optimizasyon objesi
        optimization = get_object_or_404(Optimization, id=optimization_id)
        
        print(f"   Tamamlandı: {optimization.tamamlandi}")
        print(f"   Toplam Palet: {optimization.toplam_palet}")
        
        # Eğer optimizasyon henüz tamamlanmadıysa, işleniyor sayfasına yönlendir
        if not optimization.tamamlandi:
            print(f"UYARI: Optimizasyon henuz tamamlanmamis, processing'e yonlendiriliyor...")
            return redirect('palet_app:processing')
        
        # Paletleri al
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
