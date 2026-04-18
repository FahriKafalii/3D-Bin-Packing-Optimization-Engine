from django.db import models
import json
import time
from django.utils import timezone


PHASE_RANGES = {
    'baslangic': (0, 5,   'Başlatılıyor'),
    'urunler':   (5, 10,  'Ürün verileri yükleniyor'),
    'single':    (10, 25, 'Single paletler oluşturuluyor'),
    'mix':       (25, 75, 'Mix palet optimizasyonu'),
    'merge1':    (75, 85, 'Merge & Repack (BFD)'),
    'merge2':    (85, 92, 'Merge & Repack (Random Restart)'),
    'gorsel':    (92, 98, 'Görseller hazırlanıyor'),
}


class Optimization(models.Model):
    """Optimizasyon islemi; container bilgisi JSON'dan (container_*) gelir."""

    palet_tipi = models.PositiveSmallIntegerField(null=True, blank=True, verbose_name="Palet Tipi (kullanilmiyor)")

    container_length = models.FloatField(null=True, blank=True, verbose_name="Container Uzunluk (cm)")
    container_width = models.FloatField(null=True, blank=True, verbose_name="Container Genişlik (cm)")
    container_height = models.FloatField(null=True, blank=True, verbose_name="Container Yükseklik (cm)")
    container_weight = models.FloatField(null=True, blank=True, verbose_name="Container Max Ağırlık (kg)")

    baslangic_zamani = models.DateTimeField(auto_now_add=True, verbose_name="Başlangıç Zamanı")
    bitis_zamani = models.DateTimeField(null=True, blank=True, verbose_name="Bitiş Zamanı")
    tamamlandi = models.BooleanField(default=False, verbose_name="Tamamlandı mı?")

    toplam_palet = models.PositiveIntegerField(default=0, verbose_name="Toplam Palet Sayısı")
    single_palet = models.PositiveIntegerField(default=0, verbose_name="Single Palet Sayısı")
    mix_palet = models.PositiveIntegerField(default=0, verbose_name="Mix Palet Sayısı")

    islem_durumu = models.TextField(default='{"current_step": 0, "total_steps": 5, "messages": []}', verbose_name="İşlem Durumu")
    yerlesmemis_urunler = models.JSONField(default=list, verbose_name="Yerleştirilemeyen Ürünler")
    algoritma = models.CharField(max_length=20, default='greedy', verbose_name="Algoritma")
    benchmark_group_id = models.CharField(max_length=36, null=True, blank=True, db_index=True, verbose_name="Benchmark Grup ID")

    class Meta:
        verbose_name = "Optimizasyon"
        verbose_name_plural = "Optimizasyonlar"
        ordering = ['-baslangic_zamani']

    def __str__(self):
        return f"Optimizasyon #{self.id} ({self.algoritma})"

    def tamamla(self):
        self.bitis_zamani = timezone.now()
        self.tamamlandi = True
        self.save()

    def islem_adimi_ekle(self, mesaj):
        durum = json.loads(self.islem_durumu)
        durum['messages'].append(mesaj)
        durum['current_step'] += 1
        total = max(1, durum.get('total_steps', 5))
        durum['current_step'] = min(durum['current_step'], total)
        self.islem_durumu = json.dumps(durum)
        self.save()

    def set_phase(self, phase_name, expected_sec=None):
        """Bar yüzdesi için şu anki aşamayı işaretler.
        expected_sec: o aşamanın tahmini süresi (bar ease-out ile bu süre boyunca
        aralığın sonuna asimptotik yaklaşır). None ise varsayılan kullanılır.
        """
        if phase_name not in PHASE_RANGES:
            return
        durum = json.loads(self.islem_durumu)
        durum['phase'] = phase_name
        durum['phase_start'] = time.time()
        durum['phase_expected_sec'] = float(expected_sec) if expected_sec else _default_expected_sec(phase_name)
        self.islem_durumu = json.dumps(durum)
        self.save()

    def get_islem_durumu(self):
        return json.loads(self.islem_durumu)


def _default_expected_sec(phase_name):
    return {
        'baslangic': 1.0,
        'urunler':   2.0,
        'single':    4.0,
        'mix':       60.0,
        'merge1':    8.0,
        'merge2':    6.0,
        'gorsel':    5.0,
    }.get(phase_name, 5.0)
