"""
Genetik Algoritma - Kromozom Temsili
======================================

NP-Hard 3D Bin Packing problemi için GA bireyini temsil eder.

Kromozom yapısı:
    - sira_gen: Ürünlerin yerleştirilme sırası (permütasyon)
    
Auto-Orientation: Motor, dönüşleri otomatik olarak yönetir.
GA yalnızca ürün sırasını (sequence) optimize eder.
"""

import random


class Chromosome:
    """
    GA bireyini temsil eder - SEQUENCE-ONLY MODE (Auto-Orientation).

    CRITICAL: Motor now handles rotations automatically.
    GA only optimizes the ORDER (sequence) of items.
    
    Attributes:
        urunler: Ürün listesi referansı
        n (int): Ürün sayısı
        sira_gen (list[int]): Yerleştirme sırası permütasyonu
        fitness (float): Fitness değeri
        palet_sayisi (int): Kullanılan palet sayısı
        ortalama_doluluk (float): Ortalama palet doluluk oranı
        yerlesmemis_urun_sayisi (int): Yerleştirilemeyen ürün adedi
    """

    def __init__(self, urunler, sira_gen=None):
        self.urunler = urunler
        self.n = len(urunler)

        # SIRA GENİ (PERMÜTASYON) - GA's only focus
        if sira_gen is None:
            self.sira_gen = list(range(self.n))
            random.shuffle(self.sira_gen)
        else:
            self.sira_gen = list(sira_gen)

        # FİTNESS BİLGİLERİ
        self.fitness = 0.0
        self.palet_sayisi = 0
        self.ortalama_doluluk = 0.0
        self.yerlesmemis_urun_sayisi = 0

    def copy(self):
        """Kromozomun tam bir kopyasını üretir."""
        yeni = Chromosome(
            urunler=self.urunler,
            sira_gen=self.sira_gen.copy()
        )
        yeni.fitness = self.fitness
        yeni.palet_sayisi = self.palet_sayisi
        yeni.ortalama_doluluk = self.ortalama_doluluk
        yeni.yerlesmemis_urun_sayisi = self.yerlesmemis_urun_sayisi
        return yeni

    def __repr__(self):
        _fitness  = self.fitness          if self.fitness          is not None else 0.0
        _palet    = self.palet_sayisi     if self.palet_sayisi     is not None else 0
        _doluluk  = self.ortalama_doluluk if self.ortalama_doluluk is not None else 0.0
        _unplaced = self.yerlesmemis_urun_sayisi if self.yerlesmemis_urun_sayisi is not None else 0
        return (
            f"Chromosome("
            f"n={self.n}, "
            f"fitness={_fitness:.4f}, "
            f"palet_sayisi={_palet}, "
            f"doluluk={_doluluk:.2%}, "
            f"unplaced={_unplaced}"
            f")"
        )
