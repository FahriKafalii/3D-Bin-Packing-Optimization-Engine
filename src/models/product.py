"""
Ürün (Kutu) Veri Modeli
========================

3D Bin Packing probleminde kutu/ürün parametrelerini tanımlar.
"""


class UrunData:
    """
    Sistem içinde dolaşacak standart Ürün Objesi.
    
    Attributes:
        id: Ürün benzersiz kimliği
        urun_kodu (str): Ürün kodu
        boy (float): Uzunluk (cm) - X ekseni
        en (float): Genişlik (cm) - Y ekseni
        yukseklik (float): Yükseklik (cm) - Z ekseni
        agirlik (float): Ağırlık (kg)
        quantity (int): Miktar
        is_package (bool): Paket mi, birim mi?
        donus_serbest (bool): Yatay düzlemde döndürülebilir mi?
        mukavemet (float): Üzerine binebilecek maksimum yük (kg)
    """
    
    def __init__(self, urun_id, code, boy, en, yukseklik, agirlik, quantity=1, is_package=False):
        self.id = urun_id
        self.urun_kodu = code
        self.boy = float(boy)
        self.en = float(en)
        self.yukseklik = float(yukseklik)
        self.agirlik = float(agirlik)
        self.quantity = quantity
        self.is_package = is_package
        self.donus_serbest = True
        self.mukavemet = 99999

    def __repr__(self):
        return f"<Urun {self.urun_kodu} ({self.boy}x{self.en}x{self.yukseklik})>"
