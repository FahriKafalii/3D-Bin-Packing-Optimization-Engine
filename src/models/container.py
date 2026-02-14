"""
Konteyner / Palet Konfigürasyon Modeli
=======================================

3D Bin Packing probleminde konteyner (palet) parametrelerini tanımlar.
"""


class PaletConfig:
    """
    Palet parametreleri.
    
    Attributes:
        length (float): Palet uzunluğu (cm) - X ekseni
        width (float):  Palet genişliği (cm) - Y ekseni
        height (float): Palet yüksekliği (cm) - Z ekseni
        max_weight (float): Maksimum taşıma kapasitesi (kg)
    """
    
    def __init__(self, length, width, height, max_weight):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.max_weight = float(max_weight)

    @property
    def volume(self) -> float:
        """Palet toplam hacmi (cm³)."""
        return self.length * self.width * self.height

    def __repr__(self):
        return (f"PaletConfig({self.length}x{self.width}x{self.height}, "
                f"max_weight={self.max_weight})")
