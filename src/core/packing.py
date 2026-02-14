"""
3D Maximal Rectangles Packing Engine
======================================

Endüstriyel seviye 3D kutu yerleştirme algoritmaları.

Algoritmalar:
    - Maximal Rectangles (Ana motor) - Auto-Orientation destekli
    - Shelf-Based Packing (Legacy destek)

Referanslar:
    - Jylänki, J. "A Thousand Ways to Pack the Bin" (2010)
    - Huang, E. & Korf, R. "Optimal Rectangle Packing" (2013)
"""

from ..utils.helpers import possible_orientations_for


# ====================================================================
# VERİ YAPILARI
# ====================================================================

class FreeRectangle:
    """
    Boş dikdörtgen alanı temsil eder (Maximal Rectangles için).
    
    Attributes:
        x, y, z: Sol-alt-ön köşe koordinatları
        length, width, height: Boş alanın boyutları
        volume: Boş alanın hacmi
    """
    
    def __init__(self, x, y, z, length, width, height):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height
    
    def can_fit(self, item_l, item_w, item_h):
        """Ürün bu alana sığar mı?"""
        return (self.length >= item_l and 
                self.width >= item_w and 
                self.height >= item_h)
    
    def __repr__(self):
        return f"Rect({self.x},{self.y},{self.z} | {self.length}×{self.width}×{self.height})"


# ====================================================================
# 3D MAXIMAL RECTANGLES ALGORİTMASI
# ====================================================================

def intersects_3d(rect, placed_x, placed_y, placed_z, placed_l, placed_w, placed_h):
    """
    Boş dikdörtgenin yerleştirilen kutuyla 3D kesişimini kontrol eder.
    
    Returns:
        bool: Kesişim varsa True
    """
    return not (
        rect.x >= placed_x + placed_l or
        placed_x >= rect.x + rect.length or
        rect.y >= placed_y + placed_w or
        placed_y >= rect.y + rect.width or
        rect.z >= placed_z + placed_h or
        placed_z >= rect.z + rect.height
    )


def split_rectangle_maximal(rect, placed_x, placed_y, placed_z, placed_l, placed_w, placed_h):
    """
    TRUE 3D MAXIMAL RECTANGLES SPLITTING.
    
    Bir kutu yerleştirildiğinde kesişen boş dikdörtgeni en fazla 6 yeni
    alt-dikdörtgene böler:
    - Sol, Sağ (X ekseni)
    - Ön, Arka (Y ekseni)
    - Alt, Üst (Z ekseni)
    
    Bu ÖRTÜŞEN dikdörtgenler oluşturur - Maximal Rectangles'ın temel özelliği.
    
    Returns:
        list[FreeRectangle]: Yeni boş dikdörtgenler (0-6 arası)
    """
    new_rects = []
    
    # LEFT: Yerleştirilen kutunun solundaki alan
    if rect.x < placed_x:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            placed_x - rect.x,
            rect.width,
            rect.height
        ))
    
    # RIGHT: Yerleştirilen kutunun sağındaki alan
    if placed_x + placed_l < rect.x + rect.length:
        new_rects.append(FreeRectangle(
            placed_x + placed_l, rect.y, rect.z,
            (rect.x + rect.length) - (placed_x + placed_l),
            rect.width,
            rect.height
        ))
    
    # FRONT: Yerleştirilen kutunun önündeki alan
    if rect.y < placed_y:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            rect.length,
            placed_y - rect.y,
            rect.height
        ))
    
    # BACK: Yerleştirilen kutunun arkasındaki alan
    if placed_y + placed_w < rect.y + rect.width:
        new_rects.append(FreeRectangle(
            rect.x, placed_y + placed_w, rect.z,
            rect.length,
            (rect.y + rect.width) - (placed_y + placed_w),
            rect.height
        ))
    
    # BOTTOM: Yerleştirilen kutunun altındaki alan
    if rect.z < placed_z:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            rect.length,
            rect.width,
            placed_z - rect.z
        ))
    
    # TOP: Yerleştirilen kutunun üstündeki alan
    if placed_z + placed_h < rect.z + rect.height:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, placed_z + placed_h,
            rect.length,
            rect.width,
            (rect.z + rect.height) - (placed_z + placed_h)
        ))
    
    return new_rects


def find_best_rectangle(free_rects, item_l, item_w, item_h):
    """
    Best Short Side Fit (BSSF) Heuristic.
    
    Yerleştirme sonrası minimum kısa kenar artığı oluşturacak
    boş dikdörtgeni seçer. İnce, kullanılamaz boşlukları engeller.
    
    Returns:
        FreeRectangle or None
    """
    best_rect = None
    min_short_side_residual = float('inf')
    
    for rect in free_rects:
        if rect.can_fit(item_l, item_w, item_h):
            residual_l = rect.length - item_l
            residual_w = rect.width - item_w
            short_side_residual = min(residual_l, residual_w)
            
            if short_side_residual < min_short_side_residual:
                min_short_side_residual = short_side_residual
                best_rect = rect
            elif short_side_residual == min_short_side_residual and best_rect is not None:
                current_vol_diff = rect.volume - (item_l * item_w * item_h)
                best_vol_diff = best_rect.volume - (item_l * item_w * item_h)
                if current_vol_diff < best_vol_diff:
                    best_rect = rect
    
    return best_rect


def remove_redundant_rectangles(rects):
    """
    Birbirinin içinde olan dikdörtgenleri kaldırır.
    Küçük olanı sil, büyüğü tut (daha geniş arama alanı).
    """
    filtered = []
    
    for i, rect1 in enumerate(rects):
        is_contained = False
        
        for j, rect2 in enumerate(rects):
            if i == j:
                continue
            
            if (rect2.x <= rect1.x and 
                rect2.y <= rect1.y and 
                rect2.z <= rect1.z and
                rect2.x + rect2.length >= rect1.x + rect1.length and
                rect2.y + rect2.width >= rect1.y + rect1.width and
                rect2.z + rect2.height >= rect1.z + rect1.height):
                is_contained = True
                break
        
        if not is_contained:
            filtered.append(rect1)
    
    return filtered


def pack_maximal_rectangles(urunler, palet_cfg):
    """
    TRUE 3D MAXIMAL RECTANGLES ALGORITHM with AUTO-ORIENTATION.
    
    Ana yerleştirme motoru. Temel özellikler:
    1. Kesişim tabanlı bölme: Kutu yerleştirildiğinde tüm kesişen boş
       dikdörtgenler en fazla 6 alt-dikdörtgene bölünür.
    2. Örtüşen dikdörtgenler: Guillotine'den farklı olarak örtüşen boş
       alanlar tutulur, sadece tamamen kapsananlar silinir.
    3. Auto-Orientation: Her ürün için tüm yönelimler denenir.
    
    Complexity: O(n × r × f) - n: ürün, r: yönelim, f: boş dikdörtgen
    
    Args:
        urunler: GA'dan gelen ürün sıralaması
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        list[dict]: Her palet için {'items': [...], 'weight': float}
    """
    pallets = []
    current_pallet = {
        'items': [],
        'weight': 0.0,
        'free_rects': [FreeRectangle(
            0, 0, 0, 
            palet_cfg.length, palet_cfg.width, palet_cfg.height
        )]
    }
    
    for idx, urun in enumerate(urunler):
        u_wgt = urun.agirlik
        
        # Ağırlık kontrolü - yeni palet gerekiyor mu?
        if current_pallet['weight'] + u_wgt > palet_cfg.max_weight:
            if current_pallet['items']:
                pallets.append({
                    'items': current_pallet['items'],
                    'weight': current_pallet['weight']
                })
            current_pallet = {
                'items': [],
                'weight': 0.0,
                'free_rects': [FreeRectangle(
                    0, 0, 0,
                    palet_cfg.length, palet_cfg.width, palet_cfg.height
                )]
            }
        
        # AUTO-ORIENTATION: Tüm yönelimler × tüm boş dikdörtgenler
        best_rect = None
        best_orientation = None
        min_short_side = float('inf')
        
        orientations = possible_orientations_for(urun)
        
        for dims in orientations:
            item_l, item_w, item_h = dims
            
            for rect in current_pallet['free_rects']:
                if rect.can_fit(item_l, item_w, item_h):
                    residual_l = rect.length - item_l
                    residual_w = rect.width - item_w
                    short_side = min(residual_l, residual_w)
                    
                    if short_side < min_short_side:
                        min_short_side = short_side
                        best_rect = rect
                        best_orientation = (item_l, item_w, item_h)
        
        # Hiçbir yönelimde sığmadıysa yeni palet aç
        if best_rect is None:
            if current_pallet['items']:
                pallets.append({
                    'items': current_pallet['items'],
                    'weight': current_pallet['weight']
                })
            
            current_pallet = {
                'items': [],
                'weight': 0.0,
                'free_rects': [FreeRectangle(
                    0, 0, 0,
                    palet_cfg.length, palet_cfg.width, palet_cfg.height
                )]
            }
            
            # SMART NEW PALLET: İlk ürün için tüm yönelimleri dene
            best_rect = None
            best_orientation = None
            min_short_side = float('inf')
            
            for dims in orientations:
                item_l, item_w, item_h = dims
                rect = current_pallet['free_rects'][0]
                
                if rect.can_fit(item_l, item_w, item_h):
                    residual_l = rect.length - item_l
                    residual_w = rect.width - item_w
                    short_side = min(residual_l, residual_w)
                    
                    if short_side < min_short_side:
                        min_short_side = short_side
                        best_rect = rect
                        best_orientation = (item_l, item_w, item_h)
            
            if best_rect is None:
                best_rect = current_pallet['free_rects'][0]
                best_orientation = orientations[0]
        
        # Ürünü en iyi yönelimle yerleştir
        u_l, u_w, u_h = best_orientation
        placed_x, placed_y, placed_z = best_rect.x, best_rect.y, best_rect.z
        
        current_pallet['items'].append({
            'urun': urun,
            'x': placed_x,
            'y': placed_y,
            'z': placed_z,
            'L': u_l,
            'W': u_w,
            'H': u_h
        })
        current_pallet['weight'] += u_wgt
        
        # TRUE MAXIMAL RECTANGLES SPLITTING
        new_free_rects = []
        
        for rect in current_pallet['free_rects']:
            if intersects_3d(rect, placed_x, placed_y, placed_z, u_l, u_w, u_h):
                sub_rects = split_rectangle_maximal(
                    rect, placed_x, placed_y, placed_z, u_l, u_w, u_h
                )
                new_free_rects.extend(sub_rects)
            else:
                new_free_rects.append(rect)
        
        current_pallet['free_rects'] = new_free_rects
        current_pallet['free_rects'] = remove_redundant_rectangles(
            current_pallet['free_rects']
        )
    
    # Son paleti ekle
    if current_pallet['items']:
        pallets.append({
            'items': current_pallet['items'],
            'weight': current_pallet['weight']
        })
    
    return pallets


# ====================================================================
# SHELF-BASED PACKING (Legacy Destek)
# ====================================================================

def pack_shelf_based(urunler, rot_gen, palet_cfg):
    """
    GA Motoru için Shelf (Raf) yerleştirme - Legacy.
    
    Args:
        urunler: Ürün listesi
        rot_gen: Rotasyon genleri (her ürün için yönelim indeksi)
        palet_cfg: PaletConfig nesnesi
    """
    pallets = []
    current_items = []
    
    x, y, z = 0.0, 0.0, 0.0
    current_weight = 0.0
    current_shelf_height = 0.0
    current_shelf_y = 0.0    
    
    L, W, H = palet_cfg.length, palet_cfg.width, palet_cfg.height
    
    for idx, urun in enumerate(urunler):
        r = 0
        if rot_gen and idx < len(rot_gen):
            r = rot_gen[idx]
        
        dims = possible_orientations_for(urun)
        if r >= len(dims):
            r = 0
        u_l, u_w, u_h = dims[r]
        u_wgt = urun.agirlik
        
        if current_weight + u_wgt > palet_cfg.max_weight:
            pallets.append({"items": current_items, "weight": current_weight})
            current_items = []
            current_weight = 0.0
            x, y, z = 0.0, 0.0, 0.0
            current_shelf_height, current_shelf_y = 0.0, 0.0

        if x + u_l > L:
            x = 0
            y += current_shelf_y if current_shelf_y > 0 else u_w
            current_shelf_y = 0 
            
        if y + u_w > W:
            x = 0
            y = 0
            z += current_shelf_height if current_shelf_height > 0 else u_h
            current_shelf_height = 0
            
        if z + u_h > H:
            pallets.append({"items": current_items, "weight": current_weight})
            current_items = []
            current_weight = 0.0
            x, y, z = 0.0, 0.0, 0.0
            current_shelf_height, current_shelf_y = 0.0, 0.0

        current_items.append({
            "urun": urun,
            "x": x, "y": y, "z": z,
            "L": u_l, "W": u_w, "H": u_h
        })
        current_weight += u_wgt
        
        x += u_l
        if u_h > current_shelf_height:
            current_shelf_height = u_h
        if u_w > current_shelf_y:
            current_shelf_y = u_w
        
    if current_items:
        pallets.append({"items": current_items, "weight": current_weight})
        
    return pallets


def basit_palet_paketleme(chromosome, palet_cfg):
    """
    Kromozomdan paletleri oluşturur.
    
    Args:
        chromosome: (urunler, rotations) tuple
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        list[dict]: Her palet için placements ve weight
    """
    urunler, rotations = chromosome
    pallets = pack_shelf_based(urunler, rotations, palet_cfg)
    
    result = []
    for pallet in pallets:
        placements = []
        for item in pallet['items']:
            placements.append({
                'urun': item['urun'],
                'x': item['x'],
                'y': item['y'],
                'z': item['z'],
                'L': item['L'],
                'W': item['W'],
                'H': item['H']
            })
        result.append({
            'placements': placements,
            'weight': pallet['weight']
        })
    
    return result
