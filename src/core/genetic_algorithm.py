"""
Genetik Algoritma Ana Motoru
==============================

NP-Hard 3D Bin Packing problemi için Genetik Algoritma çözücüsü.

Özellikler:
    - Adaptive Weights (Dinamik Ağırlık Ayarlama)
    - Height-Aware Seeding (Yükseklik Bazlı Tohum)
    - Anti-Stagnation (Genetik Şok Mekanizması)
    - Local Search (Hill Climbing Optimizasyonu)
    - Auto-Orientation (Otomatik Yönelim)
    
Referanslar:
    - Holland, J. "Adaptation in Natural and Artificial Systems" (1975)
    - Goldberg, D. "Genetic Algorithms in Search, Optimization" (1989)
"""

import random
from typing import List

from .chromosome import Chromosome
from .fitness import evaluate_fitness, adapt_weights, get_weights
from ..utils.helpers import urun_hacmi
from ..models.container import PaletConfig


# ====================================================================
# GENETİK OPERATÖRLER
# ====================================================================

def tournament_selection(population: List[Chromosome], k: int = 3) -> Chromosome:
    """K kişilik turnuva seçimi: en fit olan kazanır."""
    turnuva = random.sample(population, k)
    turnuva.sort(key=lambda c: c.fitness, reverse=True)
    return turnuva[0].copy()


def crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Order Crossover (OX) - SEQUENCE-ONLY MODE.
    Auto-Orientation eliminates need for rotation gene crossover.
    """
    n = parent1.n
    sira1 = parent1.sira_gen
    sira2 = parent2.sira_gen

    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)

    child_sira = [-1] * n
    child_sira[i:j] = sira1[i:j]

    p2_filtered = [g for g in sira2 if g not in child_sira]
    idx = 0
    for pos in range(n):
        if child_sira[pos] == -1:
            child_sira[pos] = p2_filtered[idx]
            idx += 1

    child = Chromosome(urunler=parent1.urunler, sira_gen=child_sira)
    return child


def mutate(individual: Chromosome, mutation_rate: float = 0.05):
    """Swap mutasyonu - SEQUENCE-ONLY MODE."""
    n = individual.n

    if random.random() < mutation_rate:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        individual.sira_gen[i], individual.sira_gen[j] = (
            individual.sira_gen[j],
            individual.sira_gen[i],
        )


# ====================================================================
# TOHUM (SEED) STRATEJİLERİ
# ====================================================================

def create_seeded_chromosome(urunler, seed_type='volume'):
    """
    Heuristik tohum kromozomu oluşturur.
    
    Args:
        urunler: Ürün listesi
        seed_type: 'volume' veya 'weight'
    """
    n = len(urunler)
    
    if seed_type == 'volume':
        indexed_items = [(i, urun_hacmi(urunler[i])) for i in range(n)]
        indexed_items.sort(key=lambda x: x[1], reverse=True)
    elif seed_type == 'weight':
        indexed_items = [(i, urunler[i].agirlik) for i in range(n)]
        indexed_items.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Geçersiz seed_type: {seed_type}")
    
    sira_gen = [item[0] for item in indexed_items]
    return Chromosome(urunler=urunler, sira_gen=sira_gen)


def create_block_sorted_chromosome(urunler):
    """
    BLOCK-AWARE Seeding: Aynı boyutlu ürünleri bloklar halinde sıralar.
    'Duvar gibi' istifleme yapılmasını sağlar.
    """
    n = len(urunler)
    
    groups = {}
    for i in range(n):
        dims = tuple(sorted([urunler[i].boy, urunler[i].en, urunler[i].yukseklik]))
        if dims not in groups:
            groups[dims] = []
        groups[dims].append(i)
    
    sorted_groups = sorted(groups.items(), key=lambda x: x[0][0] * x[0][1] * x[0][2], reverse=True)
    
    sira_gen = []
    for dims, indices in sorted_groups:
        sira_gen.extend(indices)
    
    return Chromosome(urunler=urunler, sira_gen=sira_gen)


def create_height_sorted_chromosome(urunler):
    """
    HEIGHT-AWARE Seeding: Ürünleri YÜKSEKLİĞE göre gruplar.
    
    30cm ve 35cm yükseklikli ürünlerin karışmasını önler.
    Aynı yükseklikteki ürünleri birlikte yerleştirerek düz katmanlar oluşturur.
    """
    n = len(urunler)
    
    height_groups = {}
    for i in range(n):
        height = urunler[i].yukseklik
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(i)
    
    sorted_heights = sorted(height_groups.items(), key=lambda x: x[0], reverse=True)
    
    sira_gen = []
    for height, indices in sorted_heights:
        sira_gen.extend(indices)
    
    return Chromosome(urunler=urunler, sira_gen=sira_gen)


# ====================================================================
# ANA GA DÖNGÜSÜ
# ====================================================================

def run_ga(urunler, palet_cfg: PaletConfig, population_size=None, generations=None,
           elitism=None, mutation_rate=0.4, tournament_k=2):
    """
    Genetik Algoritma Ana Döngüsü - ADAPTIVE WEIGHTS & PARAMETERS.
    
    Args:
        urunler: UrunData listesi
        palet_cfg: PaletConfig nesnesi
        population_size: Popülasyon büyüklüğü (None = otomatik)
        generations: Nesil sayısı (None = otomatik)
        elitism: Elitizm sayısı (None = %5)
        mutation_rate: Mutasyon oranı
        tournament_k: Turnuva büyüklüğü
        
    Returns:
        tuple: (best_chromosome, history_list)
    """
    if not urunler:
        return None, []

    n_urun = len(urunler)
    
    # ADAPTIVE PARAMETERS
    if n_urun > 100:
        population_size = 50
        generations = 100
        mutation_rate = 0.4
        tournament_k = 2
        print(f"LIGHT & FAST MODE (Auto-Orientation): n_urun={n_urun} > 100")
        print(f"   Parameters: pop=50, gen=100, mut=0.4, tournament_k=2")
    else:
        if population_size is None:
            population_size = 50
        if generations is None:
            generations = 30
    
    if elitism is None:
        elitism = max(2, int(population_size * 0.05))

    # ── DEFENSIVE SAFEGUARDS ────────────────────────────────────────────
    population_size = max(2, population_size)
    generations     = max(1, generations)
    mutation_rate   = max(0.0, min(1.0, mutation_rate))
    elitism         = min(elitism, population_size - 1)
    # ─────────────────────────────────────────────────────────────────────

    print(f"GA Parametreler:")
    print(f"   Ürün Sayısı: {n_urun}")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    print(f"   Elitism: {elitism}")
    print(f"   Mutation Rate: {mutation_rate}")
    print(f"   Tournament K: {tournament_k}")

    # Teorik minimum palet sayısı
    total_load_vol = sum(urun_hacmi(u) for u in urunler)
    theo_min_pallets = max(1, int(total_load_vol / palet_cfg.volume) + 1)
    
    # HEIGHT-AWARE INITIAL POPULATION
    population: List[Chromosome] = []
    
    num_height_seeds = max(1, int(population_size * 0.40))
    print(f"Creating {num_height_seeds} HEIGHT-sorted seeds (anti-staircase effect)...")
    for _ in range(num_height_seeds):
        population.append(create_height_sorted_chromosome(urunler))
    
    num_random = population_size - len(population)
    print(f"Creating {num_random} RANDOM individuals for AI discovery...")
    for _ in range(num_random):
        population.append(Chromosome(urunler=urunler))
    
    print(f"Total population: {len(population)} (Height: {num_height_seeds}, Random: {num_random})")

    # İlk fitness hesaplaması
    for c in population:
        evaluate_fitness(c, palet_cfg)

    history = []
    
    # Anti-Stagnation takip değişkenleri
    best_fitness_tracker = float('-inf')
    generations_without_improvement = 0

    for gen in range(generations):
        population.sort(key=lambda c: c.fitness, reverse=True)
        current_best = population[0]
        
        # LOCAL SEARCH (Hill Climbing)
        if gen % 5 == 0:
            original_fitness = current_best.fitness
            best_local = current_best.copy()
            
            for _ in range(10):
                candidate = current_best.copy()
                
                if random.random() < 0.7:
                    segment_size = max(5, int(candidate.n * 0.15))
                    start_idx = random.randint(0, max(0, candidate.n - segment_size))
                    end_idx = min(start_idx + segment_size, candidate.n)
                    
                    segment_items = candidate.sira_gen[start_idx:end_idx]
                    segment_with_heights = [(idx, urunler[idx].yukseklik) for idx in segment_items]
                    segment_with_heights.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (idx, _) in enumerate(segment_with_heights):
                        candidate.sira_gen[start_idx + i] = idx
                else:
                    i = random.randint(0, candidate.n - 1)
                    j = random.randint(0, candidate.n - 1)
                    candidate.sira_gen[i], candidate.sira_gen[j] = candidate.sira_gen[j], candidate.sira_gen[i]
                
                evaluate_fitness(candidate, palet_cfg)
                
                if candidate.fitness > best_local.fitness:
                    best_local = candidate.copy()
            
            if best_local.fitness > original_fitness:
                population[0] = best_local
                current_best = best_local
                print(f"  Height-Aware Local Search improved: {original_fitness:.2f} -> {best_local.fitness:.2f}")
        
        # ANTI-STAGNATION
        if current_best.fitness > best_fitness_tracker:
            best_fitness_tracker = current_best.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # GENETİK ŞOK
        if generations_without_improvement >= 20 and gen < generations - 5:
            print(f"  GENETIK SOK: {generations_without_improvement} nesil iyilesme yok")
            num_to_replace = int(population_size * 0.5)
            half = num_to_replace // 2
            
            idx = elitism
            
            for _ in range(half):
                if idx < len(population):
                    population[idx] = create_seeded_chromosome(urunler, seed_type='volume')
                    evaluate_fitness(population[idx], palet_cfg)
                    idx += 1
            
            remaining = num_to_replace - half
            for _ in range(remaining):
                if idx < len(population):
                    population[idx] = create_seeded_chromosome(urunler, seed_type='weight')
                    evaluate_fitness(population[idx], palet_cfg)
                    idx += 1
            
            print(f"  {num_to_replace} sirali tohum eklendi ({half} hacim, {remaining} agirlik)")
            generations_without_improvement = 0
        
        # ADAPTIVE WEIGHTS
        if gen % 5 == 0 and gen > 0:
            adapt_weights(current_best, theo_min_pallets)
        
        avg_fitness = sum(c.fitness for c in population) / len(population)
        
        history.append({
            "generation": gen,
            "best_fitness": current_best.fitness,
            "avg_fitness": avg_fitness,
            "best_palet_sayisi": current_best.palet_sayisi,
            "best_doluluk": current_best.ortalama_doluluk,
        })

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen}: Best Fit={current_best.fitness:.2f}, "
                  f"Palet={current_best.palet_sayisi}, Doluluk={current_best.ortalama_doluluk:.2%}")

        # Yeni popülasyon
        new_population: List[Chromosome] = []

        for i in range(min(elitism, len(population))):
            new_population.append(population[i].copy())

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, k=tournament_k)
            parent2 = tournament_selection(population, k=tournament_k)

            child = crossover(parent1, parent2)
            mutate(child, mutation_rate=mutation_rate)

            evaluate_fitness(child, palet_cfg)
            new_population.append(child)

        population = new_population

    # Son değerlendirme
    population.sort(key=lambda c: c.fitness, reverse=True)
    best_solution = population[0]

    return best_solution, history
