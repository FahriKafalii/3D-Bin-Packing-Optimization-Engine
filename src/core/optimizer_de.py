"""
Advanced Differential Evolution (DE) Optimizer for 3D Bin Packing
===================================================================

Senior Research Scientist in Optimization implementation.

Strategy: DE/current-to-best/1 with adaptive jitter, fitness caching, 
two-stage decode, elite repair, and volume-biased initialization.

Key Features:
    1. DE/current-to-best/1 mutation with jitter
    2. Split crossover (CR_p for priority, CR_r for rotation if exists)
    3. Fitness caching to avoid redundant packing evaluations
    4. Two-stage decode: cheap volume bound check before full packing
    5. Elite repair: local search on top performers every 5 iterations
    6. Biased initialization: volume-descending rank bias

References:
    - Storn & Price, "Differential Evolution – A Simple and Efficient Heuristic" (1997)
    - Das & Suganthan, "Differential Evolution: A Survey" (2011)
    - Price, Storn & Lampinen, "Differential Evolution: A Practical Approach" (2005)
"""

import math
import random
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .packing import pack_maximal_rectangles, compact_pallet, local_repair
from .fitness import get_weights

# Environment flag for support constraint verification
DEBUG_SUPPORT = os.getenv("DEBUG_SUPPORT") == "1"
from ..utils.helpers import urun_hacmi

# ====================================================================
# DE FITNESS CONSTANTS  (lexicographic priority order)
# ====================================================================
# BIG_PALLET dominates W_UTIL: 1 fewer pallet always beats any util gain.
# Guarantee:  BIG_PALLET(50k) > W_UTIL(1k) * 1.0(max possible util)  ✓
# Unplaced items are handled by the packing engine itself (pack_maximal_rectangles
# never drops items; it opens a new pallet instead).
_DE_BIG_PALLET = 50_000   # Penalty per extra pallet above theo_min
_DE_W_OPTIMAL  = 20_000   # Bonus when P_GA <= theo_min (matching/beating volumetric minimum)
_DE_W_UTIL     =  1_000   # Secondary: avg utilisation bonus [0..1 scale]
from ..models.container import PaletConfig
from ..models.product import UrunData


# ====================================================================
# DE INDIVIDUAL (CANDIDATE SOLUTION)
# ====================================================================

@dataclass
class DEIndividual:
    """
    Differential Evolution individual representation.
    
    Attributes:
        priority_keys: Continuous vector [0,1]^n defining item order
        rotation_keys: Optional continuous vector [0,1]^n for rotations (unused with Auto-Orientation)
        fitness: Evaluated fitness score
        palet_sayisi: Number of pallets used
        ortalama_doluluk: Average pallet utilization
        decoded_order: Cached decoded order (argsort of priority_keys)
    """
    priority_keys: np.ndarray
    rotation_keys: Optional[np.ndarray] = None
    fitness: float = float('-inf')
    palet_sayisi: int = 0
    ortalama_doluluk: float = 0.0
    decoded_order: Optional[List[int]] = None
    
    def copy(self) -> 'DEIndividual':
        """Create a deep copy of the individual."""
        return DEIndividual(
            priority_keys=self.priority_keys.copy(),
            rotation_keys=self.rotation_keys.copy() if self.rotation_keys is not None else None,
            fitness=self.fitness,
            palet_sayisi=self.palet_sayisi,
            ortalama_doluluk=self.ortalama_doluluk,
            decoded_order=self.decoded_order.copy() if self.decoded_order else None
        )


# ====================================================================
# FITNESS CACHING
# ====================================================================

class FitnessCache:
    """
    Cache to avoid redundant packing evaluations.
    
    Key: deterministic hash of (order, rotations if present)
    Value: (fitness, palet_sayisi, ortalama_doluluk)
    """
    
    def __init__(self):
        self.cache: Dict[Tuple, Tuple[float, int, float]] = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, individual: DEIndividual) -> Tuple:
        """Generate deterministic cache key."""
        order_tuple = tuple(individual.decoded_order)
        if individual.rotation_keys is not None:
            rot_tuple = tuple(np.round(individual.rotation_keys, 4))  # Round for stability
            return order_tuple + rot_tuple
        return order_tuple
    
    def get(self, individual: DEIndividual) -> Optional[Tuple[float, int, float]]:
        """Retrieve cached fitness if available."""
        key = self.get_key(individual)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, individual: DEIndividual, fitness: float, palet_sayisi: int, ortalama_doluluk: float):
        """Store fitness in cache."""
        key = self.get_key(individual)
        self.cache[key] = (fitness, palet_sayisi, ortalama_doluluk)
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ====================================================================
# DECODING & EVALUATION
# ====================================================================

def decode_to_order(priority_keys: np.ndarray) -> List[int]:
    """
    Decode continuous priority keys to item order.
    
    Higher priority key → earlier in sequence.
    
    Args:
        priority_keys: Continuous vector [0,1]^n
    
    Returns:
        List of item indices in order of decreasing priority
    """
    # Argsort in descending order (highest priority first)
    return np.argsort(-priority_keys).tolist()


def stage_a_lower_bound_check(
    urunler: List[UrunData], 
    palet_cfg: PaletConfig,
    order: List[int],
    best_palet_count: int
) -> bool:
    """
    STAGE A: Fast volume-based lower bound check.
    
    Estimates minimum bins required based on volume alone.
    If this cheap estimate already exceeds current best, skip full packing.
    
    Args:
        urunler: Product list
        palet_cfg: Pallet configuration
        order: Decoded item order
        best_palet_count: Current best pallet count to beat
    
    Returns:
        True if promising (proceed to Stage B), False if clearly poor
    """
    # Calculate total volume in order
    total_volume = sum(urun_hacmi(urunler[i]) for i in order)
    
    # Theoretical minimum bins (optimistic lower bound)
    pallet_volume = palet_cfg.volume
    theoretical_min = max(1, int(np.ceil(total_volume / pallet_volume)))
    
    # If theoretical minimum is already worse than best, reject early
    # Allow 1 bin tolerance for uncertainty
    if theoretical_min > best_palet_count + 1:
        return False
    
    # Additional heuristic: check if heaviest items fit weight constraint
    # (This is a cheap additional check)
    max_item_weight = max(urunler[i].agirlik for i in order)
    if max_item_weight > palet_cfg.max_weight:
        return False
    
    return True


def evaluate_de_individual(
    individual: DEIndividual,
    urunler: List[UrunData],
    palet_cfg: PaletConfig,
    cache: FitnessCache,
    best_palet_count: int = 999
) -> None:
    """
    TWO-STAGE DECODE & EVALUATE with caching.
    
    Stage A: Cheap volume-based bound check
    Stage B: Full packing evaluation (only if Stage A passes)
    
    Args:
        individual: DE individual to evaluate
        urunler: Product list
        palet_cfg: Pallet configuration
        cache: Fitness cache
        best_palet_count: Current best pallet count (for Stage A filtering)
    """
    # Decode priority keys to order
    individual.decoded_order = decode_to_order(individual.priority_keys)
    
    # Check cache first
    cached = cache.get(individual)
    if cached is not None:
        individual.fitness, individual.palet_sayisi, individual.ortalama_doluluk = cached
        return
    
    # STAGE A: Fast lower bound check
    if not stage_a_lower_bound_check(urunler, palet_cfg, individual.decoded_order, best_palet_count):
        individual.fitness = -1e8
        individual.palet_sayisi = best_palet_count + 10
        individual.ortalama_doluluk = 0.0
        cache.put(individual, individual.fitness, individual.palet_sayisi, individual.ortalama_doluluk)
        return
    
    # STAGE B: Full packing evaluation
    # Create ordered product sequence
    ordered_urunler = [urunler[i] for i in individual.decoded_order]
    
    # Pack using maximal rectangles (with Auto-Orientation & Gravity Constraint)
    # Enable debug_support if DEBUG_SUPPORT=1 env var is set
    pallets = pack_maximal_rectangles(ordered_urunler, palet_cfg, debug_support=DEBUG_SUPPORT)
    
    # COMPACTION PASS: her paleti kompaktleştir (gravity + origin)
    # Palet sayısını değiştirmez; isel doluluk artar.
    for p in pallets:
        compact_pallet(p, palet_cfg)
    
    if not pallets:
        individual.fitness = -1e9
        individual.palet_sayisi = 999
        individual.ortalama_doluluk = 0.0
        cache.put(individual, individual.fitness, individual.palet_sayisi, individual.ortalama_doluluk)
        return
    
    # ── DE LEXICOGRAPHIC FITNESS ───────────────────────────────────────
    # Priority 1 (dominant): pallet count
    # Priority 2 (secondary): avg utilisation
    # Rule: 1 fewer pallet ALWAYS beats any utilisation improvement.
    # --------------------------------------------------------------------
    weights = get_weights()

    P_GA = len(pallets)
    total_load_vol = sum(urun_hacmi(u) for u in urunler)
    theo_min = max(1, math.ceil(total_load_vol / palet_cfg.volume))

    # Avg utilisation (global scalar — avoids per-pallet volume inflation)
    total_item_vol = sum(
        sum(i["L"] * i["W"] * i["H"] for i in p["items"])
        for p in pallets
    )
    avg_doluluk = (total_item_vol / (P_GA * palet_cfg.volume)) if P_GA > 0 else 0.0

    fitness_score = 0.0

    # PRIORITY 1: Pallet count — lexicographically dominant term
    if P_GA <= theo_min:
        fitness_score += _DE_W_OPTIMAL          # Hitting or beating volumetric minimum
    else:
        fitness_score -= _DE_BIG_PALLET * (P_GA - theo_min)   # Hard penalty per extra pallet

    # PRIORITY 2: Avg utilisation — secondary (capped well below 1 pallet)
    fitness_score += _DE_W_UTIL * avg_doluluk

    if DEBUG_SUPPORT:
        print(
            f"[DE FITNESS] P={P_GA} theo={theo_min} "
            f"util={avg_doluluk:.2%} "
            f"pallet_term={fitness_score - _DE_W_UTIL * avg_doluluk:.0f} "
            f"util_term={_DE_W_UTIL * avg_doluluk:.0f} "
            f"total={fitness_score:.0f}"
        )

    # Constraint violations (weight overload → infeasible)
    has_violation = False
    for pallet in pallets:
        if pallet["weight"] > palet_cfg.max_weight:
            fitness_score -= weights["w_weight_violation"]
            has_violation = True

    if has_violation:
        fitness_score = min(fitness_score, -1e6)
    
    # Store results
    individual.fitness = fitness_score
    individual.palet_sayisi = P_GA
    individual.ortalama_doluluk = avg_doluluk
    
    # Cache the result
    cache.put(individual, fitness_score, P_GA, avg_doluluk)


# ====================================================================
# BIASED INITIALIZATION
# ====================================================================

def create_biased_population(
    n_items: int,
    population_size: int,
    urunler: List[UrunData],
    use_rotations: bool = False
) -> List[DEIndividual]:
    """
    Create population with VOLUME-BIASED initialization.
    
    Priority key initialization:
        key_i = 0.7*rand() + 0.3*(rank(volume_i)/N)
    
    Where rank(volume_i) is the position in descending volume order,
    normalized to [0,1]. Larger items get higher base priority.
    
    Args:
        n_items: Number of items
        population_size: Population size
        urunler: Product list
        use_rotations: Whether to include rotation keys (False for Auto-Orientation)
    
    Returns:
        List of initialized DE individuals
    """
    population = []
    
    # Calculate volume ranks (descending order)
    volumes = [urun_hacmi(u) for u in urunler]
    volume_ranks = np.argsort(np.argsort(-np.array(volumes)))  # Rank 0 = largest
    normalized_ranks = volume_ranks / (n_items - 1) if n_items > 1 else np.zeros(n_items)
    
    for _ in range(population_size):
        # Biased priority keys: 70% random + 30% volume rank
        random_component = np.random.uniform(0, 1, n_items) * 0.7
        bias_component = normalized_ranks * 0.3
        priority_keys = random_component + bias_component
        
        # Rotation keys (unused with Auto-Orientation, but kept for API compatibility)
        rotation_keys = np.random.uniform(0, 1, n_items) if use_rotations else None
        
        individual = DEIndividual(
            priority_keys=priority_keys,
            rotation_keys=rotation_keys
        )
        population.append(individual)
    
    return population


# ====================================================================
# DE MUTATION: HYBRID STRATEGY
# ====================================================================

def mutate_de_current_to_best(
    population: List[DEIndividual],
    target_idx: int,
    best_idx: int,
    F: float = 0.8,
    bounds: Tuple[float, float] = (0.0, 1.0)
) -> DEIndividual:
    """
    DE/current-to-best/1 mutation with adaptive mutation factor.
    
    Formula:
        Fi = uniform(0.4, 0.9)  # Adaptive mutation factor
        v = x_i + Fi*(x_best - x_i) + Fi*(x_r1 - x_r2)
    
    Args:
        population: Current population
        target_idx: Index of target individual
        best_idx: Index of best individual
        F: Base mutation scale factor (unused, kept for API compatibility)
        bounds: Bounds for clamping (min, max)
    
    Returns:
        Mutant individual
    """
    n_pop = len(population)
    n_dims = len(population[0].priority_keys)
    
    # Get distinct random indices (excluding target and best)
    indices = [i for i in range(n_pop) if i != target_idx and i != best_idx]
    r1, r2 = random.sample(indices, 2)
    
    # ADAPTIVE mutation factor: uniform(0.4, 0.9)
    Fi = np.random.uniform(0.4, 0.9)
    
    # Current individual
    x_i = population[target_idx].priority_keys
    x_best = population[best_idx].priority_keys
    x_r1 = population[r1].priority_keys
    x_r2 = population[r2].priority_keys
    
    # Mutant vector: v = x_i + Fi*(x_best - x_i) + Fi*(x_r1 - x_r2)
    mutant_priority = x_i + Fi * (x_best - x_i) + Fi * (x_r1 - x_r2)
    
    # Clamp to bounds
    mutant_priority = np.clip(mutant_priority, bounds[0], bounds[1])
    
    # Handle rotation keys if present
    mutant_rotation = None
    if population[0].rotation_keys is not None:
        x_i_rot = population[target_idx].rotation_keys
        x_best_rot = population[best_idx].rotation_keys
        x_r1_rot = population[r1].rotation_keys
        x_r2_rot = population[r2].rotation_keys
        
        mutant_rotation = x_i_rot + Fi * (x_best_rot - x_i_rot) + Fi * (x_r1_rot - x_r2_rot)
        mutant_rotation = np.clip(mutant_rotation, bounds[0], bounds[1])
    
    return DEIndividual(
        priority_keys=mutant_priority,
        rotation_keys=mutant_rotation
    )


def mutate_de_rand_1(
    population: List[DEIndividual],
    target_idx: int,
    bounds: Tuple[float, float] = (0.0, 1.0)
) -> DEIndividual:
    """
    DE/rand/1 mutation with adaptive mutation factor.
    
    Formula:
        Fi = uniform(0.4, 0.9)  # Adaptive mutation factor
        v = x_r1 + Fi*(x_r2 - x_r3)
    
    Args:
        population: Current population
        target_idx: Index of target individual (excluded from selection)
        bounds: Bounds for clamping (min, max)
    
    Returns:
        Mutant individual
    """
    n_pop = len(population)
    
    # Get distinct random indices (excluding target)
    indices = [i for i in range(n_pop) if i != target_idx]
    r1, r2, r3 = random.sample(indices, 3)
    
    # ADAPTIVE mutation factor: uniform(0.4, 0.9)
    Fi = np.random.uniform(0.4, 0.9)
    
    # Random base vector
    x_r1 = population[r1].priority_keys
    x_r2 = population[r2].priority_keys
    x_r3 = population[r3].priority_keys
    
    # Mutant vector: v = x_r1 + Fi*(x_r2 - x_r3)
    mutant_priority = x_r1 + Fi * (x_r2 - x_r3)
    
    # Clamp to bounds
    mutant_priority = np.clip(mutant_priority, bounds[0], bounds[1])
    
    # Handle rotation keys if present
    mutant_rotation = None
    if population[0].rotation_keys is not None:
        x_r1_rot = population[r1].rotation_keys
        x_r2_rot = population[r2].rotation_keys
        x_r3_rot = population[r3].rotation_keys
        
        mutant_rotation = x_r1_rot + Fi * (x_r2_rot - x_r3_rot)
        mutant_rotation = np.clip(mutant_rotation, bounds[0], bounds[1])
    
    return DEIndividual(
        priority_keys=mutant_priority,
        rotation_keys=mutant_rotation
    )


def hybrid_mutation(
    population: List[DEIndividual],
    target_idx: int,
    best_idx: int,
    F: float = 0.8,
    bounds: Tuple[float, float] = (0.0, 1.0)
) -> DEIndividual:
    """
    HYBRID mutation strategy: 50% DE/current-to-best/1, 50% DE/rand/1.
    
    This provides a balance between exploitation (best-based) and 
    exploration (random-based) search.
    
    Args:
        population: Current population
        target_idx: Index of target individual
        best_idx: Index of best individual
        F: Base mutation scale factor
        bounds: Bounds for clamping (min, max)
    
    Returns:
        Mutant individual
    """
    if random.random() < 0.5:
        # 50% probability: DE/current-to-best/1 (exploitation)
        return mutate_de_current_to_best(population, target_idx, best_idx, F, bounds)
    else:
        # 50% probability: DE/rand/1 (exploration)
        return mutate_de_rand_1(population, target_idx, bounds)


# ====================================================================
# SPLIT CROSSOVER
# ====================================================================

def crossover_split(
    target: DEIndividual,
    mutant: DEIndividual,
    CR_p: float = 0.9,
    CR_r: float = 0.65
) -> DEIndividual:
    """
    Split crossover with different rates for priority and rotation.
    
    Args:
        target: Target individual
        mutant: Mutant individual
        CR_p: Crossover rate for priority keys (default 0.9)
        CR_r: Crossover rate for rotation keys (default 0.65, unused if no rotations)
    
    Returns:
        Trial individual
    """
    n_dims = len(target.priority_keys)
    
    # Priority crossover
    trial_priority = np.zeros(n_dims)
    j_rand = random.randint(0, n_dims - 1)  # Ensure at least one gene from mutant
    
    for j in range(n_dims):
        if random.random() < CR_p or j == j_rand:
            trial_priority[j] = mutant.priority_keys[j]
        else:
            trial_priority[j] = target.priority_keys[j]
    
    # Rotation crossover (if applicable)
    trial_rotation = None
    if target.rotation_keys is not None:
        trial_rotation = np.zeros(n_dims)
        j_rand_rot = random.randint(0, n_dims - 1)
        
        for j in range(n_dims):
            if random.random() < CR_r or j == j_rand_rot:
                trial_rotation[j] = mutant.rotation_keys[j]
            else:
                trial_rotation[j] = target.rotation_keys[j]
    
    return DEIndividual(
        priority_keys=trial_priority,
        rotation_keys=trial_rotation
    )


# ====================================================================
# ELITE REPAIR
# ====================================================================

def elite_repair(
    elite_individual: DEIndividual,
    urunler: List[UrunData],
    palet_cfg: PaletConfig,
    cache: FitnessCache,
    n_swaps: int = 50
) -> DEIndividual:
    """
    ENHANCED Elite repair: local search with swap and insert mutations.
    
    Strategy:
        1. Perform n_swaps operations (70% swap, 30% insert)
        2. Swap: exchange two random positions
        3. Insert: move one item to another random position
        4. Keep improvements only
    
    Args:
        elite_individual: Elite individual to repair
        urunler: Product list
        palet_cfg: Pallet configuration
        cache: Fitness cache
        n_swaps: Number of mutation attempts (default 50)
    
    Returns:
        Improved or original individual
    """
    best = elite_individual.copy()
    current = best.copy()
    
    n_items = len(current.priority_keys)
    
    for _ in range(n_swaps):
        if random.random() < 0.7:
            # 70% probability: SWAP mutation
            i, j = random.sample(range(n_items), 2)
            current.priority_keys[i], current.priority_keys[j] = \
                current.priority_keys[j], current.priority_keys[i]
        else:
            # 30% probability: INSERT mutation
            # Remove element from position i and insert at position j
            i = random.randint(0, n_items - 1)
            j = random.randint(0, n_items - 1)
            
            if i != j:
                # Extract value
                temp_val = current.priority_keys[i]
                # Shift elements
                if i < j:
                    current.priority_keys[i:j] = current.priority_keys[i+1:j+1]
                else:
                    current.priority_keys[j+1:i+1] = current.priority_keys[j:i]
                # Insert at new position
                current.priority_keys[j] = temp_val
        
        # Evaluate (compaction applied inside evaluate_de_individual)
        evaluate_de_individual(current, urunler, palet_cfg, cache, best.palet_sayisi)
        
        # Keep if improvement
        if current.fitness > best.fitness:
            best = current.copy()
        else:
            # Revert mutation
            current = best.copy()
    
    return best


def apply_elite_repair_to_population(
    population: List[DEIndividual],
    urunler: List[UrunData],
    palet_cfg: PaletConfig,
    cache: FitnessCache,
    n_elite: int = 3
) -> None:
    """
    Apply elite repair to top n_elite individuals in population.
    
    Args:
        population: Current population (will be modified in-place)
        urunler: Product list
        palet_cfg: Pallet configuration
        cache: Fitness cache
        n_elite: Number of elite individuals to repair
    """
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Repair top n_elite
    for i in range(min(n_elite, len(population))):
        repaired = elite_repair(population[i], urunler, palet_cfg, cache)
        if repaired.fitness > population[i].fitness:
            population[i] = repaired


# ====================================================================
# MAIN DE ALGORITHM
# ====================================================================

def run_de(
    urunler: List[UrunData],
    palet_cfg: PaletConfig,
    population_size: Optional[int] = None,
    generations: Optional[int] = None,
    F: float = 0.8,
    CR_p: float = 0.9,
    CR_r: float = 0.65,
    use_rotations: bool = False
) -> Tuple[Optional[DEIndividual], List[Dict]]:
    """
    Advanced Differential Evolution Algorithm for 3D Bin Packing.
    
    Implements DE/current-to-best/1 with:
        - Jittered mutation
        - Split crossover
        - Fitness caching
        - Two-stage decode
        - Elite repair (every 5 iterations)
        - Volume-biased initialization
    
    Args:
        urunler: List of UrunData products
        palet_cfg: PaletConfig pallet configuration
        population_size: Population size (None = auto)
        generations: Number of generations (None = auto)
        F: Mutation scale factor (default 0.8)
        CR_p: Crossover rate for priority (default 0.9)
        CR_r: Crossover rate for rotation (default 0.65, unused with Auto-Orientation)
        use_rotations: Use explicit rotation keys (False for Auto-Orientation)
    
    Returns:
        Tuple of (best_individual, history_list)
            - best_individual: Best DE solution found
            - history_list: List of generation statistics
    """
    if not urunler:
        return None, []
    
    n_items = len(urunler)
    
    # ADAPTIVE PARAMETERS - UPGRADED
    user_provided_np = population_size is not None
    np_config_source = "USER" if user_provided_np else "AUTO"
    np_user_original = population_size  # Store original user input
    
    # Calculate expected minimum NP
    np_expected = max(60, int(0.8 * n_items))
    
    if population_size is None:
        # AUTO: Use adaptive formula
        population_size = np_expected
    else:
        # ENFORCE MINIMUM: User can increase NP, but not decrease below expected
        population_size = max(population_size, np_expected)
    
    # HARD VERIFICATION: Log NP calculation with enforcement details
    if user_provided_np:
        if np_user_original < np_expected:
            print(f"[DE] NP_CONFIG={np_config_source} NP_USER={np_user_original} "
                  f"NP_EXPECTED={np_expected} NP_ACTUAL={population_size} (enforced minimum) N_ITEMS={n_items}")
        else:
            print(f"[DE] NP_CONFIG={np_config_source} NP_USER={np_user_original} "
                  f"NP_EXPECTED={np_expected} NP_ACTUAL={population_size} N_ITEMS={n_items}")
    else:
        print(f"[DE] NP_CONFIG={np_config_source} NP_ACTUAL={population_size} N_ITEMS={n_items} "
              f"(formula: max(60, int(0.8*{n_items})) = {np_expected})")
    
    if generations is None:
        if n_items > 100:
            generations = 100
        else:
            generations = 50
    
    print("\n" + "="*70)
    print("🔬 ADVANCED DIFFERENTIAL EVOLUTION OPTIMIZER V2.0")
    print("="*70)
    print(f"📊 Configuration:")
    print(f"   Items: {n_items}")
    print(f"   Population: {population_size} (upgraded: max(60, 0.8*N))")
    print(f"   Generations: {generations}")
    print(f"   Strategy: HYBRID (50% DE/current-to-best/1, 50% DE/rand/1)")
    print(f"   Mutation Factor: Adaptive uniform(0.4, 0.9)")
    print(f"   Crossover Rate CR_p: {CR_p}")
    if use_rotations:
        print(f"   Crossover Rate CR_r: {CR_r}")
    else:
        print(f"   Rotation: Auto-Orientation (no explicit rotation genes)")
    print(f"   Elite Repair: Every 5 generations (top 3, 50 mutations)")
    print(f"   Diversity Injection: If stagnant for 8 generations")
    print("="*70 + "\n")
    
    # Initialize fitness cache
    cache = FitnessCache()
    
    # BIASED INITIALIZATION
    print("🌱 Initializing population with volume-biased strategy...")
    population = create_biased_population(n_items, population_size, urunler, use_rotations)
    
    # Evaluate initial population
    print("📈 Evaluating initial population...")
    for ind in population:
        evaluate_de_individual(ind, urunler, palet_cfg, cache)
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_individual = population[0].copy()
    
    print(f"✅ Initial best fitness: {best_individual.fitness:.2f}")
    print(f"   Pallets: {best_individual.palet_sayisi}, Utilization: {best_individual.ortalama_doluluk*100:.1f}%\n")
    
    # History tracking
    history = []
    
    # DIVERSITY INJECTION tracking
    last_improvement_gen = 0
    best_fitness_tracker = best_individual.fitness
    
    # Main DE loop
    for gen in range(generations):
        # Best index (already sorted)
        best_idx = 0
        
        # Track improvements in this generation
        improvements = 0
        
        # DE evolution
        for i in range(population_size):
            # MUTATION: HYBRID strategy (50% current-to-best, 50% rand)
            mutant = hybrid_mutation(population, i, best_idx, F)
            
            # CROSSOVER: Split crossover
            trial = crossover_split(population[i], mutant, CR_p, CR_r)
            
            # EVALUATION: Two-stage decode with caching
            evaluate_de_individual(trial, urunler, palet_cfg, cache, best_individual.palet_sayisi)
            
            # SELECTION: Greedy replacement
            if trial.fitness > population[i].fitness:
                population[i] = trial
                improvements += 1
                
                # Update global best
                if trial.fitness > best_individual.fitness:
                    best_individual = trial.copy()
                    last_improvement_gen = gen
                    best_fitness_tracker = best_individual.fitness
        
        # Re-sort population
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # DIVERSITY INJECTION: If no improvement for 8 generations
        if gen - last_improvement_gen >= 8:
            print(f"🌊 Generation {gen+1}: Diversity injection (stagnant for {gen - last_improvement_gen} gens)...")
            
            # Reinitialize worst 25% using biased initialization
            n_reinit = max(1, int(population_size * 0.25))
            
            # Get worst individuals
            worst_indices = list(range(population_size - n_reinit, population_size))
            
            # Reinitialize with volume bias
            new_individuals = create_biased_population(n_items, n_reinit, urunler, use_rotations)
            
            # Evaluate new individuals
            for new_ind in new_individuals:
                evaluate_de_individual(new_ind, urunler, palet_cfg, cache)
            
            # Replace worst with new
            for idx, new_ind in zip(worst_indices, new_individuals):
                population[idx] = new_ind
            
            # Re-sort
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Reset stagnation counter
            last_improvement_gen = gen
            print(f"   Reinitialized {n_reinit} individuals (worst 25%)")
        
        # ELITE REPAIR: Every 5 generations (now with 50 mutations)
        if (gen + 1) % 5 == 0:
            print(f"🔧 Generation {gen+1}: Applying enhanced elite repair (50 mutations)...")
            apply_elite_repair_to_population(population, urunler, palet_cfg, cache, n_elite=3)
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best if improved
            if population[0].fitness > best_individual.fitness:
                best_individual = population[0].copy()
                last_improvement_gen = gen
        
        # Record history
        current_best = population[0]
        history.append({
            'generation': gen + 1,
            'best_fitness': current_best.fitness,
            'mean_fitness': np.mean([ind.fitness for ind in population]),
            'best_pallets': current_best.palet_sayisi,
            'best_utilization': current_best.ortalama_doluluk,
            'improvements': improvements,
            'cache_hit_rate': cache.get_hit_rate()
        })
        
        # Progress report
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"📊 Generation {gen+1}/{generations}:")
            print(f"   Best Fitness: {current_best.fitness:.2f}")
            print(f"   Pallets: {current_best.palet_sayisi}, Utilization: {current_best.ortalama_doluluk*100:.1f}%")
            print(f"   Improvements: {improvements}, Cache Hit Rate: {cache.get_hit_rate()*100:.1f}%")
    
    # Final report
    print("\n" + "="*70)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"✅ Best Solution:")
    print(f"   Fitness: {best_individual.fitness:.2f}")
    print(f"   Pallets Used: {best_individual.palet_sayisi}")
    print(f"   Average Utilization: {best_individual.ortalama_doluluk*100:.1f}%")
    print(f"\n📊 Cache Statistics:")
    print(f"   Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"   Hit Rate: {cache.get_hit_rate()*100:.1f}%")
    print(f"   Total Evaluations Saved: {cache.hits}")
    print("="*70 + "\n")
    
    return best_individual, history


# ====================================================================
# UTILITY: CONVERT DE INDIVIDUAL TO CHROMOSOME (for compatibility)
# ====================================================================

def de_individual_to_chromosome(individual: DEIndividual, urunler: List[UrunData]):
    """
    Convert DEIndividual to Chromosome format for compatibility with existing code.
    
    Args:
        individual: DE individual
        urunler: Product list
    
    Returns:
        Chromosome-like object with compatible interface
    """
    from .chromosome import Chromosome
    
    # Decode order
    if individual.decoded_order is None:
        individual.decoded_order = decode_to_order(individual.priority_keys)
    
    # Create chromosome with decoded order
    chromosome = Chromosome(urunler=urunler, sira_gen=individual.decoded_order)
    chromosome.fitness = individual.fitness
    chromosome.palet_sayisi = individual.palet_sayisi
    chromosome.ortalama_doluluk = individual.ortalama_doluluk
    
    return chromosome


# ====================================================================
# MAIN ENTRY POINT (API-compatible with GA)
# ====================================================================

def optimize_with_de(
    urunler: List[UrunData],
    palet_cfg: PaletConfig,
    **kwargs
):
    """
    Main entry point compatible with existing GA API.
    
    Args:
        urunler: List of products
        palet_cfg: Pallet configuration
        **kwargs: Additional parameters (population_size, generations, etc.)
    
    Returns:
        Tuple of (best_chromosome, history)
    """
    # Run DE optimizer
    best_individual, history = run_de(urunler, palet_cfg, **kwargs)
    
    if best_individual is None:
        return None, history
    
    # Convert to Chromosome for compatibility
    best_chromosome = de_individual_to_chromosome(best_individual, urunler)
    
    return best_chromosome, history
