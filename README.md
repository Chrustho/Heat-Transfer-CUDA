# 🔥 Heat-Transfer-CUDA

Simulazione parallela del trasferimento di calore su GPU NVIDIA mediante Automi Cellulari.

> Progetto sviluppato per il corso di **Massively Parallel Programming on GPUs** — Università della Calabria, A.A. 2025-2026

---

## Overview

Implementazione e confronto di quattro kernel CUDA per la simulazione di diffusione termica su una griglia 2D, con focus sull'ottimizzazione degli accessi in memoria.

| Kernel | Strategia | Caratteristiche |
|--------|-----------|-----------------|
| `updateGlobal` | Global Memory | Baseline naive |
| `updateTiled` | Shared Memory | Tiling base |
| `updateTiledPadding` | Shared + Padding | Riduce bank conflicts |
| `updateTiled_wH` | Shared + Halo | Ghost cells per stencil completo |

---

## Modello Fisico

Diffusione termica con vicinato di Moore (raggio 1):

```
T(i,j)ᵗ⁺¹ = [4·(N + S + E + W) + NW + NE + SW + SE] / 20
```

**Configurazione simulazione:**
- Griglia: `256 × 4096` celle
- Iterazioni: `10.000` steps
- Boundary: righe superiori/inferiori fisse a 20°C

---

## Quick Start

```bash
# Compilazione
./compila.sh

# Esecuzione benchmark
./main
```

**Requisiti:** CUDA Toolkit, GPU con compute capability ≥ 5.2

---

## Struttura

```
├── main.cu          # Entry point e benchmarking
├── kernel.cu        # Implementazioni kernel
├── init.cu          # Inizializzazione griglia
├── include/
│   ├── kernel.cuh
│   ├── init.cuh
│   └── utility.h
└── compila.sh       # Build script
```

---

## Benchmark

Il programma testa automaticamente configurazioni di blocco `8×8`, `16×16`, `32×32` (e combinazioni) su più run, riportando il tempo migliore per ciascuna.

```
Tempo esecuzione blocco 16 x 16: 245.32 ms
Miglior tempo per blocco 16 x 16: 243.18 ms
```

---

## Autori

**Christian Bruni** · **Francesco Tieri**
