# Heat-Transfer-CUDA: Parallelizzazione CUDA di un Modello di Trasferimento di Calore

## Introduzione
Questo progetto, sviluppato nell'ambito del corso di *Massively Parallel Programming on GPUs* presso l'Università della Calabria, implementa una simulazione di trasferimento di calore in stato non stazionario su una griglia bidimensionale. Il sistema è modellato attraverso l'uso di Automi Cellulari (CA), sfruttando la potenza di calcolo delle GPU NVIDIA per l'elaborazione parallela di griglie dense.

L'obiettivo principale è lo sviluppo e il confronto critico di diversi kernel CUDA, passando da implementazioni "naive" basate su Global Memory a soluzioni ottimizzate che sfruttano la Shared Memory e tecniche di Tiling per minimizzare la latenza di accesso ai dati.

## Modello Matematico e Computazionale

La simulazione opera su un dominio computazionale suddiviso in celle quadrate uniformi. L'evoluzione temporale della temperatura segue una legge locale basata su un vicinato di Moore (raggio 1).

La funzione di transizione deterministica applicata ad ogni cella $(i, j)$ al passo temporale $t+1$ è definita dalla seguente equazione discreta:

$$
T_{i,j}^{t+1} = \frac{4(T_{i,j+1}^{t} + T_{i,j-1}^{t} + T_{i+1,j}^{t} + T_{i-1,j}^{t}) + T_{i+1,j+1}^{t} + T_{i+1,j-1}^{t} + T_{i-1,j-1}^{t}}{20}
$$

### Specifiche della Simulazione
* **Dominio:** Griglia di $2^8$ righe $\times$ $2^{12}$ colonne ($256 \times 4096$).
* **Iterazioni:** 10.000 passi temporali.
* **Condizioni al Contorno:** Le prime e le ultime 2 righe sono mantenute a 20°C (sorgente di calore), mentre il resto del dominio è inizializzato a 0°C.

## Strategie di Parallelizzazione

Il progetto esplora tre diverse strategie di gestione della memoria per valutare l'impatto sul throughput e sulla banda passante:

1.  **Global Memory Implementation (Baseline):**
    Parallelizzazione diretta che utilizza la CUDA Unified Memory. Ogni thread calcola il nuovo stato di una cella leggendo i vicini direttamente dalla memoria globale. Questa versione soffre di accessi ridondanti alla VRAM ed è limitata dalla banda di memoria (Memory Bound).

2.  **Tiled Implementation (Shared Memory):**
    Implementazione basata sul pattern di *Tiling*. I thread di un blocco caricano collaborativamente una porzione della matrice (tile) nella Shared Memory (on-chip L1 cache). Questo riduce drasticamente gli accessi alla memoria globale.

3.  **Tiled con Gestione Halo (Ghost Cells):**
    Ottimizzazione avanzata per la gestione degli stencil. I thread situati ai bordi del blocco (boundary threads) caricano in Shared Memory non solo i dati di competenza del blocco, ma anche le celle "halo" appartenenti ai blocchi adiacenti. Questo garantisce che tutti gli accessi necessari per il calcolo dello stencil avvengano in memoria a bassa latenza, eliminando la divergenza dovuta ad accessi misti Global/Shared.

## Analisi delle Prestazioni e Profiling

Parte integrante del progetto è l'assessment rigoroso delle prestazioni su hardware target (es. NVIDIA GTX 980 / architetture Maxwell o superiori). L'analisi include:

* **Benchmarking Configurazioni di Blocco:** Test esaustivi su configurazioni di blocco variabili (da $8\times8$ a $32\times32$) per identificare il setup che massimizza l'occupazione dell'hardware.
* **Warp Occupancy:** Analisi tramite profiler (nvprof/Nsight) per verificare l'utilizzo efficiente degli scheduler degli Streaming Multiprocessor.
* **Roofline Model Analysis:** Applicazione del modello Roofline per classificare i kernel. Calcolando l'Intensità Aritmetica (FLOP/Byte) e le prestazioni (GFLOP/s), i kernel vengono mappati sul grafico per determinare se le prestazioni sono limitate dalla capacità di calcolo (Compute Bound) o dalla larghezza di banda della memoria (Memory Bound).

## Struttura del Repository

* `main.cu`: Orchestrazione della simulazione, setup della memoria Unified e loop di benchmarking temporale.
* `kernel.cu`: Implementazione dei kernel CUDA (`updateNonTiled`, `updateTiledOptimized`, `tiled_wH`).
* `init.cu`: Kernel di inizializzazione della griglia e delle condizioni al contorno.
* `include/`: Header files per la definizione delle interfacce e delle costanti di sistema.
* `compila.sh`: Script di build automatizzato.

## Istruzioni per la Compilazione ed Esecuzione

### Prerequisiti
* NVIDIA CUDA Toolkit (nvcc).
* Compilatore C/C++ compatibile.

### Build
Utilizzare lo script fornito per compilare con ottimizzazioni host (`-O3`) e target architetturale specifico (es. `sm_52`):

```bash
./compila.sh
```

### Esecuzione
Lanciare l'eseguibile per avviare la suite di test:

```bash
./main
```

L'output fornirà i tempi di esecuzione per le diverse combinazioni di implementazione e dimensione dei blocchi, permettendo un confronto diretto dell'efficienza delle ottimizzazioni introdotte.


**Autori:** Christian Bruni, Francesco Tieri.
**Corso:** Massively Parallel Programming on GPUs - Università della Calabria.
**Anno Accademico:** 2025-2026.
