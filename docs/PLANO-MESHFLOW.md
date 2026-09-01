# Plano: estabilização por malha (MeshFlow)

Estado: **planejado** (nenhum código ainda). Motivação: nos vídeos de caminhada, todos os modos
globais deixam deformação de paralaxe — objetos a distâncias diferentes se movem em velocidades
diferentes, e nenhuma transformação única (nem homografia) explica o quadro inteiro. Resultado
negativo já medido (2026-09-01): amortecer a parte não-similar da homografia (constante de tempo de
12 frames) NÃO ajuda — o trapézio do resultado voltou de 1,9 para 7,7 px/par no vídeo real, porque o
keystone rápido é distorção verdadeira (7,2 px/par no fonte) e não ruído do estimador (≤1,9 px/par);
qualquer low-pass troca correção real por quase nada. O que sobra da perspectiva pura é paralaxe, e a
solução clássica sem ML é o MeshFlow (Liu et al., 2016, "MeshFlow: Minimum Latency Online Video
Stabilization"): cada região da imagem ganha o seu próprio caminho de câmera.

## Ideia central

1. Uma malha regular de vértices (o paper usa 16×16) é sobreposta ao quadro.
2. Os vetores de movimento esparsos (features rastreadas) são transferidos aos vértices próximos por
   dois filtros de mediana (f1: mediana dos vetores de cada feature na vizinhança; f2: mediana entre
   vértices) — robustez a outliers e a objetos em movimento sem custo de fluxo denso.
3. Cada vértice acumula o seu caminho no tempo (como o nosso `MotionPath`, só que um por vértice).
4. Os caminhos são suavizados em conjunto: termo temporal (caminho liso) + termo espacial (vértices
   vizinhos não podem divergir demais — é isso que evita a "gelatina" na saída).
5. O warp de cada frame vem do deslocamento dos vértices (interpolação bilinear por célula →
   um `cv2.remap`, como o unskew do rolling shutter já faz).

## O que o código atual já dá de graça

| Peça | Reuso |
|---|---|
| `global_motion.grid_features` + `track_points` | os vetores esparsos por célula, com forward-backward check e memória de célula (objetos em movimento já são filtrados) |
| Padrão `WobbleTrack`/`analyze_wobble` | a análise offline por clipe em thread própria, com progresso, cancelamento, cache por (arquivo, intervalo) e reanálise no export — o `MeshPath` segue o mesmo molde |
| `rolling_shutter.correct` | o esqueleto do render por `cv2.remap` com mapas por posição |
| `benchmarks/diagnose_shake.py` + comparador de variantes | as métricas de aceitação (trapézio, jello, residual) já medem exatamente o que a malha precisa melhorar |
| Export paralelo | o `MeshPath` é imutável e por frame → `requires_sequential_processing = False` |

## Fases

### Fase 1 — análise (`pipeline/stabilization/mesh.py`)
`analyze_mesh_path(read, first, last, ...) -> MeshPath` — uma passada de rastreamento no tamanho de
análise. Por par de frames: features → vetores → mediana por vértice (janela = célula e vizinhas) →
acumula por vértice. `MeshPath` imutável: `(frames, linhas, colunas, 2)` float32 em fração do quadro
(600 frames × 17×31 vértices ≈ 2,5 MB). Vértice sem dados no par: herda a mediana dos vizinhos
(f2); sem dado nenhum, herda o movimento global (que já calculamos).

### Fase 2 — suavização offline
Somos offline (o paper é online com buffer): minimizar
`Σ_v Σ_t |P_v(t) − C_v(t)|² + λ_t Σ_t |C_v(t) − C_v(t±1)|² + λ_s Σ_viz |C_v(t) − C_w(t)|²`
por Jacobi (poucas iterações convergem; ou separável: média móvel de Hann no tempo + blur espacial
leve por iteração). `λ_t` vem do slider de suavização existente (mesma constante de tempo);
`λ_s` fixo, calibrado no vídeo real de caminhada.

### Fase 3 — render
`mesh_warp(frame, correction_do_frame)` — deslocamentos dos vértices → `cv2.remap` bilinear
(mapas em float32, cache do grid base por tamanho). Crop: a margem fixa de 5% de hoje continua;
deslocamento de vértice clampado à margem (o teto que os modos globais já têm, agora por vértice).

### Fase 4 — integração
`StabilizationMode.MESH` ("Malha — paralaxe de caminhada"). UX igual à oscilação: marcar o modo
dispara a análise em segundo plano com progresso no rótulo; preview renderiza sem correção até a
trilha chegar; export analisa clipes sem trilha antes do primeiro frame. Persistência: nada novo no
.svep (a trilha é recomputável, como as referências do lock).

### Fase 5 — validação de aceitação
No comparador (scratchpad → promover a `benchmarks/compare_stabilization.py`):
- trapézio ≤ perspectiva amortecida, e residual de translação ≤ rotação+zoom;
- inspeção visual de "gelatina" de malha (o risco clássico: λ_s baixo demais);
- custo: análise ~1× decode do clipe (igual à oscilação), render alvo ≤ 20 ms/frame em 4K no M1.

## Riscos conhecidos

- **Céu/paredes lisas**: vértices sem features → depender da propagação espacial; se faltar, cair
  para o movimento global naquela região (nunca inventar).
- **Objetos grandes em movimento**: a memória de célula existente pesa contra eles, mas na malha um
  objeto que cobre células inteiras pode puxar seus vértices — mitigação: descartar vetores fora da
  hipótese de câmera escolhida pelo estimador global (já sabemos quais são os inliers).
- **Bordas**: deslocamentos por vértice revelam borda de formas diferentes por região; o clamp por
  vértice à margem fixa resolve, ao custo de menos correção nas bordas.
