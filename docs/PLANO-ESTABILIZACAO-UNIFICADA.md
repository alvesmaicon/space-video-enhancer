# Plano: estabilização unificada (automática única + travar objeto multi-ponto)

Estado: **Waves 1, 2 e 3 feitas (2026-09-01)**; Wave 4 planejada. Motivação (Maicon, 2026-09-01): são modos demais para escolher; as
automáticas deveriam ser uma só, e o "travar objeto" deveria sobreviver à perda do alvo usando
vários pontos de rastreio — inclusive em cantos opostos do quadro — combinado com o resto da
máquina (malha, oscilação).

Resposta à pergunta "unificar aumenta a pipeline?": **não** — a malha (MeshFlow) já é um
superconjunto medido das automáticas globais (translação, rotação, zoom e keystone viram
translação variável por vértice; tabela em PLANO-MESHFLOW.md). Unificar *remove* caminhos de
código com o tempo; o único custo novo é o fallback ao vivo da Wave 1.

Alvo final de UI: **3 modos** — Automática · Travar objeto · Manual (keyframes).

## Wave 1 — "Automática" única (consolidação) — FEITA (2026-09-01)

- "Automática" = malha. Enquanto a análise do clipe roda em segundo plano, o preview cai num
  fallback ao vivo (a trajetória suavizada de translação de hoje, o modo AUTO) para o usuário já
  ver estabilização; quando a trilha chega, a correção cruza para a malha com uma rampa curta
  (~meio segundo) para o quadro não pular. O export sempre usa a malha (a análise roda antes).
- AUTO/SIMILARITY/PERSPECTIVE saem do combo. O enum continua no motor: projeto antigo carrega e
  renderiza igual (o combo mostra a entrada legada apenas quando o clipe carregado a usa).
- Como ficou: `apply_stabilization` estima o caminho de translação (o AUTO) quando o campo da
  malha ainda não existe e o funde no próprio campo durante o handoff
  (`MESH_HANDOFF_FRAMES = 15`, um remap só — o campo é fração, a translação vira constante
  somada). O handoff só arma quando o fallback rodou na mesma sequência (`state.mesh_handoff`);
  estado novo (worker de export, scrub, frame pausado) aplica o campo inteiro de uma vez, então
  o export paralelo não mudou (workers recebem `state=None`). Entrada legada no combo:
  `LEGACY_MODE_OPTIONS` + `_sync_legacy_mode_entry` (insere no `apply()` de um clipe legado,
  remove quando o usuário troca de modo).
- Aceitação verificada em teste: o fallback corta o tremor pela metade já nos primeiros frames
  (`test_mesh_mode_falls_back_to_the_live_path_until_the_field_arrives`), o campo entra em fade
  sem pulo e satura no valor exato (`test_mesh_field_fades_in_from_the_live_path_when_it_lands`),
  e estado novo/pausa aplicam o campo direto (`test_mesh_handoff_settles_on_the_field_when_paused_or_fresh`).

## Wave 2 — "Travar objeto" multi-ponto (funde camera lock + híbrido) — FEITA (2026-09-01)

- O usuário marca **1 a 4 regiões** no preview (retângulos verdes numerados, podem estar em cantos
  opostos). Cada região vira um track independente (LK + re-âncora ORB periódica, o motor do lock
  atual) com **confiança por frame** (residual mediano + contagem de pontos).
- A correção é estimada do conjunto: similaridade ajustada sobre os centros/pontos das regiões
  vivas, ponderada pela confiança. **Uma região perdida não congela nada** — as outras seguram o
  quadro; a perdida reaquisita pelo template como hoje. Com todas perdidas, dead-reckoning do
  movimento global (o comportamento do híbrido) até reencontrar qualquer uma.
- Duas regiões em cantos opostos determinam também rotação e escala do alvo — hoje o lock só tem
  uma âncora e sofre com isso.
- Como ficou: `lock.region_match` extrai a máquina de uma região (LK → re-âncora ORB/seed →
  template como último recurso) devolvendo as correspondências; `lock.object_lock_transform`
  reúne os pontos das regiões vivas e ajusta **uma similaridade** (RANSAC descarta pontos de uma
  região que derivou; contagem de pontos pondera naturalmente; homografia desativada de propósito —
  clusters pequenos condicionam mal). Um template-match vira **uma** correspondência (o centro),
  que restringe só translação — nada de rotação fabricada. Fluxo de topo = o `hybrid` com o
  estimador multi-região (`StabilizationMode.OBJECT`), dead-reckoning intacto. UI: botões
  "Adicionar/Remover região" + overlays numerados (`focusRegionChanged(index, rect)`);
  CAMERA_LOCK/HYBRID viraram entradas "(legado)". Regiões persistem no `.svep`
  (`focus_regions`); a lista da UI é fiel ao clipe e a 1ª região é semeada só ao entrar no modo
  (semear no load sujava o documento ao abrir).
- Aceitação (tests/test_object_lock.py): oclusões alternadas com uma região sempre viva →
  residual < 1,5 px durante as oclusões; duas regiões opostas recuperam rotação de 2° com erro
  < 0,3°; todas perdidas → "dead_reckoning"; round-trip do projeto.

## Wave 3 — objeto + malha ("cadeia de pixels-chave") — FEITA (2026-09-01)

- A fusão final: as regiões-âncora definem **o caminho de referência** (o que deve ficar estável) e
  a malha corrige o resto do quadro **em volta** dessa âncora (paralaxe e jello por região).
- Como ficou: o modo **Travar objeto passou a ser apoiado na malha** (`needs_mesh_analysis` cobre
  OBJECT — uma análise serve os dois modos). O caminho das âncoras é **amostrado da própria trilha
  da malha** (`_anchor_similarities`: 5 pontos por região — centro + cantos — ajustados numa
  similaridade por frame, em unidades de altura para rotação/escala serem métricas), então arrastar
  região ou mover slider recalcula do cache em ms, sem re-rastrear. A montagem
  (`corrections_for_object`): correção da âncora = `pose_suavizada ∘ pose_atual⁻¹` avaliada em cada
  vértice (rotação/escala giram em torno do alvo), caminhos ajustados = trilha + campo da âncora, e
  a suavização padrão da malha roda sobre eles — a âncora cai EXATAMENTE na sua trajetória suavizada
  (suavizar o já-suave é no-op ali) e o fundo mantém a paralaxe lenta relativa ao alvo. Sem regiões
  degenera na malha pura (campos idênticos, testado). O v8 ensinou: pose FIXA satura em caminhada —
  por isso o alvo segue a trajetória suavizada (slider de suavização controla o quão "travado").
- Fallback ao vivo = o lock multi-região da Wave 2 (com handoff da Wave 1, agora ciente de
  rotação: a correção ao vivo vira campo por vértice no crossfade). Export analisa antes e continua
  paralelo (`requires_sequential_processing` False); oscilação suprimida também no OBJECT (a malha
  absorve).
- Overlay de diagnóstico (Ctrl+Shift+M): quando o campo assume, nada mais é rastreado ao vivo —
  o overlay passa a mostrar o que a análise mediu (`MeshCorrections.motion`/`anchor_points`):
  pontos das âncoras (5 por região, amostrados da trilha) no modo objeto, vetores por vértice na
  Automática. Durante fallback/handoff continua o rastreio ao vivo, como antes.
- Aceitação: sintético — âncora cai na trajetória suavizada (erro < 0,2% do quadro), rotação rápida
  de 1,5° cancelada (> 85%), paralaxe do fundo preservada. Vídeo real (caminhada, regiões do v8):
  alvo rápido 10,07 → 2,72 px (malha pura: 2,60 — empate no alvo estático), global 2,52 px (malha:
  2,85) e trapézio 5,04 px/par (malha: 6,40) — o objeto agora é ≥ malha nesse footage, em vez do
  1,11% vs 0,71% do lock de pose fixa do v8. O caso diferenciador (alvo que se move independente do
  fundo) está coberto pelos sintéticos; medir em footage de alvo real quando houver.

## Wave 4 — polimento e remoção do legado

- Migração silenciosa dos enums legados ao salvar projeto (AUTO/SIMILARITY/PERSPECTIVE → MESH;
  CAMERA_LOCK/HYBRID → objeto com 1 região) e remoção do código morto depois de um período.
- Presets de margem por cenário (caminhada 8–12%, câmera parada 5%).

Ordem recomendada: Wave 1 (pequena) → Wave 2 (média) → Wave 3 (grande, o prêmio) → Wave 4.
