# Plano: estabilização unificada (automática única + travar objeto multi-ponto)

Estado: **Wave 1 feita (2026-09-01)**; Waves 2-4 planejadas. Motivação (Maicon, 2026-09-01): são modos demais para escolher; as
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

## Wave 2 — "Travar objeto" multi-ponto (funde camera lock + híbrido)

- O usuário marca **1 a 4 regiões** no preview (retângulos verdes numerados, podem estar em cantos
  opostos). Cada região vira um track independente (LK + re-âncora ORB periódica, o motor do lock
  atual) com **confiança por frame** (residual mediano + contagem de pontos).
- A correção é estimada do conjunto: similaridade ajustada sobre os centros/pontos das regiões
  vivas, ponderada pela confiança. **Uma região perdida não congela nada** — as outras seguram o
  quadro; a perdida reaquisita pelo template como hoje. Com todas perdidas, dead-reckoning do
  movimento global (o comportamento do híbrido) até reencontrar qualquer uma.
- Duas regiões em cantos opostos determinam também rotação e escala do alvo — hoje o lock só tem
  uma âncora e sofre com isso.
- UI: clicar "Adicionar região" cria mais um retângulo; overlay numera; a lista não precisa de
  painel (menu de contexto no retângulo remove). CAMERA_LOCK/HYBRID saem do combo (mesma regra de
  legado da Wave 1).
- Aceitação: cena sintética com alvo + oclusões alternadas (uma região sempre viva) → residual
  contra a referência estável durante as oclusões (hoje o lock salta no "hold").

## Wave 3 — objeto + malha ("cadeia de pixels-chave")

- A fusão final: as regiões-âncora definem **o caminho de referência** (o que deve ficar parado) e
  a malha corrige o resto do quadro **em volta** dessa âncora (paralaxe e jello por região).
- Implementação: hoje a correção da malha suaviza cada vértice contra o próprio caminho; nesta wave
  o alvo do caminho passa a ser a trajetória (suavizada, ou fixa = lock) das âncoras, e cada
  vértice corrige a diferença entre o seu caminho e esse alvo. Âncoras opostas dão escala/rotação
  do alvo de graça (Wave 2).
- A oscilação/rolling shutter continua implícita na malha (medido: correções extras brigam).
- Aceitação: vídeo real seguindo um objeto — o objeto fica parado (métrica do lock) E o fundo não
  "respira" (trapézio/jello do comparador) ao mesmo tempo, o que nenhum modo atual consegue.

## Wave 4 — polimento e remoção do legado

- Migração silenciosa dos enums legados ao salvar projeto (AUTO/SIMILARITY/PERSPECTIVE → MESH;
  CAMERA_LOCK/HYBRID → objeto com 1 região) e remoção do código morto depois de um período.
- Presets de margem por cenário (caminhada 8–12%, câmera parada 5%).

Ordem recomendada: Wave 1 (pequena) → Wave 2 (média) → Wave 3 (grande, o prêmio) → Wave 4.
