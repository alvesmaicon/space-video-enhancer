# Telemetria e logs

O app grava eventos estruturados (uma linha JSON por evento) em `~/.space-video-enhancer/logs/app.jsonl`, com rotacao em 10 MB x 5. Nada sai do computador: os logs servem para medir e melhorar algoritmos e interface. Nomes de arquivo aparecem sem o diretorio.

Variaveis de ambiente: `SVE_LOG_DIR` (pasta), `SVE_LOG_LEVEL` (`INFO` padrao; `DEBUG` inclui `preview.render` durante a reproducao), `SVE_LOG_STDERR=1` (espelha no terminal). O menu `Ajuda > Abrir pasta de logs` abre a pasta; `Exibir > Metricas` (Ctrl+M, ou o botao Metricas acima do preview) mostra um HUD com tempo de render por estagio, fila, decode, estabilizacao e backend. `SVE_PREVIEW_AUDIO=0` desliga o som do preview.

Eventos principais:

| Evento | Quando | Campos uteis |
|---|---|---|
| `app.start` / `app.stop` | inicio/fim | versoes, CPU, OpenCL (dispositivo), ffmpeg |
| `video.open` | abrir ou adicionar um arquivo | dimensoes, fps, frames, codec, `open_ms` |
| `sequence.add` / `sequence.add_refused` / `sequence.split` / `sequence.remove` / `sequence.move` | edicao da sequencia de clipes | nome, `clips`, `frames`, `reason` |
| `settings.snapshot` / `settings.change` | primeiro render de um video / cada ajuste | `changes`: `{"filters.contrast": [50, 62]}` |
| `preview.render` | cada render do preview (playback em DEBUG) | `total_ms`, `decode_ms`, `decode_sequential`, `stages` (`decode`, `stabilize`, `context`, `filters`, `filters.denoise`, `preset`, `geometry`), `queue_wait_ms`, `dropped_total`, notas do pipeline |
| `playback.start` / `playback.stop` | reproducao | `speed`, `audio`, `effective_fps`, `frames_shown`, `seeks`, `sequential_reads`, `decode_ms`, `audio_seconds`, `stabilization` (histograma de `stabilization.result` durante a reproducao) |
| `audio.preview_device` / `audio.preview_disabled` / `audio.preview_decode` / `audio.preview_skipped` | som do preview | dispositivo, motivo, `tempo`, `seconds`, `megabytes`, `elapsed_ms` |
| `export.start` / `export.progress` / `export.finish` / `export.cancelled` / `export.failed` | exportacao | `fps`, `elapsed_s`, `render` (estatisticas por estagio), `stabilization` (histograma de estimadores/resultados), `encoder`, `backend`, `output_bytes`, `stderr_tail` |
| `ui.*` | interacoes (`ui.tab`, `ui.preset`, `ui.render_mode`, `ui.preview_zoom`, `ui.clip_range`, `ui.cuts`, `ui.clip_active`, `ui.hud`, `ui.motion_vectors`, `ui.stabilization_reference`, `ui.undo`, `ui.redo`, `ui.project`, `ui.monitor`) | |
| `session.summary` | fechar a janela | renders, descartados, exports, erros |
| `preview.error`, `export.error`, `app.unhandled`, `qt.message` | falhas | `exception`, `settings` |

Notas do pipeline registradas em cada render (`preview.render`) e agregadas no export: `stabilization.mode`, `stabilization.estimator` (`lk_translation`, `phase_correlate`, `lk_similarity`, `lk_perspective`, `lk_lock`, `orb_anchor`, `lk_seed`, `template_match`, `primed`, `sequence_break`, `cached`, `none`), `stabilization.hypotheses` (pares `[inliers, extensao em celulas]` de cada hipotese de movimento) e `stabilization.chosen` (indice da escolhida), `stabilization.model` (`similarity`/`homography` no modo perspectiva), `stabilization.lock_tracked`, `stabilization.lock_blur_match` (sigma aplicado a referencia), `stabilization.lock_residual_px`, `stabilization.lock_model`, `stabilization.result` (`applied`, `negligible`, `no_transform`), `stabilization.path` (`start`, `follow`, `held`, `lock`, `reacquire`, `dead_reckoning`), `stabilization.inliers`, `stabilization.tracked_points`, `stabilization.rejected` (`too_few_inliers`, `unreasonable_similarity`, `too_few_perspective_inliers`, `unreasonable_perspective`, `accumulated`), `stabilization.translation_px`, `denoise.mode`, `denoise.sigma` (ruido estimado em luma e cor), `denoise.h`, `filters.backend` (`opencl`, `cpu`, `cpu_no_denoise`), `exposure.current` (medianas B/G/R da regiao no frame) e `exposure.gains` (ganhos aplicados pela trava de exposicao).

Analise rapida com pandas:

```python
import pandas as pd

df = pd.read_json("~/.space-video-enhancer/logs/app.jsonl", lines=True)
renders = df[df.event == "preview.render"]
print(pd.json_normalize(renders.stages).describe())
```
