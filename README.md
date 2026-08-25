# Space Video Enhancer

MVP desktop para abrir um video local, comparar antes e depois quadro a quadro, ajustar a intensidade dos filtros e exportar o resultado sem upload.

## Requisitos

- Python 3.9+
- FFmpeg instalado e disponivel no `PATH`

No macOS com Homebrew:

```bash
brew install ffmpeg
```

No Windows, instale Python 3.9+ 64-bit e FFmpeg com a pasta `bin` no `PATH` (por exemplo `winget install --id Python.Python.3.12 -e` e `winget install --id Gyan.FFmpeg -e`). Confirme com `python --version` e `ffmpeg -version` em um PowerShell novo. Microsoft Visual C++ Redistributable 2015-2022 x64 so e necessario se aparecer erro de DLL ao abrir o app.

## Executar

```bash
./run.sh
```

O script cria a `.venv` se necessario, garante `pip`, instala as dependencias quando o projeto muda e inicia a interface.

Manualmente (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m space_video_enhancer
```

No Windows (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
python -m space_video_enhancer
```

Se a ativacao do ambiente virtual for bloqueada, rode antes `Set-ExecutionPolicy -Scope Process Bypass`. `python main.py` continua funcionando como atalho.

## Estrutura do projeto

```
space_video_enhancer/
  settings.py              # enums + snapshots imutaveis (RenderSettings) que a UI entrega ao pipeline
  video.py                 # leitura de frames com decodificacao sequencial; usado pelo worker de preview e pelo export
  pipeline/
    render.py              # render_frame(frame, settings, ...) — o unico pipeline (preview, miniaturas, export)
    filters.py             # denoise/contraste/gama/saturacao/temperatura/nitidez/escala (CPU ou OpenCL)
    presets.py             # presets tonais ("Sombras abertas", "Atenuar quentes", ...)
    denoise.py             # estimativa de ruido (modos Auto e Amostra)
    geometry.py            # rotacao, espelhamento e crop
    preview.py             # pedidos de preview/miniaturas e a caixa postal "o ultimo vence" do worker
    stabilization/         # estado, transformacoes, estimadores e orquestracao
  export/
    ffmpeg.py              # descoberta do ffmpeg, escolha de encoder e montagem do comando
    job.py                 # exportacao pura (sem Qt): decodifica -> render_frame -> stdin do ffmpeg
    worker.py              # adaptador QThread com sinais de progresso/termino
  ui/
    main_window.py         # orquestra toolbar, splitter, status bar e abas; produz RenderSettings; nao processa imagem
    tabs/                  # uma classe por aba: Cor, Filtros, Estabilizar, Geometria, Ruido, Exportar
    preview_worker.py      # QThread que decodifica e renderiza preview e miniaturas; a UI nunca bloqueia
    widgets/               # ComparePreview (+ SelectionOverlay), TransportBar, StatusPanel, LabeledSlider, FrameRangeSlider
    theme.qss              # estilo (icones em ui/icons/)
tests/                     # pytest; roda sem display (o pipeline nao depende de Qt)
```

Regras da arquitetura:

- O pipeline e puro: `render_frame(frame, settings, ...)` nao le widgets nem estado global. A UI le os widgets uma vez, monta um `RenderSettings` congelado e entrega ao pipeline, ao worker de export e as miniaturas.
- Estabilizacao automatica (translacao, similaridade, perspectiva) acumula o movimento entre frames em uma correcao que segue uma versao suavizada da trajetoria da camera (`correction_k = decay * (correction_{k-1} o delta_k)`, em `stabilization/stabilize.py`); alinhar cada frame ao anterior bruto so atrasaria o tremor em um frame. A suavizacao e uma constante de tempo (2 a 60 frames, escala log), a translacao da correcao e limitada a 8% do frame, e um frame sem estimativa mantem a correcao anterior. Camera lock segura a ultima correcao quando perde o alvo; o hibrido segue o movimento global enquanto perdido e volta suavemente ao lock. `benchmarks/stabilization_shake.py` mede o tremor residual em uma sequencia sintetica.
- Ordem fixa: estabilizacao -> filtros basicos (inclui escala) -> preset de imagem -> geometria -> crop. O crop e o ultimo passo, no mesmo sistema de coordenadas do preview rotacionado/espelhado, entao o retangulo desenhado e exatamente o exportado.
- Preview e miniaturas rodam em um `QThread` proprio (`ui/preview_worker.py`): a UI envia um `PreviewRequest` por ajuste e so o mais recente e renderizado; resultados antigos sao descartados por numero de geracao. Miniaturas so sao calculadas com a aba Filtros visivel.
- Nada de trabalho por frame na GUI thread: o worker e dono do `VideoSource` (a UI pede um `frame_index` e recebe o frame decodificado junto com o render, em `PreviewImages.source`, que serve de referencia para camera lock e para as miniaturas). A estimativa de denoise mostrada na aba Ruido vem do trace do ultimo render. Miniaturas sao renderizadas em pixels fisicos (`devicePixelRatio`) para ficarem nitidas em telas Retina.
- Exportacao roda em `QThread` (`export/worker.py`) sobre `run_export` (`export/job.py`). Cancelamento e falha sempre encerram o ffmpeg e apagam o arquivo parcial.
- O MP4 sai em `yuv420p` com `+faststart` e dimensoes pares, para tocar em QuickTime, iOS e decoders de hardware.
- Layout: toolbar (Abrir, Exportar, zoom, qualidade, metricas) + `QSplitter` com o preview a esquerda e as abas a direita + barra de status persistente (mensagens, indicador de render, progresso/cancelamento da exportacao, info do video). A largura minima do painel de abas e calculada a partir do conteudo real (`MainWindow._fit_settings_panel`), entao nada e cortado nem escondido atras de setas; `tests/test_ui_layout.py` verifica isso offscreen.
- Zoom do preview: "Ajustar" (padrao) preenche a area disponivel; os percentuais sao pixels do frame de saida por pixel logico da tela (100% = 1:1). Ctrl+0 ajusta, Ctrl++/Ctrl+- mudam de nivel.
- Teclado: Espaco, Home/End, setas e I/O controlam a reproducao e o corte, mas so quando o widget focado nao usa a tecla (um slider focado mantem as setas, um botao focado mantem o Espaco). Clicar no preview da o foco a ele; abrir um video tambem.
- Overlays: o crop aparece sempre que estiver ativo, mas so e editavel na aba Geometria; a amostra de ruido e o ponto focal aparecem apenas nas abas Ruido e Estabilizar, editaveis direto no preview (nao existe modo "editar").

## Desenvolvimento

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy space_video_enhancer
pytest
```

Os testes de exportacao ponta a ponta usam o `ffmpeg` do `PATH` e sao pulados se ele nao existir.

## Telemetria e logs

O app grava eventos estruturados (uma linha JSON por evento) em `~/.space-video-enhancer/logs/app.jsonl`, com rotacao em 10 MB x 5. Nada sai do computador: os logs servem para medir e melhorar algoritmos e interface. Nomes de arquivo aparecem sem o diretorio.

Variaveis de ambiente: `SVE_LOG_DIR` (pasta), `SVE_LOG_LEVEL` (`INFO` padrao; `DEBUG` inclui `preview.render` durante a reproducao), `SVE_LOG_STDERR=1` (espelha no terminal). O menu `Ajuda > Abrir pasta de logs` abre a pasta; `Exibir > Metricas` (Ctrl+M, ou o botao Metricas na toolbar) mostra um HUD com tempo de render por estagio, fila, decode, estabilizacao e backend.

Eventos principais:

| Evento | Quando | Campos uteis |
|---|---|---|
| `app.start` / `app.stop` | inicio/fim | versoes, CPU, OpenCL (dispositivo), ffmpeg |
| `video.open` | abrir video | dimensoes, fps, frames, codec, `open_ms` |
| `settings.snapshot` / `settings.change` | primeiro render de um video / cada ajuste | `changes`: `{"filters.contrast": [50, 62]}` |
| `preview.render` | cada render do preview (playback em DEBUG) | `total_ms`, `decode_ms`, `decode_sequential`, `stages` (`decode`, `stabilize`, `context`, `filters`, `filters.denoise`, `preset`, `geometry`), `queue_wait_ms`, `dropped_total`, notas do pipeline |
| `playback.start` / `playback.stop` | reproducao | `effective_fps`, `frames_shown`, `seeks`, `sequential_reads`, `decode_ms` |
| `export.start` / `export.progress` / `export.finish` / `export.cancelled` / `export.failed` | exportacao | `fps`, `elapsed_s`, `render` (estatisticas por estagio), `stabilization` (histograma de estimadores/resultados), `encoder`, `backend`, `output_bytes`, `stderr_tail` |
| `ui.*` | interacoes (`ui.tab`, `ui.preset`, `ui.render_mode`, `ui.preview_zoom`, `ui.clip_range`, `ui.hud`) | |
| `session.summary` | fechar a janela | renders, descartados, exports, erros |
| `preview.error`, `export.error`, `app.unhandled`, `qt.message` | falhas | `exception`, `settings` |

Notas do pipeline registradas em cada render (`preview.render`) e agregadas no export: `stabilization.mode`, `stabilization.estimator` (`lk_translation`, `phase_correlate`, `lk_similarity`, `lk_perspective`, `orb_masked`, `orb_unmasked`, `template_match`, `primed`, `sequence_break`, `cached`, `none`), `stabilization.result` (`applied`, `negligible`, `no_transform`), `stabilization.path` (`start`, `follow`, `held`, `lock`, `reacquire`, `dead_reckoning`), `stabilization.inliers`, `stabilization.tracked_points`, `stabilization.rejected` (`too_few_inliers`, `unreasonable_similarity`, `too_few_perspective_inliers`, `unreasonable_perspective`, `accumulated`), `stabilization.translation_px`, `denoise.mode`, `denoise.h`, `filters.backend`.

Analise rapida com pandas:

```python
import pandas as pd

df = pd.read_json("~/.space-video-enhancer/logs/app.jsonl", lines=True)
renders = df[df.event == "preview.render"]
print(pd.json_normalize(renders.stages).describe())
```

## O que o MVP faz

- seleciona um video local e navega pelos frames com slider, botoes de transporte e reproducao quadro a quadro
- zoom do preview por botoes e percentual, com modos de qualidade (alta, balanceado, rapido) e pulo de frames na reproducao
- comparacao antes/depois com divisor arrastavel
- abas tematicas: cor e detalhe, filtros (presets com miniaturas do frame atual), estabilizacao, geometria, ruido e midia
- presets de melhoria e reset por slider para o valor do preset ativo
- denoise desativado, manual, auto ou por amostra escolhida no preview; nitidez, contraste, brilho, gama, saturacao, temperatura e escala de saida
- estabilizacao por translacao, similaridade, perspectiva, camera lock por regiao focal ou hibrida, com intensidade, suavizacao temporal, crop automatico e zoom out
- filtros tonais para sombras, realces, contraste local e atenuacao de cores quentes/frias
- crop interativo com presets de proporcao e modo livre; rotacao 90/180/270 e espelhamento
- corte temporal por frame inicial/final com marcadores no slider
- exportacao local com barra de progresso e cancelamento, usando FFmpeg apenas para encode e mux do audio original
- tenta encoders de hardware (NVENC, VAAPI) e OpenCL quando disponiveis, com fallback automatico para CPU/libx264

## Aceleracao por GPU

O processamento em OpenCV pode usar OpenCL (T-API) e o encode pode usar NVENC ou VAAPI. Na pratica o ganho depende do video, dos filtros ativos e do driver: o denoise ganha um pouco, os filtros leves nao. Se OpenCL ou o encoder de hardware falharem, o app cai para CPU e `libx264` automaticamente. No Linux, VAAPI exige um dispositivo como `/dev/dri/renderD128`; para NVIDIA, o FFmpeg precisa ter sido compilado com `nvenc`.
