# Space Video Enhancer

MVP desktop para abrir um video local, comparar antes e depois quadro a quadro, ajustar a intensidade dos filtros e exportar o resultado sem upload.

## Requisitos

- Python 3.9+
- FFmpeg instalado e disponivel no `PATH`

No macOS com Homebrew:

```bash
brew install ffmpeg
```

No Ubuntu/Debian (veja tambem "Linux com GPU" mais abaixo para OpenCL e NVENC):

```bash
sudo apt install ffmpeg python3-venv ocl-icd-libopencl1 clinfo
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
    tabs/                  # uma classe por aba: Cor, HSL, Filtros, Estabilizar, Geometria, Ruido, Exportar
    preview_worker.py      # QThread que decodifica e renderiza preview e miniaturas; a UI nunca bloqueia
    widgets/               # ComparePreview (+ SelectionOverlay), TransportBar, TimelineWidget, StatusPanel, LabeledSlider
    theme.qss              # estilo (icones em ui/icons/)
tests/                     # pytest; roda sem display (o pipeline nao depende de Qt)
```

Regras da arquitetura:

- O pipeline e puro: `render_frame(frame, settings, ...)` nao le widgets nem estado global. A UI le os widgets uma vez, monta um `RenderSettings` congelado e entrega ao pipeline, ao worker de export e as miniaturas.
- Estabilizacao automatica (translacao, similaridade, perspectiva) acumula o movimento entre frames em uma correcao que segue uma versao suavizada da trajetoria da camera (`correction_k = decay * (correction_{k-1} o delta_k)`, em `stabilization/stabilize.py`); alinhar cada frame ao anterior bruto so atrasaria o tremor em um frame. A suavizacao e uma constante de tempo (2 a 60 frames, escala log), a translacao da correcao e limitada a 8% do frame, um frame sem estimativa mantem a correcao anterior, e saltos de ate 12 frames (frames descartados pelo preview quando o render e mais lento que a reproducao) continuam a sequencia em vez de reinicia-la. Camera lock segura a ultima correcao quando perde o alvo e reutiliza as features ORB do frame de referencia (`estimators.reference_features`, cache por identidade do array + regiao + escala); o hibrido segue o movimento global enquanto perdido e volta suavemente ao lock. O crop automatico recorta uma margem fixa igual a esse teto (5% por lado) em todos os frames — inclusive nos que nao receberam correcao — entao o zoom nao respira; a margem so cresce se um camera lock exceder o teto. `benchmarks/stabilization_shake.py` mede o tremor residual em uma sequencia sintetica.
- Ordem fixa: estabilizacao -> filtros basicos (inclui escala) -> preset de imagem -> geometria -> crop. O crop e o ultimo passo, no mesmo sistema de coordenadas do preview rotacionado/espelhado, entao o retangulo desenhado e exatamente o exportado.
- Preview e miniaturas rodam em um `QThread` proprio (`ui/preview_worker.py`): a UI envia um `PreviewRequest` por ajuste e so o mais recente e renderizado; resultados antigos sao descartados por numero de geracao. Miniaturas so sao calculadas com a aba Filtros visivel.
- Denoise no preview: com o denoise ligado, scrub e reproducao renderizam com o lado maior limitado a 640 px (o custo do NLMeans e proporcional aos pixels) e o frame em que o usuario parou recebe um render completo 250 ms depois (`reason=idle`). A forca automatica vem do ruido medido (MAD da resposta do Laplaciano, `pipeline/denoise.py`), nao do desvio padrao da imagem; `h = 0.8 * sigma`.
- Nada de trabalho por frame na GUI thread: o worker e dono do `VideoSource` (a UI pede um `frame_index` e recebe o frame decodificado junto com o render, em `PreviewImages.source`, que serve de referencia para camera lock e para as miniaturas). A estimativa de denoise mostrada na aba Ruido vem do trace do ultimo render. Miniaturas sao renderizadas em pixels fisicos (`devicePixelRatio`) para ficarem nitidas em telas Retina.
- Exportacao roda em `QThread` (`export/worker.py`) sobre `run_export` (`export/job.py`). Cancelamento e falha sempre encerram o ffmpeg e apagam o arquivo parcial.
- O MP4 sai em `yuv420p` com `+faststart` e dimensoes pares, para tocar em QuickTime, iOS e decoders de hardware.
- Layout: toolbar (Abrir, Exportar, zoom, qualidade, metricas) + `QSplitter` com o preview a esquerda e as abas a direita + barra de status persistente (mensagens, indicador de render, progresso/cancelamento da exportacao, info do video). A largura minima do painel de abas e calculada a partir do conteudo real (`MainWindow._fit_settings_panel`), entao nada e cortado nem escondido atras de setas; `tests/test_ui_layout.py` verifica isso offscreen.
- Zoom do preview: "Ajustar" (padrao) preenche a area disponivel; os percentuais sao pixels do frame de saida por pixel logico da tela (100% = 1:1). Ctrl+0 ajusta, Ctrl++/Ctrl+- mudam de nivel.
- Teclado: Espaco, Home/End, setas, I/O e X controlam a reproducao e os cortes, mas so quando o widget focado nao usa a tecla (um slider focado mantem as setas, um botao focado mantem o Espaco). Clicar no preview da o foco a ele; abrir um video tambem.
- Overlays: o crop aparece sempre que estiver ativo, mas so e editavel na aba Geometria; a amostra de ruido e o ponto focal aparecem apenas nas abas Ruido e Estabilizar, editaveis direto no preview (nao existe modo "editar").
- Cortes internos (`timeline.py`): alem do inicio/fim (I/O), a aba Exportar guarda uma lista de trechos removidos — X marca inicio e fim no frame atual, ou digite os tempos (`mm:ss`, `h:mm:ss`, `mm:ss.ff`). Os trechos aparecem em vermelho na linha do tempo, o playback pula por cima deles e o export le so os trechos mantidos (com seek entre eles) concatenando o audio correspondente (`atrim`/`concat`); fontes sem audio saem mudas em vez de falhar.
- Sequencia (`sequence.py`): o modelo e uma `Sequence` de `Clip`s — cada clipe e um trecho (`range`) de um `ClipSource` (arquivo) com cortes e `RenderSettings` proprios (ajustes por trecho = dividir um clipe em dois com `split`). Dois eixos: *frames da linha do tempo* (todos os clipes lado a lado, cortes incluidos e navegaveis) e *frames exportados* (linha do tempo menos os cortes). `locate`/`timeline_frame` convertem entre eles, `with_timeline_cuts` distribui cortes desenhados na linha do tempo pelos clipes que atravessam. Tudo imutavel: cada operacao devolve outra `Sequence`. O export (`export/job.py`) percorre `export_segments()` com um `VideoCapture` por arquivo, um contexto de denoise e um estado de estabilizacao por clipe, encaixa cada frame no tamanho do primeiro (letterbox) e manda o ffmpeg concatenar o audio de cada trecho (`AudioSegment`; silencio para fontes sem audio, `aresample`/`aformat` para fontes diferentes, `apad` no fim para o audio nunca ser o stream mais curto e o `-shortest` nao derrubar o ultimo frame). v1 exige o mesmo fps em todos os clipes (adicionar um arquivo com outro fps e recusado com aviso).
- Sequencia na interface (`ui/main_window.py`): a janela guarda uma `Sequence`; playhead, marcadores I/O e cortes vivem em frames da linha do tempo. `Arquivo > Adicionar video a sequencia…` (Ctrl+Shift+O, tambem na toolbar e na aba Exportar) acrescenta clipes ao fim; a aba Exportar lista os clipes (clicar leva o playhead ate o clipe), remove, reordena e divide no playhead (Ctrl+K, tambem no menu de contexto da linha do tempo). O *clipe ativo* e o que esta sob o playhead: as abas mostram e editam os ajustes dele — a cada render a janela grava os widgets de volta no clipe (`_store_active_clip_settings`) e, ao cruzar uma fronteira, carrega os ajustes do outro clipe em todas as abas (`apply()` de cada aba, sem emitir sinais), inclusive regioes (crop, amostra de ruido, ponto focal, cinza de referencia), ganhos de balanco de branco, referencia de exposicao (`Clip.exposure_reference_frame`) e frame de referencia do camera lock (cache por `(arquivo, frame)`; as metades de um split compartilham). Um clipe novo nasce com o look atual (sliders) e regioes/referencias zeradas. O preview decodifica de um `VideoSource` por arquivo (`PreviewController.set_videos`; `PreviewRequest.source_path`) e encaixa o resultado na proporcao do canvas do primeiro clipe (`fit_to_aspect`, `RenderSettings.output_size`) para mostrar as barras que o export vai adicionar. A exportacao usa `Sequence.for_export(inicio, fim)`: o que esta fora dos marcadores vira corte, clipes vazios sao descartados. Eventos `sequence.add`/`sequence.add_refused`/`sequence.split`/`sequence.remove`/`sequence.move` e `ui.clip_active`.
- Trava de exposicao (`pipeline/exposure.py`): cameras em auto-exposicao/AWB re-medem quando o foco caça; a correcao e sem estado — mede a mediana B/G/R de uma regiao estatica da cena no frame fonte (antes da estabilizacao, mesmas coordenadas do retangulo roxo do preview), compara com a mesma regiao no frame de referencia e aplica a razao como ganhos por canal (uma `cv2.LUT` logo apos o denoise, cuja intensidade foi estimada no frame sem ganho). Ganhos limitados a ±`max_ev` stops e atenuados por `strength`; `color=False` usa um ganho unico pela luma. Sem estado, preview e export (serial ou paralelo) produzem o mesmo resultado. Medir o frame inteiro e opcional e sensivel ao conteudo (uma pessoa entrando escurece a media e a correcao 'pulsa'); a regiao fixa nao sofre disso, mas erra enquanto algo a tapa. Realces estourados no frame nao voltam. Na aba HSL (grupo "Iluminacao constante") a regiao e o mesmo retangulo roxo do balanco de branco; "Usar este frame como referencia" mede o frame decodificado atual (mesmo frame do lado "Antes"), mover a regiao ou trocar o escopo re-mede se ainda estamos no frame de referencia e caso contrario apaga a referencia (aviso na barra de status); o rotulo mostra os ganhos do ultimo render lidos do trace. Eventos `ui.exposure`.
- Diagnostico (`diagnostics.py`, `ui/diagnostics_dialog.py`): relatorio puro (sem Qt) com sistema, OpenCL (linha `OpenCL:` do build, `haveOpenCL`, dispositivo, ICDs) e ffmpeg (versao, encoders conhecidos, `/dev/dri/renderD*`), mais dicas em portugues por caso (`opencl_hint`, `ffmpeg_hint`) e o rotulo esperado na barra de status. Logado na abertura em thread daemon (`diagnostics`) e mostrado em Ajuda; `diagnostics.shown` quando o dialogo abre.
- Linha do tempo (`widgets/timeline.py`): regua + trilhas com zoom (Ctrl+roda, pinca do trackpad, botoes −/Ajustar/+) e rolagem (roda, barra). `TimelineScale` faz a conversao frame<->pixel (zoom = pixels por frame, scroll = primeiro frame visivel) e todo hit-test usa a mesma conversao. Um corte `(inicio, fim)` ocupa os pixels de `x_of(inicio)` a `x_of(fim + 1)`; arrastar uma borda redimensiona, arrastar o corpo move, clicar seleciona (Delete remove; a selecao e a mesma da lista na aba Exportar), botao direito abre o menu. Os marcadores I/O sao arrastados na regua. Acima da trilha de video ha a faixa de clipes (`ClipBlock`: nome, frames da linha do tempo e frame de origem de cada clipe; o ativo em destaque, fronteiras tracejadas descendo pelas trilhas); clicar nela posiciona o playhead, arrastar um bloco (mais de 6 px) o reordena — um fantasma segue o mouse e uma barra marca o espaco entre clipes onde ele vai cair (`clipMoveRequested(indice, novo_indice)`; o playhead acompanha o clipe movido) — e o menu de contexto oferece dividir/remover o clipe. Enquanto um corte esta sendo arrastado, o widget ignora `set_cuts` (o arraste e a fonte da verdade) e so na soltura o modelo normalizado volta para ele. Na reproducao, quando o playhead sai da area visivel a vista rola para deixa-lo a 10% da borda esquerda.
- Filmstrip (`ui/timeline_media.py`): a trilha de video mostra miniaturas decodificadas por um `FilmstripController` com thread e um `VideoSource` por arquivo da sequencia (um seek de miniatura nunca atrasa o preview). Cada pedido e uma lista de `FilmstripCell` (frame da linha do tempo -> arquivo + frame de origem, via `Sequence.locate`), e cada miniatura e encaixada na celula com letterbox quando a proporcao difere. As celulas sao ancoradas no frame 0 (rolar reaproveita o cache LRU de 600 imagens), o widget pede apenas as celulas visiveis que faltam apos 60 ms de descanso do zoom/scroll, um pedido novo cancela o anterior entre dois frames e, enquanto a miniatura exata nao chega, a vizinha mais proxima dentro de uma celula e desenhada no lugar. Evento `filmstrip.render` (DEBUG) com quantidade e tempos de decode.
- Reproducao e som (`ui/audio_preview.py`): o relogio de parede e o mestre. Ao dar play, a janela monta a lista de frames mantidos ate o fim do trecho (`Sequence.playback_runs`: cortes e fronteiras de clipe ja resolvidos) e, a cada tique, mostra o frame que o tempo decorrido pede (`frame = ancora + tempo * fps * velocidade`); quando o render nao acompanha, frames sao pulados, mas o tempo nunca atrasa. O combo Velocidade (1x, 1/2x, 1/4x) substitui o antigo "Pular N". O som toca continuo a partir de PCM mono 22,05 kHz que o ffmpeg decodifica por arquivo (uma vez por velocidade, em thread propria: `PcmController`), ja esticado com `atempo` (mantem o tom) — como video e audio seguem o mesmo relogio, ficam em sincronia por construcao; a defasagem e so o atraso do render. O `AudioFeeder` percorre as mesmas `PlaybackRun`s do video (`AudioPiece` por trecho: arquivo + segundos de origem, ou silencio para arquivo sem audio), entao cortes e trocas de clipe sao emendados antes de tocar; so um seek do usuario, a troca de velocidade ou religar o som reancoram e recomecam. O `QAudioSink` em modo push e mantido ~300 ms a frente (timer de 40 ms na GUI thread). Botao "Som" na barra de transporte; sem dispositivo de saida (ou com `SVE_PREVIEW_AUDIO=0`, que a suite de testes usa) o botao fica desligado e a reproducao segue muda; arquivos acima de 30 min tocam sem som (o PCM nao caberia com folga na memoria; ~160 MB por hora e por velocidade). Eventos `audio.preview_device`/`audio.preview_disabled`/`audio.preview_decode`/`audio.preview_skipped`, `ui.playback_speed`, `ui.sound`; `playback.start`/`playback.stop` trazem `speed` e `audio_seconds`.
- Forma de onda (`audio.py`, `WaveformController`): ffmpeg decodifica o primeiro stream de audio para PCM mono 8 kHz e o numpy reduz a picos min/max por bucket (4 buckets por frame de video; 1 h cabe em ~4 MB). Cada coluna de pixel da trilha mostra os extremos dos buckets que cobre (`np.minimum/maximum.reduceat`); com varios clipes, cada coluna le os picos do arquivo do clipe que esta sob ela, com o deslocamento do trecho (`ClipBlock.source_start`), e um unico ganho de exibicao vale para a trilha inteira. Um arquivo e decodificado uma vez por sessao mesmo aparecendo em varios clipes; o resultado chega marcado com o caminho e e descartado se o arquivo saiu da sequencia. Sem ffmpeg a trilha avisa, sem stream de audio mostra \"Sem audio\" no trecho do clipe. Evento `waveform.load`.
- Cores (aba HSL, `pipeline/grade.py`): balanco de branco por ganhos B/G/R (calculados de uma regiao que deveria ser cinza — retangulo roxo no preview — ou do frame inteiro, gray world) mais tint verde/magenta, aplicados antes da curva tonal; e ajustes de matiz/saturacao/luminancia por oito faixas de matiz interpoladas linearmente no circulo de cores (hue vs hue / sat / lum). Tudo vira tabelas (`cv2.LUT`) resolvidas uma vez por `ProcessingContext`; os ganhos ficam gravados nas settings, entao a exportacao usa exatamente o que o preview mostra.

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

Variaveis de ambiente: `SVE_LOG_DIR` (pasta), `SVE_LOG_LEVEL` (`INFO` padrao; `DEBUG` inclui `preview.render` durante a reproducao), `SVE_LOG_STDERR=1` (espelha no terminal). O menu `Ajuda > Abrir pasta de logs` abre a pasta; `Exibir > Metricas` (Ctrl+M, ou o botao Metricas na toolbar) mostra um HUD com tempo de render por estagio, fila, decode, estabilizacao e backend. `SVE_PREVIEW_AUDIO=0` desliga o som do preview.

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
| `ui.*` | interacoes (`ui.tab`, `ui.preset`, `ui.render_mode`, `ui.preview_zoom`, `ui.clip_range`, `ui.cuts`, `ui.clip_active`, `ui.hud`) | |
| `session.summary` | fechar a janela | renders, descartados, exports, erros |
| `preview.error`, `export.error`, `app.unhandled`, `qt.message` | falhas | `exception`, `settings` |

Notas do pipeline registradas em cada render (`preview.render`) e agregadas no export: `stabilization.mode`, `stabilization.estimator` (`lk_translation`, `phase_correlate`, `lk_similarity`, `lk_perspective`, `orb_masked`, `orb_unmasked`, `template_match`, `primed`, `sequence_break`, `cached`, `none`), `stabilization.result` (`applied`, `negligible`, `no_transform`), `stabilization.path` (`start`, `follow`, `held`, `lock`, `reacquire`, `dead_reckoning`), `stabilization.inliers`, `stabilization.tracked_points`, `stabilization.rejected` (`too_few_inliers`, `unreasonable_similarity`, `too_few_perspective_inliers`, `unreasonable_perspective`, `accumulated`), `stabilization.translation_px`, `denoise.mode`, `denoise.sigma` (ruido estimado em luma e cor), `denoise.h`, `filters.backend` (`opencl`, `cpu`, `cpu_no_denoise`), `exposure.current` (medianas B/G/R da regiao no frame) e `exposure.gains` (ganhos aplicados pela trava de exposicao).

Analise rapida com pandas:

```python
import pandas as pd

df = pd.read_json("~/.space-video-enhancer/logs/app.jsonl", lines=True)
renders = df[df.event == "preview.render"]
print(pd.json_normalize(renders.stages).describe())
```

## O que o MVP faz

- seleciona um video local e navega pelos frames com a linha do tempo (zoom e rolagem), botoes de transporte e reproducao quadro a quadro
- junta varios arquivos (mesmo fps) em uma sequencia na mesma linha do tempo, com faixa de clipes, filmstrip e forma de onda por arquivo; divide um clipe no playhead, remove e reordena (botoes ou arrastando o bloco na faixa de clipes); cada clipe tem seus proprios ajustes (as abas mostram os do clipe sob o playhead) e a exportacao concatena tudo, com letterbox para tamanhos diferentes
- zoom do preview por botoes e percentual, com modos de qualidade (alta, balanceado, rapido); reproducao em tempo real ou camera lenta (1/2x, 1/4x) com o som dos clipes em sincronia, respeitando cortes e fronteiras
- comparacao antes/depois com divisor arrastavel
- abas tematicas: cor e detalhe, HSL (balanco de branco e ajuste por cor), filtros (presets com miniaturas do frame atual), estabilizacao, geometria, ruido e midia
- corte temporal com inicio/fim e uma lista de trechos removidos de dentro do video (video e audio)
- presets de melhoria e reset por slider para o valor do preset ativo
- denoise desativado, manual, auto ou por amostra escolhida no preview; nitidez, contraste, brilho, gama, saturacao, temperatura e escala de saida
- estabilizacao por translacao, similaridade, perspectiva, camera lock por regiao focal ou hibrida, com intensidade, suavizacao temporal, crop automatico e zoom out
- filtros tonais para sombras, realces, contraste local e atenuacao de cores quentes/frias
- crop interativo com presets de proporcao e modo livre; rotacao 90/180/270 e espelhamento
- corte temporal por frame inicial/final com marcadores arrastaveis na linha do tempo
- exportacao local com barra de progresso e cancelamento, usando FFmpeg apenas para encode e mux do audio original
- tenta encoders de hardware (NVENC, VAAPI) e OpenCL quando disponiveis, com fallback automatico para CPU/libx264

## Aceleracao por GPU

O processamento em OpenCV pode usar OpenCL (T-API) e o encode pode usar NVENC ou VAAPI. O OpenCL so entra quando o denoise NLMeans esta ligado (num M1, 260 ms contra 1.340 ms na CPU em 720p); os filtros leves rodam na CPU porque a transferencia para o dispositivo custa mais do que o trabalho. Nesse caminho toda a cadeia (denoise, contraste, gama, saturacao, temperatura, nitidez, escala) roda no `UMat` com um unico upload e download. Saturacao e temperatura sao tabelas (`cv2.LUT`), nao aritmetica em float32. Se OpenCL ou o encoder de hardware falharem, o app cai para CPU e `libx264` automaticamente. No Linux, VAAPI exige um dispositivo como `/dev/dri/renderD128`; para NVIDIA, o FFmpeg precisa ter sido compilado com `nvenc`.

**Ajuda -> Diagnostico do sistema** mostra o que a maquina oferece (ICDs OpenCL, dispositivo que o OpenCV usaria, encoders do ffmpeg) com a dica do que falta, e o mesmo relatorio vai para o log na abertura (evento `diagnostics`). O rotulo da barra de status durante o export e `backend + encoder`, por exemplo `OpenCL + NVIDIA NVENC` ou `CPU (OpenCL indisponivel) + libx264 (fallback)`.

### Linux com GPU (Ubuntu 22.04/24.04)

O OpenCV do PyPI ja vem com OpenCL compilado, mas em tempo de execucao ele carrega `libOpenCL.so.1` (o *ICD loader*) e, por ele, o driver da GPU (`/etc/OpenCL/vendors/*.icd`). O caso mais comum de "OpenCL indisponivel" e o driver da NVIDIA instalado sem o loader.

1. Driver da GPU. NVIDIA: `sudo ubuntu-drivers install` (ou `sudo apt install nvidia-driver-550`) e reinicie; o metapacote traz `libnvidia-compute-550`, que e o ICD OpenCL. Intel: `sudo apt install intel-opencl-icd`. AMD: `sudo apt install mesa-opencl-icd` (ou o ROCm).
2. Loader e utilitario: `sudo apt install ocl-icd-libopencl1 clinfo`.
3. Confira, dentro da `.venv` do projeto:

```bash
clinfo | grep -E "Number of platforms|Device Name|Device Version"
ls /etc/OpenCL/vendors/
python -c "import cv2; print(cv2.ocl.haveOpenCL()); cv2.ocl.setUseOpenCL(True); print(cv2.ocl.useOpenCL()); d=cv2.ocl.Device.getDefault(); print(d.name(), d.version())"
ffmpeg -hide_banner -encoders | grep -E "nvenc|vaapi|qsv"
```

Esperado com uma RTX: `Number of platforms 1`, `NVIDIA GeForce RTX ...`, `True` / `True` e `h264_nvenc` na lista do ffmpeg (o ffmpeg do apt ja vem com nvenc; ele carrega as bibliotecas da NVIDIA em tempo de execucao).

Problemas comuns:

- `Number of platforms 0` e `vendors/` vazio: falta o ICD do driver (`libnvidia-compute-<versao>`), reinstale o metapacote do driver.
- `clinfo` lista a GPU mas `haveOpenCL()` e `False`: falta o loader (`ocl-icd-libopencl1`); `ldconfig -p | grep libOpenCL` deve mostrar `libOpenCL.so.1`.
- WSL: a NVIDIA expoe CUDA, mas nao OpenCL, dentro do WSL; o denoise fica na CPU.
- VAAPI (Intel/AMD): precisa de `/dev/dri/renderD128` acessivel (usuario nos grupos `render` e `video`) e do driver (`intel-media-va-driver-non-free` ou `mesa-va-drivers`); `vainfo` confirma.
- A variavel `OPENCV_OPENCL_RUNTIME=disabled` desliga o OpenCL no OpenCV; nao a defina.

Lembre que o OpenCL so acelera o denoise NLMeans (a cadeia leve e a estabilizacao sao CPU por escolha); o NVENC acelera a codificacao de todo export.
