# Benchmarks

Scripts de medicao usados para decidir mudancas de algoritmo. Rodam com o pacote instalado
(`python -m pip install -e ".[dev]"`) e imprimem tabelas no terminal; nao fazem parte dos testes.

| Script | O que mede |
|---|---|
| `stabilization_shake.py` | Tremor residual (px) de cada modo de estabilizacao em uma sequencia sintetica com tremor gaussiano e panoramica lenta, contra a entrada e contra uma referencia offline (trajetoria + media movel). |
| `estimator_costs.py` | Custo por frame (ms) dos estimadores, do warp e do auto crop em 720p, e a variacao de zoom do auto crop frame a frame. |
| `denoise_variants.py` | Tempo e PSNR de variantes de denoise em um frame real. A referencia limpa vem da mediana temporal de frames vizinhos (camera parada) ou, com `--synthetic-noise`, do proprio frame antes de receber ruido. |

Exemplos:

```bash
python benchmarks/stabilization_shake.py
python benchmarks/estimator_costs.py
python benchmarks/denoise_variants.py captura.avi --frame 120 --median-window 15
python benchmarks/denoise_variants.py captura.avi --frame 120 --synthetic-noise 6
```
