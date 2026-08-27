# Aceleracao por GPU

O processamento em OpenCV pode usar OpenCL (T-API) e o encode pode usar NVENC ou VAAPI. O OpenCL so entra quando o denoise NLMeans esta ligado (num M1, 260 ms contra 1.340 ms na CPU em 720p); os filtros leves rodam na CPU porque a transferencia para o dispositivo custa mais do que o trabalho. Nesse caminho toda a cadeia (denoise, contraste, gama, saturacao, temperatura, nitidez, escala) roda no `UMat` com um unico upload e download. Saturacao e temperatura sao tabelas (`cv2.LUT`), nao aritmetica em float32. Se OpenCL ou o encoder de hardware falharem, o app cai para CPU e `libx264` automaticamente. No Linux, VAAPI exige um dispositivo como `/dev/dri/renderD128`; para NVIDIA, o FFmpeg precisa ter sido compilado com `nvenc`.

**Ajuda -> Diagnostico do sistema** mostra o que a maquina oferece (ICDs OpenCL, dispositivo que o OpenCV usaria, encoders do ffmpeg) com a dica do que falta, e o mesmo relatorio vai para o log na abertura (evento `diagnostics`). O rotulo da barra de status durante o export e `backend + encoder`, por exemplo `OpenCL + NVIDIA NVENC` ou `CPU (OpenCL indisponivel) + libx264 (fallback)`.

## Linux com GPU (Ubuntu 22.04/24.04)

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
