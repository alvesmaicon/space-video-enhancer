# Space Video Enhancer

MVP desktop para abrir um video local, comparar antes e depois quadro a quadro, ajustar a intensidade dos filtros e exportar o resultado sem upload.

## Requisitos

- Python 3.9+
- FFmpeg instalado e disponivel no `PATH`

No Windows, instale:

- Python 3.9+ 64-bit
- FFmpeg com a pasta `bin` adicionada ao `PATH`
- Microsoft Visual C++ Redistributable 2015-2022 x64 apenas se aparecer erro de DLL ao abrir o app

No macOS com Homebrew:

```bash
brew install ffmpeg
```

No Windows, uma opcao comum e:

1. baixar o FFmpeg em build pronta para Windows
2. extrair para uma pasta como `C:\ffmpeg`
3. adicionar `C:\ffmpeg\bin` ao `PATH`
4. abrir um novo PowerShell e confirmar com `ffmpeg -version`

Se preferir instalar com `winget`, um caminho comum e:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Gyan.FFmpeg -e
```

Depois, feche e abra um novo PowerShell e confirme:

```powershell
python --version
ffmpeg -version
```

## Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install .
```

## Executar

```bash
./run.sh
```

O script cria a `.venv` se necessario, garante `pip`, instala as dependencias quando o projeto muda e inicia a interface.

## Windows

O `run.sh` e voltado para macOS/Linux. No Windows, rode manualmente no PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install .
python main.py
```

Se a ativacao do ambiente virtual for bloqueada no PowerShell, rode antes:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Se preferir, tambem pode iniciar sem ativar o ambiente virtual:

```powershell
.\.venv\Scripts\python.exe main.py
```

Observacoes para Windows:

- a interface pode abrir sem FFmpeg, mas a exportacao do video nao funciona
- normalmente nao e necessario instalar Visual Studio Build Tools, porque as dependencias costumam ter wheels prontas
- Git so e necessario se voce for clonar o repositorio

## Gerenciamento de pacotes

Voce nao precisa de algo como `yarn` para este projeto. Aqui o fluxo padrao e:

- `venv` para isolar o ambiente
- `pip` para instalar dependencias
- `pyproject.toml` para declarar os pacotes do projeto

Se depois voce quiser uma experiencia mais moderna e rapida, a alternativa natural no ecossistema Python e `uv`, nao `yarn`.

## O que o MVP faz

- seleciona um video local
- permite navegar pelos frames com um slider
- inclui botoes para ir ao inicio, voltar um frame, avancar um frame, ir ao final e reproduzir a visualizacao quadro a quadro
- exibe comparacao antes/depois com divisor arrastavel
- inclui crop interativo sobre o preview, com presets de proporcao e modo livre
- aplica a melhoria sempre com OpenCV frame a frame no preview e na exportacao
- oferece presets de melhoria para partir de um ajuste base
- permite resetar cada slider para o valor do preset ativo
- ajusta denoise, nitidez, contraste, brilho, gama, saturacao, temperatura e escala de saida
- permite marcar frame inicial e final para exportar apenas um corte do video, com indicadores visuais no slider
- mostra barra de progresso durante a exportacao
- permite cancelar uma exportacao em andamento sem fechar o programa
- exporta um novo arquivo localmente, usando FFmpeg apenas para encode final e mux do audio original