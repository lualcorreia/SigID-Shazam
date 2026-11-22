# 📡 SigID Shazam — PP2LA

**Um "Shazam" para sinais de rádio.** Ouve o áudio demodulado de um receptor (SDR, Tecsun,
HT...) e diz qual modo/protocolo está sendo transmitido — RTTY, STANAG 4285, ALE, DSC, HFDL,
PACTOR, DMR, POCSAG, SSTV, WEFAX, radares OTH, estações de números e mais: **550 sinais** do
[Signal Identification Wiki](https://www.sigidwiki.com/wiki/Signal_Identification_Guide).

| | Windows (PC) | Android |
|---|---|---|
| Entrada | placa de som / cabo virtual (VB-Cable) do SDR | microfone do celular perto do alto-falante do rádio |
| Download | `SigID-Shazam-Setup-1.0.0.exe` | `SigID-Shazam-1.0.0.apk` |
| Modelo | treinado em áudio de linha | treinado com simulação alto-falante → sala → microfone |
| Requisitos | Windows 10/11 64 bits | Android 7.0+ |
| Internet | não precisa | não precisa (só para abrir o sigidwiki) |

➡️ **[Baixar na página de Releases](../../releases/latest)**

## O que ele faz

- Waterfall + espectro ao vivo, palpite principal com confiança e **top-8 de candidatos**.
- **Modulação provável** (FSK, PSK, AFSK, MFSK, OFDM, OOK/CW, AM, FM...) por um modelo separado
  que generaliza melhor quando o sinal exato não está na base.
- **Dicas opcionais** que só reforçam candidatos (o áudio continua decidindo):
  frequência sintonizada (faixas do sigidwiki), modo de demodulação (USB/LSB/CW/AM/NFM/WFM) e a
  lista do **[QRGBuscador](https://github.com/lualcorreia/QRGBuscador)** (ILGA + transmissões
  programadas, ~77 mil entradas), com tolerância, peso e horário UTC. Quando o palpite bate com
  a lista, mostra **estação e indicativo**.
- Histórico de detecções, congelar/limpar, suavização, **gravar os últimos 30 s** em WAV e
  **analisar arquivos** de áudio.

## Desempenho medido

Validação cruzada **por gravação** (o modelo nunca viu o arquivo testado), 317 sinais com 2+
gravações, entre 550 classes possíveis:

| Modelo | Condição de teste | top-1 | top-5 |
|---|---|---|---|
| PC | áudio de linha (5 folds) | 54,9 % | 71,7 % |
| Android | microfone simulado (2 de 5 folds) | 54,1 % | 72,8 % |

Com o mesmo áudio passado pelo microfone simulado, o modelo Android acertou 23/30 no top-1
contra 9/30 do modelo de PC. Sinais parecidos (PACTOR I/II, variantes de STANAG...) podem se
confundir: olhe o top-5 e a modulação.

## Como funciona

Cada janela de 4 s (16 kHz) vira 468 features de DSP: espectro mel/MFCC, banda ocupada,
picos e espaçamento entre tons (shift FSK, portadoras OFDM), histograma da frequência dominante
relativo à mediana (independe da sintonia), linhas do envelope² e da frequência instantânea
(taxa de baud/keying), potências a²/a⁴/a⁸ do sinal analítico (assinatura BPSK/QPSK/8PSK),
envelope constante × variável e estrutura temporal (ciclo de trabalho, pulsos, varreduras).
Um ensemble de 3 redes MLP + kNN cosseno (PCA) classifica o sinal. No Android, todo o DSP e a
inferência foram reescritos em Kotlin nativo e validados contra o Python (`ParityTest`).

## Estrutura

```
pc/        app Windows (Python/PyQt6), treino, download da base, empacotamento (PyInstaller + Inno Setup)
android/   app Android (Kotlin), modelo exportado em app/src/main/assets
```

### Rodar/compilar a partir do código

**PC** (Python 3.12+):
```bat
cd pc
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python sigid_download.py      :: baixa as amostras do sigidwiki (~1,4 GB, respeita o limite do site)
.venv\Scripts\python qrg_hints.py --update  :: listas do QRGBuscador
.venv\Scripts\python train.py               :: modelo PC (ou baixe sigid_model.joblib da Release para models\)
.venv\Scripts\python train.py --mobile      :: modelo Android
.venv\Scripts\python app.py
packaging\build_installer.bat               :: gera o instalador
```

**Android**: `android\build_apk.bat` (exporta o modelo mobile, roda o teste de paridade e gera o APK).

## Créditos

- Amostras de áudio e metadados dos sinais: **[Signal Identification Wiki](https://www.sigidwiki.com)**
  e seus colaboradores. Os áudios não são redistribuídos aqui; o script baixa direto do wiki.
- Lista de frequências: **ILGA** (International Listening Guide) via
  [QRGBuscador](https://github.com/lualcorreia/QRGBuscador).

73 de **PP2LA** — Lucas Alexandre Gama Correia
