# SigID Shazam

Identificador de sinais de rádio pelo **áudio demodulado** — um "Shazam" para modos
de transmissão. Treinado com as amostras de áudio do
[Signal Identification Wiki](https://www.sigidwiki.com/wiki/Signal_Identification_Guide).

## Uso rápido

1. Dê dois cliques em **`SigID.bat`**.
2. Em *Entrada de áudio*, escolha por onde chega o som do SDR:
   - **CABLE Output (VB-Audio Virtual Cable)** ← recomendado. No SDR# / SDR++ / SDRuno,
     configure a *saída de áudio* para **CABLE Input**. O app já pré-seleciona o cabo.
   - ou **Mixagem estéreo** (captura o que toca nos alto-falantes).
3. Clique em **▶ Ouvir**. Depois de ~4 s aparece o palpite, atualizado a cada segundo.
4. Opcional, mas melhora muito: marque **Dica de frequência** (MHz) e o **Modo de
   demodulação** que você está usando no SDR. Candidatos fora da faixa/modo do sinal
   no sigidwiki perdem peso (não são eliminados).

5. Opcional: marque **Lista QRGBuscador (ILGA + programadas)**. O app consulta as listas do
   [QRGBuscador](https://github.com/lualcorreia/QRGBuscador) na frequência digitada acima
   (± tolerância) e **reforça** os modos/protocolos registrados ali (STANAG, ALE, DSC, HFDL,
   SITOR, Radiofax, VOLMET...). É só um empurrão no peso (leve/médio/forte, até ~1,5×/3×/5×):
   nada é eliminado, e se o áudio não bater, a lista não decide. Entradas fora do horário UTC
   pesam menos. Quando o palpite combina com a lista, aparecem a **estação e o indicativo**
   (só informação). A aba **QRG (lista)** mostra todas as entradas próximas; ✓ = combina
   com o palpite atual. Teste sem a interface: `.venv\Scripts\python qrg_hints.py 8414.5 2`.

Outros recursos:
- **Analisar arquivo…** — identifica uma gravação WAV/MP3/FLAC/OGG.
- **● Salvar últimos 30 s** — grava o áudio em `recordings/` (bom para comparar depois).
- **Congelar / Limpar / Suavização** — controle da média temporal do resultado.
- Duplo clique num candidato abre a página do sinal no sigidwiki.

Linha de comando:
```
.venv\Scripts\python identify.py gravacao.wav
.venv\Scripts\python test_synthetic.py      # CW, RTTY, BPSK31, AFSK... sintéticos
```

## Dicas de áudio no SDR

- Use a **largura de filtro típica do modo** (SSB ~2,4–3 kHz; NFM 12,5 kHz) e AGC ligado.
- Desligue *noise reduction*, *notch* e squelch: eles mudam a "assinatura" do sinal.
- Volume do SDR em nível bom: o medidor deve ficar entre −40 e −10 dBFS.
- Sinais digitais em SSB: o deslocamento de sintonia é tolerado (±250 Hz são treinados).

## Como funciona

`sigfeatures.py` descreve cada janela de 4 s de áudio (reamostrado a 16 kHz) com 468
features de DSP: espectro mel/MFCC, banda ocupada, picos e **espaçamento entre tons**
(shift FSK, portadoras OFDM), histograma da frequência dominante **relativo à mediana**
(não depende da sintonia), **linhas do envelope² e da frequência instantânea** (taxa
de baud/keying), **potências a², a⁴, a⁸** do sinal analítico (assinatura de BPSK/QPSK/8PSK),
envelope constante vs. variável, ciclo de trabalho e periodicidade lenta (pulsos, CW,
varreduras).

`train.py` treina um ensemble **ExtraTrees + kNN (cosseno)** por sinal e um segundo
modelo só de **modulação** (FSK, PSK, AM, FM, OFDM, MFSK...), que generaliza melhor quando
o sinal exato não está na base. Os dados são aumentados com deslocamento de frequência
e ruído de receptor, e há uma classe sintética **"Ruído / sem sinal"**. A validação usa
folds por arquivo (nenhum trecho do mesmo arquivo em treino e teste).

## Atualizar a base

`Atualizar_base_e_treinar.bat` baixa amostras novas do wiki (respeitando o limite de
requisições do site, ~1 por segundo) e retreina. As features ficam em cache em
`data/features`, então só arquivos novos são processados.

## Desempenho medido (21/11/2026)

Base: 1.694 gravações de 552 sinais do sigidwiki (IQ/baseband ignorados), 36.338 janelas.
Validação cruzada em 5 folds **por gravação** (o modelo nunca viu o arquivo testado),
nos 317 sinais com 2+ gravações, entre 550 classes possíveis:

| | top-1 | top-5 |
|---|---|---|
| por gravação inteira | 54,9 % | 71,7 % |
| por janela de 4 s | 50,7 % | 68,3 % |

Teste com sinais sintéticos (CW, RTTY, AFSK 1200, BPSK31, ruído): a **modulação**
acertou todos; o sinal exato cai em vizinhos físicos plausíveis (ex.: RTTY → SITOR-B).

## Limitações honestas

- Muitas páginas do wiki têm 1 ou 2 amostras; sinais parecidos (PACTOR I/II, variantes
  de STANAG, DMR vs. dPMR...) podem se confundir. Olhe o **top-5** e a modulação.
- As amostras do wiki são MP3s de qualidades variadas; sinal fraco ou com QRM derruba a
  confiança.
- O app só vê o áudio demodulado: sinais largos (DVB, LTE, Wi-Fi) só aparecem se o SDR
  entregar o áudio no mesmo formato das gravações do wiki.
