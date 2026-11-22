# SigID Shazam — Android

Versão para celular do SigID Shazam: identifica o modo/protocolo de rádio **ouvindo pelo
microfone** o alto-falante de um receptor (Tecsun, XHDATA, HT, SDR no notebook...). Funciona
**100% offline** em campo.

## Instalar

1. Copie `SigID-Shazam.apk` para o celular (cabo, Drive, WhatsApp para você mesmo).
2. Abra o arquivo e permita "instalar apps desta fonte" quando o Android pedir.
3. Na primeira vez que tocar em **Ouvir**, autorize o microfone.

Requer Android 7.0 ou mais novo.

## Uso em campo

- Volume do rádio em nível confortável, celular a 10–30 cm do alto-falante.
- O medidor deve ficar entre −45 e −15 dBFS. Abaixo de −72 aparece "Sem áudio".
- O primeiro palpite sai após ~4 s e é atualizado continuamente; mantenha 10–20 s no
  mesmo sinal para a média estabilizar (ajuste em *Suavização*).
- **Dicas** (opcionais): frequência em MHz, faixas do sigidwiki, modo do rádio (USB/LSB/AM...)
  e a **lista QRGBuscador** (ILGA + programadas, embutida no app) com tolerância, peso e
  horário UTC. Elas só reforçam candidatos; quem decide é o áudio. Quando o palpite bate
  com a lista, aparecem estação e indicativo.
- **Congelar** trava o resultado; **Salvar 30 s** grava um WAV e abre o compartilhamento;
  **Arquivo…** analisa uma gravação (WAV/MP3/OGG/M4A/FLAC).
- Toque num candidato para ver detalhes e abrir a página do sigidwiki (essa parte precisa de internet).

## Como foi feito

- Modelo treinado no PC (`SigID-Shazam/train.py --mobile`) com simulação de
  **alto-falante → sala → microfone de celular**: passa-faixa e ressonância de alto-falante
  pequeno, distorção, reverberação (RT60 0,08–0,6 s), ruído de campo (vento, falatório,
  ruído rosa, trânsito), passa-altas/AGC de microfone.
- DSP portado 1:1 para Kotlin (`dsp/Features.kt`): as mesmas 468 features do Python.
  `ParityTest` compara Kotlin × Python em vetores de teste exportados.
- Inferência nativa (`model/SigModel.kt`): 3 redes MLP + kNN cosseno (PCA 96) + rede de
  modulação, pesos exportados por `export_android.py`.
- Captura com `AudioRecord` a 16 kHz, fonte UNPROCESSED quando o aparelho suporta, com
  AGC/supressão de ruído/eco desligados.

## Recompilar

`build_apk.bat` exporta o modelo mobile, roda o teste de paridade e gera `SigID-Shazam.apk`.
Ferramentas (JDK 17, Android SDK, Gradle) ficam em `C:\Users\lualc\AndroidBuild`.
