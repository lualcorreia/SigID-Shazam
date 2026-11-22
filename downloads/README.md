# Downloads — SigID Shazam 1.0.0

| Arquivo | Tamanho | Para quê |
|---|---|---|
| [**SigID-Shazam-Setup-1.0.0.exe**](https://github.com/lualcorreia/SigID-Shazam/raw/main/downloads/SigID-Shazam-Setup-1.0.0.exe) | 81 MB | Instalador Windows 10/11 64 bits |
| [**SigID-Shazam-1.0.0.apk**](https://github.com/lualcorreia/SigID-Shazam/raw/main/downloads/SigID-Shazam-1.0.0.apk) | 29,7 MB | App Android 7.0+ |
| [sigid_model.joblib](https://github.com/lualcorreia/SigID-Shazam/raw/main/downloads/sigid_model.joblib) | 22 MB | Modelo de PC, para rodar a partir do código (copie para `pc/models/`) |

## Windows

1. Baixe e execute `SigID-Shazam-Setup-1.0.0.exe`. Ele instala só para o seu usuário
   (não pede administrador) em `%LOCALAPPDATA%\Programs\SigID Shazam`, com atalho no menu
   Iniciar e desinstalador em *Configurações → Aplicativos*.
2. O instalador não tem assinatura digital: se o SmartScreen avisar "editor desconhecido",
   clique em **Mais informações → Executar assim mesmo**.
3. No SDR (SDR#, SDR++, SDRuno, HDSDR), mande o áudio para **CABLE Input** do
   [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) e escolha **CABLE Output** no SigID.
   As gravações ficam em `Documentos\SigID-Shazam\gravacoes`.

## Android

1. Baixe `SigID-Shazam-1.0.0.apk` no celular e abra; permita *instalar apps desta fonte*.
2. Toque em **Ouvir** e autorize o microfone. Celular a 10–30 cm do alto-falante do rádio,
   medidor entre −45 e −15 dBFS.

## Verificação (SHA-256)

```
b1daa147144d2320a7efb58adb82098a9dcc2ce1ae763a9e52ea272de6b1fe8f  SigID-Shazam-Setup-1.0.0.exe
a32d2237e694474bc07d10de7561f7e591ea42a40110568adba185790d15e92e  SigID-Shazam-1.0.0.apk
6c3dbe7db7b023644680b2f4d82f267c6e417a099cc083b2f0afe623c4ee849c  sigid_model.joblib
```

No Windows: `certutil -hashfile SigID-Shazam-Setup-1.0.0.exe SHA256`
