# Tela de 3,5" do Prometheus (Raspberry Pi 5) — ARQUIVADO (04/11/2026)

Ideia: um Pi 5 com uma tela de 3,5" (480x320) ligado na USB-C da PGX, mostrando a trava da ponte, as temperaturas e
os serviços, com os botões TRAVAR / Rearmar / poses. Parado a pedido; o que existe:

- `tela.py` — o programa (pygame, nativo, sem navegador). **Funciona**: testado no PC com `--falso`
  (`SDL_VIDEODRIVER=dummy TELA_CAPTURA=quadro.png python tela.py --falso --janela`).
- `../wla/painel_mini.py` — o lado da PGX (porta 8097): junta o painel do robô (:8095) e os serviços, repassa os botões.
- `prometheus-tela.service`, `monta_seed.py`, `roda_qemu.sh` — Debian 12 ARM64 no QEMU (cloud-init por HTTP).

Onde parou: no QEMU (TCG, PC x86) a VM **ficou no firmware UEFI (AAVMF) sem passar** — só o banner no serial, um
núcleo a 100% por >5 min, tanto com `-cpu max` quanto com `max,pauth-impdef=on`. Próximo passo seria testar
`-cpu cortex-a72` sem janela, ou dar o boot direto no kernel (`-kernel/-initrd`) em vez do UEFI.
No Pi de verdade falta: o driver da tela (modelo?), a USB-C como rede (`dwc2` + `g_ether`, Pi 10.55.0.2 / PGX
10.55.0.1) e a imagem (`rpi-image-gen` com os mesmos arquivos).
A imagem baixada está em `~/vm_tela` (546 MB) — pode apagar se não for voltar a isso.
