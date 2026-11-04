#!/bin/bash
# A TELA DO PROMETHEUS NO QEMU (04/11): Debian 12 ARM64 (a base do Raspberry Pi OS atual) numa janela de 480x320,
# o mouse fazendo papel de toque. Na 1ª vez o cloud-init instala o pygame e a tela (uns minutos: é ARM emulado).
#   bash lerobot-ext/tela_rpi/roda_qemu.sh            (fecha: feche a janela)
#   SSH de teste: ssh -p 2222 -i <chave de teste> tela@127.0.0.1
set -e
AQUI=$(cd "$(dirname "$0")" && pwd)
VM=${VM:-$HOME/vm_tela}
BASE=$VM/debian-12-generic-arm64.qcow2
test -f "$BASE" || curl -fL -o "$BASE" https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-arm64.qcow2
test -f "$VM/tela.qcow2" || { qemu-img create -q -f qcow2 -b "$BASE" -F qcow2 "$VM/tela.qcow2" 6G; }
test -f "$VM/vars.fd" || cp /usr/share/AAVMF/AAVMF_VARS.fd "$VM/vars.fd"
# o "seed" do cloud-init por HTTP (sem precisar de ISO nem de root): o cloud-init.yaml com a tela embutida
mkdir -p "$VM/seed"
python3 "$AQUI/monta_seed.py" "$VM/seed"
( cd "$VM/seed" && exec python3 -m http.server 8765 --bind 127.0.0.1 ) > "$VM/seed.log" 2>&1 &
SEED=$!
trap 'kill $SEED 2>/dev/null' EXIT
qemu-system-aarch64 -name "Prometheus - tela 3.5 pol (QEMU)" -M virt -cpu max,pauth-impdef=on -accel tcg,thread=multi -smp 4 -m 2048 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/AAVMF/AAVMF_CODE.no-secboot.fd \
  -drive if=pflash,format=raw,file="$VM/vars.fd" \
  -drive file="$VM/tela.qcow2",if=virtio,format=qcow2 \
  -device virtio-gpu-pci,xres=480,yres=320 -display gtk,zoom-to-fit=off \
  -device qemu-xhci -device usb-tablet -device usb-kbd \
  -netdev user,id=n0,hostfwd=tcp:127.0.0.1:2222-:22 -device virtio-net-pci,netdev=n0 \
  -smbios "type=1,serial=ds=nocloud;s=http://10.0.2.2:8765/" \
  -serial file:"$VM/console.log" \
  -monitor unix:"$VM/monitor.sock",server,nowait
