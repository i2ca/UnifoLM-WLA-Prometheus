#!/bin/bash
# ZED (Stereolabs, 2b03) que não aparece no USB depois de ligar o robô: a ZED 1 às vezes não enumera
# quando liga junto com o hub, e só volta desconectando e conectando de novo. Este script faz a
# mesma coisa por software: se a ZED não estiver no USB, desautoriza e reautoriza os HUBS USB 3
# (o das câmeras), e o kernel enumera de novo o que está pendurado neles. Precisa de root.
#
#   sudo /usr/local/sbin/reenumera_zed.sh          # o init chama com `sudo -n` antes das câmeras
set -u
tem_zed() { grep -qi "^2b03" /sys/bus/usb/devices/*/idVendor 2>/dev/null; }

if tem_zed; then
    echo "[zed] já está no USB — nada a fazer"
    exit 0
fi
if [ "$(id -u)" != "0" ]; then
    echo "[zed] preciso de root para reenumerar o USB"
    exit 2
fi
for tentativa in 1 2 3; do
    # hubs USB 3 externos (classe 09), não os controladores raiz
    for d in /sys/bus/usb/devices/[0-9]*-[0-9]*; do
        [ -f "$d/bDeviceClass" ] || continue
        [ "$(cat "$d/bDeviceClass")" = "09" ] || continue
        [ "$(cat "$d/speed" 2>/dev/null)" -ge 5000 ] 2>/dev/null || continue
        echo "[zed] tentativa $tentativa: reenumerando o hub $(basename "$d")"
        echo 0 > "$d/authorized"
        sleep 1
        echo 1 > "$d/authorized"
    done
    for _ in $(seq 1 10); do
        sleep 1
        if tem_zed; then
            echo "[zed] ✅ ZED apareceu no USB (tentativa $tentativa)"
            exit 0
        fi
    done
done
echo "[zed] ❌ a ZED não apareceu — desconecte e conecte o cabo dela"
exit 1
