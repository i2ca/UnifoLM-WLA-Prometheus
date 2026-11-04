#!/usr/bin/env python3
"""Monta o seed do cloud-init (user-data, meta-data) com o tela.py e o serviço embutidos — para o QEMU.
A mesma lista (pacotes, usuário, arquivos, serviço) é a que vai para a imagem do Pi (rpi-image-gen) depois."""
import base64
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
dst = Path(sys.argv[1])
dst.mkdir(parents=True, exist_ok=True)
b64 = lambda p: base64.b64encode((AQUI / p).read_bytes()).decode()   # noqa: E731
chave = os.environ.get("TELA_SSH_PUB", "")
ud = f"""#cloud-config
hostname: prometheus-tela
users:
  - name: tela
    lock_passwd: true
    groups: [video, input, render]
    shell: /bin/bash
{"    ssh_authorized_keys: [" + repr(chave) + "]" if chave else ""}
ssh_pwauth: false
package_update: true
packages: [python3-pygame]
write_files:
  - path: /opt/prometheus-tela/tela.py
    encoding: b64
    permissions: "0755"
    content: {b64("tela.py")}
  - path: /etc/systemd/system/prometheus-tela.service
    encoding: b64
    content: {b64("prometheus-tela.service")}
runcmd:
  - [systemctl, daemon-reload]
  - [systemctl, enable, --now, prometheus-tela.service]
"""
(dst / "user-data").write_text(ud)
(dst / "meta-data").write_text("instance-id: prometheus-tela-1\nlocal-hostname: prometheus-tela\n")
(dst / "vendor-data").write_text("")
print(f"seed em {dst} (tela.py {len((AQUI / 'tela.py').read_bytes())} bytes)")
