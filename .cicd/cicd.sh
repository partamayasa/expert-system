#!/bin/bash

# 1. Masuk ke folder proyek
cd /home/expert-system

# 2. Catat ID komit lokal saat ini sebelum diperbarui
OLD_COMMIT=$(git rev-parse HEAD)

# 3. Paksa ambil kode terbaru dari server dan timpa file lokal
git fetch origin development
git reset --hard origin/development

# 4. Catat ID komit setelah proses reset
NEW_COMMIT=$(git rev-parse HEAD)

# 5. Cek apakah ID komit berubah (artinya ada kode baru yang masuk)
if [ "$OLD_COMMIT" != "$NEW_COMMIT" ]; then
    echo "$(date): New code detected; initiating server update" >> /home/expert-system/.cicd/cicd.log

    # Restart service agar perubahan langsung aktif
    sudo systemctl restart expert-system.service
fi