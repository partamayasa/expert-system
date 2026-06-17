import os
import time
import seedir as sd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # kunci lokasi folder tempat skrip app.py berada saat ini (tools/generator)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # cari letak root proyek 'app tesis' secara dinamis dengan mendeteksi file penanda utama
    # strategi ini menjamin target pemindaian selalu tepat di root proyek anda, di mana pun skrip dijalankan
    project_root = current_dir
    while project_root and project_root != os.path.dirname(project_root):
        if 'requirements.txt' in os.listdir(project_root) or 'README.md' in os.listdir(project_root):
            break
        project_root = os.path.dirname(project_root)
        
    # jaga-jaga jika file penanda tidak ketemu, fallback ke 3 tingkat ke atas
    if not project_root or project_root == os.path.dirname(project_root):
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        
    target_path = project_root
    
    # kunci folder output agar tepat berada di dalam folder kerja generator anda (tools/generator/data)
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    
    # membuat nilai penanda waktu unix saat ini untuk nama berkas dinamis
    unix_time = int(time.time())
    
    # tentukan jalur lengkap untuk output teks dan gambar dengan format unixtime_TreeStructure
    output_txt_path = os.path.join(output_dir, f'{unix_time}_TreeStructure.txt')
    output_png_path = os.path.join(output_dir, f'{unix_time}_TreeStructure.png')
    
    print("=" * 50)
    print(f"Struktur Direktori Utama Proyek (Target: {target_path})\n")
    
    # ambil hasil struktur proyek (saring folder tools agar tidak looping memindai diri sendiri)
    tree_string = sd.seedir(
        target_path,
        style='lines',
        depthlimit=3,
        # Mengunci '.venv' dan variasi virtual environment agar fokus penuh ke modul tesis Anda
        exclude_folders=['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'ENV', 'tools', 'generator', 'seedir', 'other'],
        exclude_files=['.DS_Store', 'desktop.ini'],
        sort=True,
        printout=False 
    )
    
    # cetak manual ke terminal agar anda tetap bisa melihat prosesnya langsung
    print(tree_string)
    print("=" * 50)
    
    # simpan hasil teks (.txt)
    print(f"\n[Info] Menyimpan teks struktur ke '{output_txt_path}'...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(tree_string)
        
    # konversi string teks pohon menjadi gambar png background putih
    print(f"[Info] Membuat gambar struktur direktori ke '{output_png_path}'...")
    
    lines = tree_string.split('\n')
    num_lines = len(lines)
    
    fig, ax = plt.subplots(figsize=(12, max(4, num_lines * 0.3)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    formatted_text = "\n".join(lines)
    
    ax.text(
        0.01, 0.99, 
        formatted_text, 
        fontfamily='monospace', 
        fontsize=11, 
        color='black',
        verticalalignment='top', 
        horizontalalignment='left'
    )
    
    plt.savefig(output_png_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    
    print(f"[Info] Semua output berhasil disimpan dengan aman di '{output_dir}'!")