import os
import csv
import sys
import argparse
import rasterio


# Per eseguire 'python Controlla_dimensioni.py /pathConCartelleDeiComuni'
# ex: python Controlla_dimensioni.py C:\Users\spert\Desktop\TESI\immagini\Dati_super_res-20250705T164910Z-1-001\Dati_super_res

keywords_2m = ["agea", "cgr"]
keywords_10m = ["Sentinel2_pre", "Sentinel2_post"]

def find_file(folder, keyword):
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            return os.path.join(folder, f)
    return None

def main():
    parser = argparse.ArgumentParser(description='Controlla le dimensioni dei raster e genera un report CSV')
    parser.add_argument('root_dir', 
                       help='Percorso della directory root contenente i dati dei comuni')
    parser.add_argument('-o', '--output', 
                       default='coerenza_raster.csv',
                       help='Nome del file CSV di output (default: coerenza_raster.csv)')
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    output_csv = args.output
    
    # Verifica che la directory esista
    if not os.path.exists(root_dir):
        print(f"Errore: La directory '{root_dir}' non esiste!")
        sys.exit(1)
    
    if not os.path.isdir(root_dir):
        print(f"Errore: '{root_dir}' non è una directory!")
        sys.exit(1)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Comune", "Pair", "File_2m", "File_10m", "W_2m", "H_2m", "W_10m", "H_10m", "Ratio_W", "Ratio_H", "Coerenza"])

        for comune in sorted(os.listdir(root_dir)):
            comune_path = os.path.join(root_dir, comune)
            if not os.path.isdir(comune_path):
                continue

            for kw2, kw10 in zip(keywords_2m, keywords_10m):
                p2 = find_file(comune_path, kw2)
                p10 = find_file(comune_path, kw10)
                pair_name = f"{kw2} ↔ {kw10}"

                if not p2 or not p10:
                    print(f"[ATTENZIONE] File mancanti per {comune} - {pair_name}")
                    writer.writerow([comune, pair_name, p2 or "", p10 or "", "", "", "", "", "", "", "MANCA_FILE"])
                    continue

                with rasterio.open(p2) as src2, rasterio.open(p10) as src10:
                    print(f"{comune} - {pair_name}")
                    print(f"  2m file: {p2}, resolution: {src2.res}")
                    print(f" 10m file: {p10}, resolution: {src10.res}")
                    w2, h2 = src2.width, src2.height
                    w10, h10 = src10.width, src10.height
                    rw = w2 / w10
                    rh = h2 / h10
                    coherent = "OK" if (4.5 <= rw <= 5.5 and 4.5 <= rh <= 5.5) else "NO"
                    writer.writerow([comune, pair_name, p2, p10, w2, h2, w10, h10, f"{rw:.2f}", f"{rh:.2f}", coherent])

    print(f"Report generato con SUCCESSO: {output_csv}")

if __name__ == "__main__":
    main()
