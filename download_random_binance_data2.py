#!/usr/local/bin/python3.11
import requests, os, random, csv, numpy as np

# ==== Jouw analyse constants ====
UP = 1
DOWN = 2

# ==== Analyse functie (jouw read_csv) ====
def read_csv(input_path):
    bins = {}
    bins_up = {}
    bins_down = {}
    price_moves = {}
    prev_price = None
    prev_direction = None
    counter = 0
    streak_start_price = None

    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            price = float(row[2])
            if prev_price is None:
                prev_price = price
                continue
            if price > prev_price:
                direction = UP
            elif price < prev_price:
                direction = DOWN
            else:
                prev_price = price
                continue
            if counter == 0:
                streak_start_price = prev_price
            counter += 1
            if prev_direction is not None and direction != prev_direction:
                streak_len = counter
                delta = abs(prev_price - streak_start_price)
                bins[streak_len] = bins.get(streak_len, 0) + 1
                price_moves.setdefault(streak_len, []).append(delta)
                if prev_direction == UP:
                    bins_up[streak_len] = bins_up.get(streak_len, 0) + 1
                else:
                    bins_down[streak_len] = bins_down.get(streak_len, 0) + 1
                counter = 1
                streak_start_price = prev_price
            prev_direction = direction
            prev_price = price

    total = sum(bins.values())
    print("\n=== Hit streak stats ===\n")
    print(f"Total hits: {total}\n")
    for s in sorted(bins.keys()):
        arr = np.array(price_moves.get(s, []))
        if len(arr) == 0:
            continue
        print(f"Streak {s-1}: {bins[s]} hits ({bins[s]/total*100:.2f}%)")
        print(f"  min Δ: {arr.min():.8f}")
        print(f"  max Δ: {arr.max():.8f}")
        print(f"  mean Δ: {arr.mean():.8f}")
        print(f"  median Δ: {np.median(arr):.8f}")
        print(f"  p10: {np.percentile(arr, 10):.8f}")
        print(f"  p25: {np.percentile(arr, 25):.8f}")
        print(f"  p50: {np.percentile(arr, 50):.8f}")
        print(f"  p75: {np.percentile(arr, 75):.8f}")
        print(f"  p90: {np.percentile(arr, 90):.8f}")
        print()

# ==== Stap 1: haal alle pairs op ====
def fetch_pairs():
    url = "https://data.binance.vision/?prefix=data/futures/cm/daily/bookTicker/"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Kan URL niet openen: {url}")

    pairs = []
    for line in r.text.splitlines():
        line = line.strip()
        if line.endswith("/") and line != "../" and not line.startswith("Home"):
            pair_name = line.strip("/").split("/")[-1]
            pairs.append(pair_name)

    with open("pairs.txt", "w") as f:
        f.write("\n".join(pairs))
    return pairs

# ==== Stap 2: random dag CSV selecteren en downloaden ====
def fetch_random_day_csv(pair):
    base_url = "https://data.binance.vision/data/futures/cm/daily/bookTicker/"
    pair_url = base_url + f"{pair}/"
    r = requests.get(pair_url)
    if r.status_code != 200:
        print(f"[WARN] Kan pair URL niet openen: {pair_url}")
        return None

    csv_files = []
    for line in r.text.splitlines():
        line = line.strip()
        if line.endswith(".csv"):
            # strip query strings als aanwezig
            csv_files.append(line.split("?")[0])

    if not csv_files:
        print(f"[WARN] Geen CSV bestanden gevonden voor pair {pair}")
        return None

    selected = random.choice(csv_files)
    csv_url = pair_url + selected

    local_dir = os.path.join("tick_data", pair)
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, selected)

    r2 = requests.get(csv_url)
    if r2.status_code == 200:
        with open(local_file, "wb") as f:
            f.write(r2.content)
        print(f"[INFO] CSV opgeslagen: {local_file}")
        return local_file
    else:
        print(f"[WARN] Download mislukt: {csv_url}")
        return None

# ==== Main workflow ====
if __name__ == "__main__":
    # 1. Haal alle pairs op en sla op
    pairs = fetch_pairs()
    print(f"[INFO] Totaal {len(pairs)} pairs gevonden.")

    # 2. Geef je pair
    my_pair = "ADAUSD_PERP"
    if my_pair not in pairs:
        print(f"[ERROR] Pair {my_pair} niet gevonden in de lijst!")
    else:
        # 3. Pak random dag CSV
        csv_file = fetch_random_day_csv(my_pair)
        if csv_file:
            # 4. Analyse uitvoeren
            read_csv(csv_file)
