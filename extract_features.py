import cv2
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import os, glob
import numpy as np

def horizontal_black_density(img, k=4, dtype=np.float32):
    """
    Funkcija deli već binarnu sliku na k horizontalnih segmenata i računa
    gustinu crnih piksela u svakom segmentu.
    Očekuje se da je 'img' 2D numpy niz sa vrednostima 0/1 ili 0/255.
    Rezultat: numpy niz dužine k sa vrednostima [0,1].
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("Očekujem 2D sliku (H,W).")

    H, W = img.shape
    edges = np.linspace(0, H, num=k+1, dtype=int)

    dens = np.empty(k, dtype=dtype)
    for i in range(k):
        r0, r1 = edges[i], edges[i + 1]
        seg = img[r0:r1, :]
        dens[i] = np.count_nonzero(seg == 0) / seg.size if seg.size > 0 else 0.0
    return dens

def vertical_black_density(img, k=4, dtype=np.float32):
    """
    Deli binarnu sliku (0=crno, >0=belo) na k vertikalnih segmenata
    i vraća gustinu crnih piksela po segmentu kao niz dužine k.
    """
    H, W = img.shape
    edges = np.linspace(0, W, k+1, dtype=int)

    dens = np.empty(k, dtype=dtype)
    for i in range(k):
        c0, c1 = edges[i], edges[i+1]
        seg = img[:, c0:c1]
        dens[i] = np.count_nonzero(seg == 0) / seg.size if seg.size > 0 else 0.0
    return dens

def plot_image_with_densities(img):
    """
    Prikazuje sliku, ispod gustinu crnog po koloni (x), desno po redu (y).
    Pretpostavlja binarnu sliku (0=crno, >0=belo).
    """
    H, W = img.shape

    y_density = horizontal_black_density(img, k=H)  # po redovima
    x_density = vertical_black_density(img, k=W)  # po kolonama

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[H, max(1, H // 5)],
                  width_ratios=[W, max(1, W // 5)])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_img)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_img)

    # slika
    ax_img.imshow(img, cmap="gray", interpolation="nearest")
    ax_img.set_title("Slika (binarna)")
    ax_img.axis("off")

    # gustina po kolonama (x-density)
    ax_bot.plot(np.arange(W), x_density)
    ax_bot.set_xlim(0, W - 1)
    ax_bot.set_xlabel("x (kolona)")
    ax_bot.set_ylabel("gustina crnog")
    ax_bot.grid(True, alpha=0.3)

    # gustina po redovima (y-density)
    ax_right.plot(y_density, np.arange(H))
    ax_right.set_ylim(H - 1, 0)
    ax_right.set_xlabel("gustina crnog")
    ax_right.set_ylabel("y (red)")
    ax_right.grid(True, alpha=0.3)

    plt.show()
    return x_density, y_density

def analyze_rows_transitions_splits_merges(img: np.ndarray):
    """
      transitions : np.ndarray (H,)  - broj prelaza crno↔belo po redu
      black_runs  : np.ndarray (H,)  - broj crnih run-ova po redu
      splits      : np.ndarray (H,)  - koliko se run-ova u redu r "raspalo" u r+1 (>=0)
      merges      : np.ndarray (H,)  - koliko se run-ova u redu r nastalo spajanjem iz r-1 (>=0)
    """

    def _row_runs(mask_row: np.ndarray):
        """
        Vrati listu crnih intervala [start, end) u jednom redu (bool niz).
        """
        if mask_row.ndim != 1:
            raise ValueError("Očekujem 1D bool niz za red.")
        if mask_row.size == 0:
            return []
        # prelazi 0->1 i 1->0 u bool->int diff
        x = mask_row.astype(np.uint8)
        d = np.diff(x, prepend=0, append=0)  # +1 na početku run-a, -1 na kraju
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        return list(zip(starts, ends))  # [start, end)

    def _overlap_count(interval, runs_next):
        """
        Koliko run-ova iz runs_next se preklapa sa datim intervalom [a,b).
        """
        a, b = interval
        cnt = 0
        for c, d in runs_next:
            if not (d <= a or b <= c):  # postoji presek
                cnt += 1
        return cnt

    if img.ndim != 2:
        raise ValueError("Očekujem 2D sliku (H,W).")
    H, W = img.shape
    black = (img == 0)

    transitions = np.zeros(H, dtype=np.int32)
    black_runs  = np.zeros(H, dtype=np.int32)
    splits      = np.zeros(H, dtype=np.int32)
    merges      = np.zeros(H, dtype=np.int32)

    # Pre-izračunaj run-ove po redovima
    runs_per_row = [ _row_runs(black[r]) for r in range(H) ]

    for r in range(H):
        row = black[r]
        # broj prelaza je broj promena 0/1 u redu
        transitions[r] = int(np.count_nonzero(np.diff(row.astype(np.uint8))) )
        black_runs[r]  = len(runs_per_row[r])

        # splits: koliko run-ova iz r prelazi u više različitih run-ova u r+1
        if r + 1 < H and black_runs[r] > 0:
            runs_next = runs_per_row[r+1]
            s = 0
            for iv in runs_per_row[r]:
                k = _overlap_count(iv, runs_next)
                if k > 1:
                    s += (k - 1)  # n->k daje (k-1) split "dogadjaja"
            splits[r] = s

        # merges: koliko run-ova u r nastalo spajanjem iz više run-ova u r-1
        if r - 1 >= 0 and black_runs[r] > 0:
            runs_prev = runs_per_row[r-1]
            m = 0
            for iv in runs_per_row[r]:
                k = _overlap_count(iv, runs_prev)
                if k > 1:
                    m += (k - 1)
            merges[r] = m

    return transitions, black_runs, splits, merges

##set feature extraction
def extract_den_dataset(dataset_root, k=4, axis='h', ensure_binary=False, threshold=127):
    """
    Prođe kroz klasne foldere 0..9, uzme sliku i računa gustine crnih piksela
    po segmentima duž zadate ose.

    Parametri:
        dataset_root : str
        k            : int  - broj segmenata
        axis         : {'h','v'}  - 'h' = horizontal_black_density, 'v' = vertical_black_density
        ensure_binary: bool - ako je True, radi cv2.threshold pre računanja
        threshold    : int  - prag za binarizaciju (ako ensure_binary=True)

    Vraća:
        X     : np.ndarray shape (N, k)   - matrica obeležja
        y     : np.ndarray shape (N,)     - labele (0–9)
        paths : list[str] dužine N        - putanje do slika
    """
    axis = axis.lower()
    if axis not in ('h', 'v'):
        raise ValueError("axis mora biti 'h' ili 'v'.")

    dens_fn = horizontal_black_density if axis == 'h' else vertical_black_density

    X, y, paths = [], [], []
    for cls in map(str, range(10)):  # '0'..'9'
        class_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(class_dir):
            continue
        for p in sorted(glob.glob(os.path.join(class_dir, "*.png"))):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if ensure_binary:
                _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            dens = dens_fn(img, k=k)
            X.append(dens)
            y.append(int(cls))
            paths.append(p)

    X = np.vstack(X).astype(np.float32) if X else np.zeros((0, k), dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, paths

## summarize class

def summarize_by_class(X, y, n_classes=10, ddof=1):
    means = []
    stds  = []
    counts = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        idx = (y == c)
        counts[c] = int(idx.sum())
        if counts[c] > 0:
            means.append(X[idx].mean(axis=0))
            stds.append(X[idx].std(axis=0, ddof=ddof))
        else:
            d = X.shape[1]
            means.append(np.full(d, np.nan, dtype=np.float32))
            stds .append(np.full(d, np.nan, dtype=np.float32))
    return np.vstack(means), np.vstack(stds), counts

def plot_class_feature_bars(mean_c, std_c, feature_names=None):
    C, D = mean_c.shape
    xs = np.arange(D)
    plt.figure(figsize=(min(12, 2+D*0.7), 6))
    for c in range(C):
        m, s = mean_c[c], std_c[c]
        plt.errorbar(xs, m, yerr=s, fmt='-o', alpha=0.6, label=f"class {c}")
    plt.xticks(xs, feature_names or [f"f{i}" for i in range(D)], rotation=45, ha='right')
    plt.ylabel("srednja vrednost gustine ± sd")
    plt.title("Sažetak feature-a po klasama")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def plot_class_densities(dataset_root, class_label, limit=12):
    """
    Učita slike iz foldera za datu klasu (npr. dataset_root/3/*.png)
    i prikaže do 'limit' primera koristeći plot_image_with_densities.
    """
    class_dir = os.path.join(dataset_root, str(class_label))
    # pokupi sve fajlove tipa n_n_n.png
    paths = sorted(glob.glob(os.path.join(class_dir, "*.png")))
    if not paths:
        print(f"Nema slika za klasu {class_label} u {class_dir}")
        return

    print(f"Plotujem do {limit} primera za klasu {class_label} (pronađeno {len(paths)} slika)...")
    for p in paths[:limit]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Preskačem {p} (ne može da se učita).")
            continue
        plot_image_with_densities(img)

##utils and preprocessing

def load_image(filepath):
    """
    Učitava sliku sa zadate putanje pomoću OpenCV-a i vraća je kao numpy niz.
    Ako je as_gray=True, slika se učitava u grayscale modu (2D niz).
    """
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(filepath, flag)
    if img is None:
        raise FileNotFoundError(f"Ne mogu da učitam sliku sa putanje: {filepath}")
    return img

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey()

def binarize_image(img, threshold=127, invert=False):
    """
    Binarizuje grayscale sliku. Ako invert=True, crno i belo se zamene.
    Rezultat: 0 i 255.
    """
    if img.ndim != 2:
        raise ValueError("Ocekujem grayscale sliku.")
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, threshold, 255, ttype)
    return binary

if __name__ == "__main__":
    DATASET_ROOT = "data/skeletonized/"
    K = 5

    # X, y, paths = extract_den_dataset(DATASET_ROOT, k=K, axis='v', ensure_binary=False)
    #
    # mean_c, std_c, cnts = summarize_by_class(X, y, n_classes=10, ddof=1)
    # print("Broj uzoraka po klasama:", cnts)
    # print("Dimenzija vektora obeležja:", X.shape[1])
    #
    # feat_names = [f"H{i}" for i in range(K)]  # umesto liste od 2K+5
    # plot_class_feature_bars(mean_c, std_c, feature_names=feat_names)

    image_example = load_image(DATASET_ROOT+'0/0_4_5.png')
    image_example = binarize_image(image_example)
    plot_image_with_densities(image_example)

    plot_class_densities(DATASET_ROOT, class_label=7, limit=12)


