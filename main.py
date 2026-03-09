import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1) READ RGB IMAGE
# =============================
IMAGE_PATH = "edited-image.jpg"

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError("Не знайдено зображення")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =============================
# 2) RGB -> GRAY
# =============================
r = img_rgb[:, :, 0].astype(np.float32)
g = img_rgb[:, :, 1].astype(np.float32)
b = img_rgb[:, :, 2].astype(np.float32)

img_gray = 0.299 * r + 0.587 * g + 0.114 * b
img_gray = np.round(img_gray).clip(0, 255).astype(np.uint8)

# =============================
# ФУНКЦІЇ ДЛЯ ПОБУДОВИ ГРАФІКІВ
# =============================
def segment_into_blocks(gray, block):
    H, W = gray.shape
    H2 = (H // block) * block
    W2 = (W // block) * block
    cropped = gray[:H2, :W2]
    blocks = cropped.reshape(H2 // block, block, W2 // block, block).swapaxes(1, 2)
    return blocks, cropped

def compute_features_no_entropy(blocks):
    """
    Без ентропії:
    - variance: для вибору більш "однорідного" сегмента
    - gradient: для вибору більш "деталізованого" сегмента
    """
    bh, bw, bs, _ = blocks.shape
    variance = np.zeros((bh, bw))
    gradient = np.zeros((bh, bw))

    for i in range(bh):
        for j in range(bw):
            blk = blocks[i, j]
            variance[i, j] = np.var(blk)

            gx = np.abs(np.diff(blk.astype(np.int16), axis=1)).mean()
            gy = np.abs(np.diff(blk.astype(np.int16), axis=0)).mean()
            gradient[i, j] = gx + gy

    return variance, gradient

def plot_histograms(pixels, title):
    vals = pixels.flatten()

    plt.figure(figsize=(12, 4))

    # Frequency
    plt.subplot(1, 2, 1)
    plt.hist(vals, bins=256, range=(0, 256))
    plt.title("Частота інтенсивності пікселів")
    plt.xlabel("Інтенсивність (0-255)")
    plt.ylabel("Частота")
    plt.xlim(0, 255)
    plt.ylim(bottom=0)

    # Density
    plt.subplot(1, 2, 2)
    plt.hist(vals, bins=256, range=(0, 256), density=True)
    plt.title("Розподіл інтенсивності пікселів (щільність)")
    plt.xlabel("Інтенсивність (0-255)")
    plt.ylabel("Щільність")
    plt.xlim(0, 255)
    plt.ylim(bottom=0)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# =============================
# Показ RGB і Grayscale
# =============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("RGB зображення")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Зображення у відтінках сірого")
plt.axis("off")

plt.tight_layout()
plt.show()

# Гістограми для whole image
plot_histograms(img_gray, "Гістограма всього зображення (фіксована вісь X: 0..255)")

# =============================
# 3) SEGMENTATION (8 / 16 / 32)
# =============================
for BLOCK_SIZE in [8, 16, 32]:
    blocks, cropped = segment_into_blocks(img_gray, BLOCK_SIZE)
    variance, gradient = compute_features_no_entropy(blocks)

    # 1) First segment = мін variance (найбільш однорідний)
    first_idx = np.unravel_index(np.argmin(variance), variance.shape)

    # 2) Third segment = макс gradient (найбільш деталізований)
    third_idx = np.unravel_index(np.argmax(gradient), gradient.shape)

    # 3) Second segment = "середній" (візьмемо блок з variance ближче до медіани)
    var_flat = variance.flatten()
    median_var = np.median(var_flat)
    second_flat_idx = np.argmin(np.abs(var_flat - median_var))
    second_idx = np.unravel_index(second_flat_idx, variance.shape)

    segments = {
        "Перший сегмент": blocks[first_idx],
        "Другий сегмент": blocks[second_idx],
        "Третій сегмент": blocks[third_idx]
    }

    # ---- Показ розбиття на блоки (рамки) ----
    preview = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    for name, (i, j) in zip(segments.keys(), [first_idx, second_idx, third_idx]):
        y0 = i * BLOCK_SIZE
        x0 = j * BLOCK_SIZE
        cv2.rectangle(
            preview,
            (x0, y0),
            (x0 + BLOCK_SIZE - 1, y0 + BLOCK_SIZE - 1),
            (0, 0, 255),
            2
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.title(f"Сегментація {BLOCK_SIZE}x{BLOCK_SIZE} (3 вибрані блоки)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---- Показ кожного сегмента + гістограм ----
    for name, block in segments.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(block, cmap="gray")
        plt.title(f"{name} ({BLOCK_SIZE}x{BLOCK_SIZE})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        plot_histograms(block, f"{name} - Гістограма ({BLOCK_SIZE}x{BLOCK_SIZE})")

# 4. Функція розрахування ентропії Шенона
def calculate_shannon_entropy(image):
    # Отримуємо гістограму і нормалізуємо її для отримання ймовірностей
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    probs = hist.flatten() / hist.sum()
    # Відфільтровуємо нульові значення для уникнення log(0)
    probs = probs[probs > 0]
    # Формула: H = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


# 5. Функція розрахування міри Хартлі
def calculate_hartley_measure(image):
    # Міра Хартлі базується на кількості унікальних станів (рівнів яскравості)
    unique_values = np.unique(image)
    N = len(unique_values)
    if N > 0:
        return np.log2(N)
    return 0


# 6. Функція розрахування Марковського процесу першого порядку (умовна ентропія)
def calculate_markov_entropy(image):
    # Розраховуємо умовну ентропію H(X|Y) = H(X,Y) - H(Y)
    # Розглядаємо горизонтальні пари сусідніх пікселів
    y = image[:, :-1].flatten()  # поточний піксель
    x = image[:, 1:].flatten()  # наступний піксель

    # Спільна ймовірність p(x, y) через 2D гістограму
    joint_hist, _, _ = np.histogram2d(y, x, bins=256, range=[[0, 256], [0, 256]])
    joint_probs = joint_hist.flatten() / joint_hist.sum()
    joint_probs = joint_probs[joint_probs > 0]
    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))

    # Маргінальна ймовірність p(y)
    marginal_probs = joint_hist.sum(axis=1) / joint_hist.sum()
    marginal_probs = marginal_probs[marginal_probs > 0]
    marginal_entropy = -np.sum(marginal_probs * np.log2(marginal_probs))

    # Умовна ентропія
    conditional_entropy = joint_entropy - marginal_entropy
    return conditional_entropy

# =============================
# 7) Застосувати функції 4–6 до grayscale (img_gray)
# =============================

# 7.1 Розрахунок для grayscale
sh_gray = calculate_shannon_entropy(img_gray)
ha_gray = calculate_hartley_measure(img_gray)
mk_gray = calculate_markov_entropy(img_gray)

print("\n=== Пункт 7: Метрики для grayscale (img_gray) ===")
print(f"Shannon (gray): {sh_gray:.4f} bits")
print(f"Hartley (gray): {ha_gray:.4f} bits")
print(f"Markov  (gray): {mk_gray:.4f} bits")

# 7.2 Візуалізація: зліва зображення, справа стовпчики
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Усе зображення (відтінки сірого)")
plt.axis("off")

plt.subplot(1, 2, 2)
metrics_names = ["Шеннон", "Хартлі", "Марков"]
metrics_vals = [sh_gray, ha_gray, mk_gray]
plt.bar(metrics_names, metrics_vals)
plt.title("Значення ентропій (усе зображення)")
plt.ylabel("Значення (біти)")
for i, v in enumerate(metrics_vals):
    plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# =============================
# 8) Застосувати функції 4–6 до сегментів
# IMPORTANT:
# - беремо сегменти, розміру 16 х 16
# =============================

# 8.0 Спочатку ще раз отримаємо сегменти саме для BLOCK_SIZE=16
BLOCK_SIZE_FOR_P8 = 16
blocks16, cropped16 = segment_into_blocks(img_gray, BLOCK_SIZE_FOR_P8)
variance16, gradient16 = compute_features_no_entropy(blocks16)

first_idx16 = np.unravel_index(np.argmin(variance16), variance16.shape)
third_idx16 = np.unravel_index(np.argmax(gradient16), gradient16.shape)

var_flat16 = variance16.flatten()
median_var16 = np.median(var_flat16)
second_flat_idx16 = np.argmin(np.abs(var_flat16 - median_var16))
second_idx16 = np.unravel_index(second_flat_idx16, variance16.shape)

segments16 = {
    "Перший сегмент": blocks16[first_idx16],
    "Другий сегмент": blocks16[second_idx16],
    "Третій сегмент": blocks16[third_idx16]
}

print("\n=== Пункт 8: Метрики для сегментів (BLOCK_SIZE=16) ===")

# 8.1 Для кожного сегмента: зліва сегмент, справа відображення 3 метрик
for seg_name, seg_img in segments16.items():

    sh = calculate_shannon_entropy(seg_img)
    ha = calculate_hartley_measure(seg_img)
    mk = calculate_markov_entropy(seg_img)

    print(f"{seg_name}: Shannon={sh:.4f}, Hartley={ha:.4f}, Markov={mk:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(seg_img, cmap="gray")
    plt.title(f"{seg_name} (16x16)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    vals = [sh, ha, mk]
    plt.bar(metrics_names, vals)
    plt.title(f"Значення ентропій ({seg_name})")
    plt.ylabel("Значення (біти)")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()