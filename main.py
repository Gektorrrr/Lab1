import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування RGB зображення
image_path = "edited-image.jpg"

bgr = cv2.imread(image_path)
if bgr is None:
    raise FileNotFoundError(
        f"Не можу знайти/прочитати зображення: {image_path}. "
        f"Перевір шлях і назву файлу."
    )

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# 2. Перетворення в градації сірого
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# 3. Сегментація (k-means)
Z = gray.reshape((-1, 1))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
k = 3
ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

labels = labels.reshape(gray.shape)

# Візуалізація сегментації
segmented = (labels * (255 // (k - 1))).astype(np.uint8)

# Відображення
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("RGB")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmentation")
plt.imshow(segmented, cmap="gray")
plt.axis("off")

plt.show()

# Гістограми для кожного сегмента
plt.figure(figsize=(8, 6))

for i in range(k):
    pixels = gray[labels == i]
    plt.hist(pixels, bins=256, alpha=0.5, label=f"Segment {i}")

plt.title("Histograms of Segments")
plt.legend()
plt.show()


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


# Застосування до вашого зображення
img = cv2.imread('edited-image.jpg')

entropy_s = calculate_shannon_entropy(img)
measure_h = calculate_hartley_measure(img)
entropy_m = calculate_markov_entropy(img)

print(f"Ентропія Шенона: {entropy_s:.4f} біт")
print(f"Міра Хартлі: {measure_h:.4f} біт")
print(f"Марковська ентропія (1-го порядку): {entropy_m:.4f} біт")

# 7. Застосувати функції 4–6 до зображення в градаціях сірого (gray)
entropy_s_gray = calculate_shannon_entropy(gray)
measure_h_gray = calculate_hartley_measure(gray)
entropy_m_gray = calculate_markov_entropy(gray)

print("\n=== Пункт 7: Результати для grayscale (gray) ===")
print(f"Ентропія Шенона (gray): {entropy_s_gray:.4f} біт")
print(f"Міра Хартлі (gray): {measure_h_gray:.4f} біт")
print(f"Марковська ентропія 1-го порядку (gray): {entropy_m_gray:.4f} біт")


# 8. Застосувати функції 4–6 до сегментів
print("\n=== Пункт 8: Результати для кожного сегмента ===")

segment_results = []  # (segment_id, n_pixels, shannon, hartley, markov)

for seg_id in range(k):
    mask = (labels == seg_id)

    # Робимо 2D-зображення сегмента: пікселі сегмента залишаємо, решту зануляємо
    # (так markov_entropy працює коректно з 2D, бо в ньому використовуються сусідні пікселі)
    seg_img = np.zeros_like(gray)
    seg_img[mask] = gray[mask]

    n_pixels = int(mask.sum())

    sh = calculate_shannon_entropy(seg_img)
    ha = calculate_hartley_measure(seg_img)
    ma = calculate_markov_entropy(seg_img)

    segment_results.append((seg_id, n_pixels, sh, ha, ma))

    print(f"Сегмент {seg_id}: pixels={n_pixels}, "
          f"Shannon={sh:.4f}, Hartley={ha:.4f}, Markov={ma:.4f}")

# Усереднення результатів по сегментах:
#
# 1) просте середнє (кожен сегмент має однакову вагу)
avg_sh = np.mean([r[2] for r in segment_results])
avg_ha = np.mean([r[3] for r in segment_results])
avg_ma = np.mean([r[4] for r in segment_results])

# 2) зважене середнє (вага = кількість пікселів у сегменті)
total_pixels = sum(r[1] for r in segment_results)
wavg_sh = sum(r[2] * r[1] for r in segment_results) / total_pixels
wavg_ha = sum(r[3] * r[1] for r in segment_results) / total_pixels
wavg_ma = sum(r[4] * r[1] for r in segment_results) / total_pixels

print("\n--- Усереднені значення по сегментах ---")
print(f"Середнє (просте): Shannon={avg_sh:.4f}, Hartley={avg_ha:.4f}, Markov={avg_ma:.4f}")
print(f"Середнє (зважене): Shannon={wavg_sh:.4f}, Hartley={wavg_ha:.4f}, Markov={wavg_ma:.4f}")


# Візуалізація результатів пунктів 7 і 8
#
# 1) Порівняння: grayscale vs average сегментів (просте і зважене)
labels_x = ["Shannon", "Hartley", "Markov"]
vals_gray = [entropy_s_gray, measure_h_gray, entropy_m_gray]
vals_avg = [avg_sh, avg_ha, avg_ma]
vals_wavg = [wavg_sh, wavg_ha, wavg_ma]

x = np.arange(len(labels_x))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, vals_gray, width, label="Gray (п.7)")
plt.bar(x, vals_avg, width, label="Avg сегментів (просте)")
plt.bar(x + width, vals_wavg, width, label="Avg сегментів (зважене)")
plt.xticks(x, labels_x)
plt.title("Порівняння результатів: grayscale vs сегменти")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Окремо: значення по сегментах
seg_ids = [r[0] for r in segment_results]
sh_vals = [r[2] for r in segment_results]
ha_vals = [r[3] for r in segment_results]
ma_vals = [r[4] for r in segment_results]

plt.figure(figsize=(10, 4))
plt.bar(seg_ids, sh_vals)
plt.title("Ентропія Шенона по сегментах")
plt.xlabel("Segment id")
plt.ylabel("bits")
plt.xticks(seg_ids)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(seg_ids, ha_vals)
plt.title("Міра Хартлі по сегментах")
plt.xlabel("Segment id")
plt.ylabel("bits")
plt.xticks(seg_ids)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(seg_ids, ma_vals)
plt.title("Марковська ентропія 1-го порядку по сегментах")
plt.xlabel("Segment id")
plt.ylabel("bits")
plt.xticks(seg_ids)
plt.tight_layout()
plt.show()
