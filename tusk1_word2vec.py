import numpy as np

#функция расчета косинусного сходства
def get_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)    #скалярное
    norm1 = np.linalg.norm(vec1)        #длина
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

print("Пункт 1:")
print("Три теста расчета косинусного сходства:")
#сонаправленные векторы (угол 0°, cos = 1)
v1 = [1, 2, 3]
v2 = [2, 4, 6]
print(f"Тест 1 (Одинаковое направление): {get_cosine_similarity(v1, v2):.4f}")

#ортогональные векторы (угол 90°, cos = 0)
v3 = [1, 0]
v4 = [0, 1]
print(f"Тест 2 (Ортогональные): {get_cosine_similarity(v3, v4):.4f}")

#противоположные векторы (угол 180°, cos = -1)
v5 = [1, 1]
v6 = [-1, -1]
print(f"Тест 3 (Противоположные): {get_cosine_similarity(v5, v6):.4f}\n")

#[Главенство, Мужской, Женский, Франция, Россия, Медицина, Здание, Еда]

word_embeddings = {
    "король": np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    "мужчина": np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    "женщина": np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    "королева": np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),

    "париж": np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    "франция": np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    "москва": np.array([0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    "россия": np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),

    "доктор": np.array([0.8, 0.5, 0.5, 0.1, 0.1, 0.9, 0.1, 0.1]),
    "больница": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1]),

    "яблоко": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]),
    "фрукт": np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8])
}

print("косинусное сходство для пяти пар:")
pairs = [
    ("король", "королева"),
    ("франция", "россия"),
    ("доктор", "больница"),
    ("яблоко", "фрукт"),
    ("мужчина", "женщина")
]

for word1, word2 in pairs:
    sim = get_cosine_similarity(word_embeddings[word1], word_embeddings[word2])
    print(f"Сходство ({word1}, {word2}): {sim:.4f}")
print("\n\n")

print("Пункт 2:")

#функция для тестирования пропорциональных словесных аналогий с помощью векторов
def test_analogy(w1, w2, w3, expected_w4, embeddings):
    vec_res = embeddings[w1] - embeddings[w2] + embeddings[w3]
    vec_expected = embeddings[expected_w4]

    sim = get_cosine_similarity(vec_res, vec_expected)
    return sim


print("Поиск аналогий для 5 примеров")
analogies = [
    ("король", "мужчина", "женщина", "королева"),
    ("париж", "франция", "россия", "москва"),
    ("доктор", "мужчина", "женщина", "доктор"),
    ("москва", "россия", "франция", "париж"),
    ("женщина", "королева", "король", "мужчина")
]

for w1, w2, w3, expected in analogies:
    similarity = test_analogy(w1, w2, w3, expected, word_embeddings)
    print(f"{w1} - {w2} + {w3} ≈ {expected} | Косинусная мера: {similarity:.4f}")