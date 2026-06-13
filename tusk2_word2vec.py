import numpy as np
import gensim.downloader as api
from numpy.linalg import norm

print("Загрузка модели")
model = api.load("glove-wiki-gigaword-50")
print("Модель загружена\n")

allowed_words = [
    'joy', 'happiness', 'delight', 'ecstasy', 'euphoria', 'cheer',
    'amusement', 'gladness', 'bliss', 'triumph', 'optimism',

    'sadness', 'grief', 'sorrow', 'melancholy', 'depression', 'despair',
    'heartbreak', 'misery', 'loneliness', 'hopelessness', 'gloom',

    'anger', 'rage', 'fury', 'frustration', 'annoyance', 'resentment',
    'outrage', 'bitterness', 'hostility', 'irritation',

    'fear', 'terror', 'panic', 'anxiety', 'stress', 'worry',
    'dread', 'nervousness', 'horror', 'apprehension', 'phobia',

    'calm', 'peace', 'relax', 'quiet', 'harmony', 'serenity',
    'tranquility', 'relief', 'comfort', 'stillness', 'zen',

    'love', 'affection', 'passion', 'romance', 'empathy', 'sympathy',
    'compassion', 'tenderness', 'devotion', 'attraction',

    'surprise', 'amazement', 'astonishment', 'shock', 'curiosity',
    'fascination', 'awe', 'wonder', 'anticipation',

    'disgust', 'revulsion', 'contempt', 'shame', 'guilt', 'regret',
    'remorse', 'embarrassment', 'humiliation',

    'apathy', 'boredom', 'indifference', 'confusion', 'doubt',
    'pride', 'trust', 'jealousy', 'envy', 'nostalgia', 'fatigue'
]


def find_closest_in_list(target_vector, exclude_list, topn=1):
    results = []
    for w in allowed_words:
        if w in model and w not in exclude_list:
            w_vec = model[w]
            #скалярное/длины
            sim = np.dot(target_vector, w_vec) / (norm(target_vector) * norm(w_vec))
            results.append((w, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topn]


words = ['peace', 'calm', 'relax', 'quiet', 'harmony']
trigger = 'stress'

print(f"Прямой и обратный эмоциональный сдвиг (триггер: {trigger})")
for word in words:
    forward_vector = model[word] + model[trigger]
    forward_shift = find_closest_in_list(forward_vector, exclude_list=[word, trigger], topn=2)

    backward_vector = model[word] - model[trigger]
    backward_shift = find_closest_in_list(backward_vector, exclude_list=[word, trigger], topn=2)

    print(f"Слово: {word}")
    print(f"  + {trigger} -> {forward_shift[0][0]}, {forward_shift[1][0]} (сходство: {forward_shift[0][1]:.2f}, {forward_shift[1][1]:.2f})")
    print(f"  - {trigger} -> {backward_shift[0][0]}, {backward_shift[1][0]} (сходство: {backward_shift[0][1]:.2f}, {backward_shift[1][1]:.2f})")
print("\n")

emotions = ['happy', 'sad', 'angry']


def get_normalized_vector(word):
    vec = model[word]
    return vec / norm(vec) #делится вектор на его длину, делая его длину равной ровно 1

norm_vectors = {emotion: get_normalized_vector(emotion) for emotion in emotions}

print("Попарное косинусное сходство")
for i in range(len(emotions)):
    for j in range(i + 1, len(emotions)):
        em1, em2 = emotions[i], emotions[j]
        #косинусное сходство = скалярное произведение
        similarity = np.dot(norm_vectors[em1], norm_vectors[em2])
        print(f"Сходство между '{em1}' и '{em2}': {similarity:.4f}")
print("\n")

print("Cмешанные эмоции")
contradictions = [
    ('love', 'hate'),
    ('happy', 'sad'),
    ('peace', 'war')
]

for e1, e2 in contradictions:
    mixed_vector = model[e1] + model[e2]
    mixed_results = find_closest_in_list(mixed_vector, exclude_list=[e1, e2], topn=3)   #три ближ слов эмоций
    mixed_words = [res[0] for res in mixed_results]
    print(f"{e1} + {e2} = {mixed_words}")

#эмоциональьный накал
#нейтрализация
#тревога и неопределенность