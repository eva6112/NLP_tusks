import numpy as np

d = 3
m = 2

vocab = {
    "я": np.array([0.1, 0.5, -0.2]),
    "пытаюсь": np.array([0.3, 0.1, 0.4]),
    "реализовать": np.array([0.6, 0.7, -0.1]),
    "эту": np.array([0.0, 0.2, 0.1]),
    "задачу": np.array([0.4, -0.3, 0.8])
}

#инициализация параметоров RNN
W_xh = np.array([[0.5, -0.2, 0.1],          #как вход влияет на память
                 [-0.1, 0.6, -0.3]])

W_hh = np.array([[0.8, 0.1],                #как прошлая память влияет на новую
                 [0.0, 0.5]])

b_h = np.array([0.1, -0.1])                 #смещение скрытого слоя для активации


def rnn_forward_pass(sentence_words):
    h_t = np.zeros(m)

    print(f"Обработка предложения: {' '.join(sentence_words)}")

    for t, word in enumerate(sentence_words):
        if word not in vocab:
            print(f"Слово '{word}' не найдено в словаре, пропускаем.")
            continue

        x_t = vocab[word]

        #h_t = tanh(W_xh * x_t + W_hh * h_t-1 + b_h)
        linear_combination = np.dot(W_xh, x_t) + np.dot(W_hh, h_t) + b_h
        h_t = np.tanh(linear_combination)       #функция активации (-1 до 1)

        print(f"Шаг {t + 1} (слово '{word}'): скрытое состояние h_{t + 1} = {h_t}")

    return h_t


def predict_sentiment(final_h):
    score = np.sum(final_h)
    if score > 0:
        return f"Позитивная тональность (score: {score:.3f})"
    else:
        return f"Негативная/Нейтральная тональность (score: {score:.3f})"

def analyze_voice_emotion(volume, speed):
    if volume > 7 and speed > 7:
        return "Гнев или сильное возбуждение"
    elif volume < 4 and speed < 4:
        return "Грусть или усталость"
    elif volume > 6 and speed >= 4 and speed <= 7:
        return "Радость / Уверенность"
    else:
        return "Нейтральная эмоция"

if __name__ == "__main__":
    sentence = ["я", "пытаюсь", "реализовать", "эту", "задачу"]
    final_memory_state = rnn_forward_pass(sentence)

    print("-----")
    sentiment_result = predict_sentiment(final_memory_state)
    print(f"Результат анализа текста: {sentiment_result}")

    print("\n" + "-----" + "\n")

    #анализ голоса
    user_volume = 5.0
    user_speed = 5.0

    voice_result = analyze_voice_emotion(user_volume, user_speed)
    print(f"Анализ голоса (громкость: {user_volume}, темп: {user_speed}):")
    print(f"Распознанная эмоция: {voice_result}")