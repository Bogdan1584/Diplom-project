import numpy as np
import pandas as pd
import time
import warnings
import re
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel

# Настройка matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

print("Проверка ресурсов NLTK...")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# ГЛАВА 2.1 ЗАГРУЗКА ДАТАСЕТА


class SMSLoader:
    def load(self):
        print("\n" + "="*80)
        print("ГЛАВА 2.1: ЗАГРУЗКА И АНАЛИЗ ДАТАСЕТА")
        print("="*80)

        start = time.time()
        df = pd.read_csv("spam.csv", encoding="latin-1")
        df = df[['v1','v2']]
        df.columns = ['label','message']
        load_time = time.time() - start

        total = len(df)
        spam = (df.label == "spam").sum()
        ham = (df.label == "ham").sum()

        print("\nОбщая информация о датасете:")
        print(f"  Всего сообщений: {total}")
        print(f"  Спам (spam): {spam} ({spam/total*100:.1f}%)")
        print(f"  Не спам (ham): {ham} ({ham/total*100:.1f}%)")

        df['length'] = df['message'].str.len()
        print(f"\nСтатистика длины сообщений:")
        print(f"  Средняя: {df['length'].mean():.1f} символов")
        print(f"  Медиана: {df['length'].median():.1f} символов")
        print(f"  Минимум: {df['length'].min()}")
        print(f"  Максимум: {df['length'].max()}")
        print(f"  Время загрузки: {load_time:.2f} сек")

        print("\nПримеры сообщений:")
        print("\nHAM (не спам):")
        for m in df[df.label=="ham"].message.iloc[:2]:
            print(f"  → {m[:100]}...")

        print("\nSPAM (спам):")
        for m in df[df.label=="spam"].message.iloc[:2]:
            print(f"  → {m[:100]}...")

        return df


# ПРЕДОБРАБОТКА


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def preprocess_corpus(self, corpus):
        print("\nПредобработка текстов...")
        start = time.time()
        processed = []
        for i, text in enumerate(corpus):
            if i % 1000 == 0:
                print(f"  Обработано {i} из {len(corpus)}")
            processed.append(self.preprocess(text))
        prep_time = time.time() - start
        print(f"  Готово! Обработано {len(processed)} сообщений")
        print(f"  Время предобработки: {prep_time:.2f} сек")
        return processed, prep_time


# ГЛАВА 2.2 КЛАССИЧЕСКИЕ МЕТОДЫ


class ClassicMethods:
    def apply_bow(self, texts):
        print("\n" + "="*80)
        print("ГЛАВА 2.2.1 BAG OF WORDS (BoW)")
        print("="*80)

        start = time.time()

        vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )

        X = vectorizer.fit_transform(texts)
        vec_time = time.time() - start

        print(f"  Размерность: {X.shape}")
        print(f"  Разреженность: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
        print(f"  Время векторизации: {vec_time:.2f} сек")

        return X, vectorizer, vec_time

    def apply_tfidf(self, texts):
        print("\n" + "="*80)
        print("ГЛАВА 2.2.2 TF-IDF")
        print("="*80)

        start = time.time()

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )

        X = vectorizer.fit_transform(texts)
        vec_time = time.time() - start

        print(f"  Размерность: {X.shape}")
        print(f"  Время векторизации: {vec_time:.2f} сек")

        return X, vectorizer, vec_time


# ГЛАВА 2.3 ПЛОТНЫЕ МЕТОДЫ


class DenseMethods:
    def apply_word2vec(self, texts, vector_size=100):
        print("\n" + "="*80)
        print("ГЛАВА 2.3.1 WORD2VEC (статический эмбеддинг)")
        print("="*80)

        start = time.time()

        sentences = [text.split() for text in texts]
        sentences = [s for s in sentences if len(s) > 0]

        print(f"  Обучение Word2Vec на {len(sentences)} предложениях...")

        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=10
        )
        train_time = time.time() - start

        print(f"  Размер словаря: {len(model.wv)} слов")
        print(f"  Размерность эмбеддинга: {vector_size}")

        def get_doc_vector(text):
            words = text.split()
            vectors = [model.wv[w] for w in words if w in model.wv]
            if len(vectors) == 0:
                return np.zeros(vector_size)
            return np.mean(vectors, axis=0)

        start_vec = time.time()
        X = np.array([get_doc_vector(t) for t in texts])
        vec_time = time.time() - start_vec

        total_time = time.time() - start
        print(f"  Итоговая размерность: {X.shape}")
        print(f"  Время обучения: {train_time:.2f} сек")
        print(f"  Время векторизации: {vec_time:.2f} сек")
        print(f"  Общее время: {total_time:.2f} сек")

        return X, model, total_time

    def apply_bert(self, texts, max_samples=2000):
        print("\n" + "="*80)
        print("ГЛАВА 2.3.2 BERT (контекстуальный эмбеддинг)")
        print("="*80)

        start = time.time()

        model_name = "bert-base-uncased"
        
        print(f"  Загрузка модели {model_name}...")
        load_start = time.time()
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        load_time = time.time() - load_start

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        bert_model.eval()

        print(f"  Устройство: {device}")
        print(f"  Время загрузки модели: {load_time:.2f} сек")
        print(f"  Обрабатывается {min(max_samples, len(texts))} сообщений...")

        def get_bert_embedding(text):
            encoded = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embedding[0]

        X_list = []
        for i, text in enumerate(texts[:max_samples]):
            if i % 200 == 0:
                print(f"    Обработано {i} из {min(max_samples, len(texts))}")
            X_list.append(get_bert_embedding(text))

        X = np.array(X_list)
        total_time = time.time() - start

        print(f"  Итоговая размерность: {X.shape}")
        print(f"  Время загрузки модели: {load_time:.2f} сек")
        print(f"  Время векторизации: {total_time - load_time:.2f} сек")
        print(f"  Общее время: {total_time:.2f} сек")

        return X, total_time


# ОЦЕНКА МОДЕЛЕЙ


class Evaluator:
    def evaluate(self, X, y, name):
        print(f"\n{'='*60}")
        print(f"ОЦЕНКА: {name}")
        print('='*60)

        if hasattr(X, "toarray"):
            X = X.toarray()

        start = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        train_time = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam")
        rec = recall_score(y_test, y_pred, pos_label="spam")
        f1 = f1_score(y_test, y_pred, pos_label="spam")

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  Время обучения и оценки: {train_time:.2f} сек")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "train_time": train_time}

    def evaluate_with_cv(self, X, y, name, cv_folds=5):
        """Оценка с помощью стратифицированной кросс-валидации"""
        print(f"\n{'='*60}")
        print(f"КРОСС-ВАЛИДАЦИЯ ({cv_folds} folds): {name}")
        print('='*60)

        start = time.time()

        if hasattr(X, "toarray"):
            X = X.toarray()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, C=1.0)

        # Стратифицированная кросс-валидация
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Вычисляем метрики для каждого фолда
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Для precision, recall, f1 нужно передавать pos_label через make_scorer
        from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
        
        prec_scorer = make_scorer(precision_score, pos_label='spam')
        rec_scorer = make_scorer(recall_score, pos_label='spam')
        f1_scorer = make_scorer(f1_score, pos_label='spam')
        
        prec_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=prec_scorer)
        rec_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=rec_scorer)
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer)

        cv_time = time.time() - start

        print(f"  Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
        print(f"  Precision: {prec_scores.mean():.4f} (+/- {prec_scores.std():.4f})")
        print(f"  Recall: {rec_scores.mean():.4f} (+/- {rec_scores.std():.4f})")
        print(f"  F1-score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print(f"  Время выполнения: {cv_time:.2f} сек")

        return {
            "accuracy": acc_scores.mean(),
            "accuracy_std": acc_scores.std(),
            "precision": prec_scores.mean(),
            "precision_std": prec_scores.std(),
            "recall": rec_scores.mean(),
            "recall_std": rec_scores.std(),
            "f1": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "cv_time": cv_time
        }

    def analyze_coefficients(self, X, y, vectorizer, name):
        """Анализ коэффициентов логистической регрессии (для интерпретируемости)"""
        print(f"\n{'='*60}")
        print(f"ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ: {name}")
        print('='*60)
        
        if hasattr(X, "toarray"):
            X = X.toarray()
        
        # Разбиение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Масштабирование
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Обучение модели
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)
        
        # Получение коэффициентов для класса "spam"
        coefficients = model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        
        # Топ-10 слов, связанных со спамом (наибольшие положительные коэффициенты)
        top_spam_idx = np.argsort(coefficients)[-10:][::-1]
        print("\nТоп-10 слов, характерных для СПАМА:")
        for idx in top_spam_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
        
        # Топ-10 слов, связанных с легитимными сообщениями (наименьшие/отрицательные коэффициенты)
        top_ham_idx = np.argsort(coefficients)[:10]
        print("\nТоп-10 слов, характерных для ЛЕГИТИМНЫХ сообщений (НЕ СПАМ):")
        for idx in top_ham_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
        
        return model, coefficients
# ВИЗУАЛИЗАЦИЯ


def visualize_embeddings(X_bow, X_tfidf, X_w2v, y, sample_size=500):
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ (PCA)")
    print("="*80)

    y_numeric = [1 if lbl == "spam" else 0 for lbl in y[:sample_size]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # BoW
    if hasattr(X_bow, "toarray"):
        X_bow_dense = X_bow[:sample_size].toarray()
    else:
        X_bow_dense = X_bow[:sample_size]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_bow_dense)
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', alpha=0.6, s=10)
    axes[0].set_title(f'BoW (PCA, {pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # TF-IDF
    if hasattr(X_tfidf, "toarray"):
        X_tfidf_dense = X_tfidf[:sample_size].toarray()
    else:
        X_tfidf_dense = X_tfidf[:sample_size]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf_dense)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', alpha=0.6, s=10)
    axes[1].set_title(f'TF-IDF (PCA, {pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    # Word2Vec
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_w2v[:sample_size])
    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', alpha=0.6, s=10)
    axes[2].set_title(f'Word2Vec (PCA, {pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')

    plt.tight_layout()
    plt.savefig('embeddings_pca.png', dpi=150, bbox_inches='tight')
    print("  График сохранен как 'embeddings_pca.png'")
    plt.show()


def visualize_bert(X_bert, y, sample_size=500):
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ BERT (PCA)")
    print("="*80)

    X_sample = X_bert[:min(sample_size, len(X_bert))]
    y_sample = y[:min(sample_size, len(X_bert))]
    
    y_numeric = [1 if lbl == "spam" else 0 for lbl in y_sample]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', alpha=0.6, s=10)
    plt.title(f'BERT (PCA, {pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Spam'),
                       Patch(facecolor='blue', alpha=0.6, label='Ham')]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('bert_pca.png', dpi=150, bbox_inches='tight')
    print("  График BERT сохранен как 'bert_pca.png'")
    plt.show()


# MAIN


def main():
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ВЕКТОРИЗАЦИИ")
    print("ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА")
    print("="*80)

    # 2.1 Загрузка данных
    loader = SMSLoader()
    df = loader.load()

    # Предобработка
    preprocessor = TextPreprocessor()
    texts_processed, prep_time = preprocessor.preprocess_corpus(df.message.tolist())
    labels = df.label.tolist()

    # 2.2 Классические методы
    classic = ClassicMethods()
    X_bow, _, bow_time = classic.apply_bow(texts_processed)
    X_tfidf, _, tfidf_time = classic.apply_tfidf(texts_processed)
    # Анализ интерпретируемости (топ-10 слов)
    evaluator = Evaluator()
    evaluator.analyze_coefficients(X_bow, labels, bow_vectorizer, "Bag of Words")
    evaluator.analyze_coefficients(X_tfidf, labels, tfidf_vectorizer, "TF-IDF")
    
    # 2.3 Плотные методы
    dense = DenseMethods()
    
    # Word2Vec
    X_w2v, _, w2v_time = dense.apply_word2vec(texts_processed, vector_size=100)
    
    # BERT
    X_bert, bert_time = dense.apply_bert(texts_processed, max_samples=2000)
    labels_bert = labels[:len(X_bert)]

    # 2.4 Визуализация
    visualize_embeddings(X_bow, X_tfidf, X_w2v, labels, sample_size=500)
    visualize_bert(X_bert, labels_bert, sample_size=500)

    # 2.5 Сравнительная оценка
    evaluator = Evaluator()

    results = {}

    print("\n" + "="*80)
    print("2.5 СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)

    results["BoW"] = evaluator.evaluate(X_bow, labels, "Bag of Words")
    results["TF-IDF"] = evaluator.evaluate(X_tfidf, labels, "TF-IDF")
    results["Word2Vec"] = evaluator.evaluate(X_w2v, labels, "Word2Vec")
    results["BERT"] = evaluator.evaluate(X_bert, labels_bert, "BERT")

    # ДОПОЛНИТЕЛЬНО: Оценка с кросс-валидацией
    print("\n" + "="*80)
    print("ДОПОЛНИТЕЛЬНАЯ ОЦЕНКА С КРОСС-ВАЛИДАЦИЕЙ (5-FOLD)")
    print("="*80)

    cv_results = {}
    cv_results["BoW"] = evaluator.evaluate_with_cv(X_bow, labels, "Bag of Words")
    cv_results["TF-IDF"] = evaluator.evaluate_with_cv(X_tfidf, labels, "TF-IDF")
    cv_results["Word2Vec"] = evaluator.evaluate_with_cv(X_w2v, labels, "Word2Vec")
    cv_results["BERT"] = evaluator.evaluate_with_cv(X_bert, labels_bert, "BERT")

    # Итоговая таблица с метриками и временем
    print("\n" + "="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ (FIXED SPLIT 80/20)")
    print("="*80)

    comparison = pd.DataFrame(results).T
    comparison = comparison.round(4)
    print("\n" + comparison.to_string())
    
    # Таблица времени выполнения
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ПО ВРЕМЕНИ ВЫПОЛНЕНИЯ (сек)")
    print("="*80)
    
    time_data = {
        "Метод": ["BoW", "TF-IDF", "Word2Vec", "BERT"],
        "Время векторизации": [bow_time, tfidf_time, w2v_time, bert_time],
        "Время обучения модели": [results[m]["train_time"] for m in ["BoW", "TF-IDF", "Word2Vec", "BERT"]]
    }
    time_df = pd.DataFrame(time_data)
    time_df["Общее время"] = time_df["Время векторизации"] + time_df["Время обучения модели"]
    print(time_df.to_string(index=False))
    
    # Таблица результатов кросс-валидации
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ (5-FOLD)")
    print("="*80)
    
    cv_comparison = pd.DataFrame(cv_results).T
    cv_comparison = cv_comparison.round(4)
    print("\n" + cv_comparison[["accuracy", "accuracy_std", "f1", "f1_std"]].to_string())
    
    # Сохранение результатов
    comparison.to_csv('results_comparison.csv')
    time_df.to_csv('time_comparison.csv', index=False)
    cv_comparison.to_csv('cv_results.csv')
    print("\nРезультаты сохранены в:")
    print("  - results_comparison.csv (фиксированное разбиение)")
    print("  - time_comparison.csv (время выполнения)")
    print("  - cv_results.csv (кросс-валидация)")

    best_method = comparison['accuracy'].idxmax()
    print(f"\nЛучший метод по accuracy (fixed split): {best_method} с результатом {comparison.loc[best_method, 'accuracy']:.4f}")

    best_cv_method = cv_comparison['accuracy'].idxmax()
    print(f"Лучший метод по accuracy (CV): {best_cv_method} с результатом {cv_comparison.loc[best_cv_method, 'accuracy']:.4f}")

    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("="*80)


if __name__ == "__main__":
    main()