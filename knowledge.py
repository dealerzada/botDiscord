import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


class ProductKnowledge:
    def __init__(self, products_dir="products"):
        self.products_dir = products_dir
        self.documents = []  # Cada archivo .txt completo = 1 documento
        self.vectorizer = None
        self.tfidf_matrix = None
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._load_knowledge()

    def _load_knowledge(self):
        if not os.path.exists(self.products_dir):
            os.makedirs(self.products_dir)
            print(f"⚠️  Crea archivos .txt en ./{self.products_dir}/")
            return

        documents = []
        for filename in os.listdir(self.products_dir):
            if not filename.endswith('.txt'):
                continue
            # Ignorar archivos que no son productos
            if filename in ('requirements.txt',):
                continue
            filepath = os.path.join(self.products_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # 🔑 FIX 1: Incluir el nombre del archivo en el texto indexado
                    # Repetimos el nombre 3 veces para darle peso en TF-IDF
                    name = filename.replace('.txt', '').lower()
                    enriched_text = f"{name} {name} {name}\n{content}"
                    documents.append({
                        "text": content,           # Texto original puro
                        "indexed_text": enriched_text,  # Texto enriquecido para TF-IDF
                        "source": filename
                    })

        if not documents:
            print("⚠️  No hay archivos de productos cargados")
            return

        self.documents = documents
        texts = [d["indexed_text"] for d in documents]

        self.vectorizer = TfidfVectorizer(lowercase=True, max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ {len(documents)} documentos de productos cargados")

    def _extract_prices(self, text: str) -> set:
        """Extrae todos los precios tipo $10, 10 USD, 10$, etc."""
        patterns = [
            r'\$\s*\d+(?:\.\d{1,2})?',
            r'\d+(?:\.\d{1,2})?\s*\$',
            r'\d+(?:\.\d{1,2})?\s*(?:USD|usd)',
        ]
        prices = set()
        for pat in patterns:
            prices.update(re.findall(pat, text))
        return prices

    def _is_generic_query(self, question: str) -> bool:
        """Detecta si la pregunta es genérica sobre productos/planes/precios."""
        q = question.lower()
        generic_terms = {
            "planes", "plan", "precios", "precio", "productos", "producto",
            "menu", "menú", "todos", "lista", "catalogo", "catálogo",
            "cuanto cuestan", "cuánto cuestan", "que tienen", "qué tienen",
            "disponibles", "ofrecen", "tienen", "cuales son", "cuáles son"
        }
        for term in generic_terms:
            if term in q:
                return True
        return False

    def _find_by_filename(self, question: str):
        """🔑 FIX 2: Fallback — busca si la pregunta menciona el nombre de algún archivo."""
        q = question.lower()
        matches = []
        for doc in self.documents:
            name = doc["source"].replace('.txt', '').lower()
            # Coincidencia exacta o que la pregunta contenga el nombre del archivo
            if name in q:
                matches.append(doc)
        return matches

    def ask(self, question: str) -> dict:
        if not self.documents:
            return {
                "answer": "Aún no tengo productos cargados en mi memoria.",
                "sources": [],
                "confidence": 0
            }

        # ─── Pregunta genérica: mostrar TODOS los productos ───
        if self._is_generic_query(question):
            product_docs = [d for d in self.documents if d["source"] != "faq.txt"]
            if not product_docs:
                product_docs = self.documents

            context = "\n\n".join(
                [f"--- {d['source'].replace('.txt','').upper()} ---\n{d['text']}" for d in product_docs]
            )
            sources = [d["source"] for d in product_docs]
            confidence = 1.0
        else:
            # ─── Pregunta específica: usar TF-IDF ───
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()

            SIMILARITY_THRESHOLD = 0.05  # Bajado para captar preguntas cortas
            top_indices = similarities.argsort()[-3:][::-1]
            top_docs = [self.documents[i] for i in top_indices if similarities[i] > SIMILARITY_THRESHOLD]

            # ─── Fallback por nombre de archivo ───
            if not top_docs:
                filename_matches = self._find_by_filename(question)
                if filename_matches:
                    top_docs = filename_matches
                    confidence = 0.95
                else:
                    return {
                        "answer": "Necesito consultar esto con el equipo, un momento por favor 🙏",
                        "sources": [],
                        "confidence": 0
                    }

            context = "\n\n".join(
                [f"--- {d['source'].replace('.txt','').upper()} ---\n{d['text']}" for d in top_docs]
            )
            sources = list(set([d["source"] for d in top_docs]))
            confidence = round(float(similarities[top_indices[0]]), 3) if top_docs and similarities[top_indices[0]] > 0 else 0.95

        # Precios que existen en el contexto real
        context_prices = self._extract_prices(context)

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un vendedor de nuestra tienda. INSTRUCCIONES ABSOLUTAS:\n"
                            "1. Usa ÚNICAMENTE la información del CONTEXTO proporcionado abajo.\n"
                            "2. Si el cliente pregunta por un producto que NO aparece en el CONTEXTO, "
                            "di EXACTAMENTE: 'Necesito consultar esto con el equipo, un momento por favor 🙏'.\n"
                            "3. NO inventes precios. Di EXACTAMENTE los precios que aparecen en el CONTEXTO.\n"
                            "4. NO inventes funciones, características ni planes que no estén en el CONTEXTO.\n"
                            "5. NO digas 'archivo', 'documento', 'contexto' ni 'base de datos' en tu respuesta.\n"
                            "6. Si no hay información suficiente en el CONTEXTO, di EXACTAMENTE: "
                            "'Necesito consultar esto con el equipo, un momento por favor 🙏'.\n"
                            "7. Formato: emojis, listas con viñetas (-), negritas para precios y nombres.\n"
                            "8. Sé directo, no más de 8 líneas.\n"
                            "9. Responde en el idioma del cliente.\n"
                            "10. REGLA DE ORO: Si la pregunta menciona un producto/plan que NO está en el CONTEXTO, "
                            "NO intentes responder. Di la frase exacta de escalación."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"CONTEXTO EXACTO DE NUESTROS PRODUCTOS (USA SOLO ESTO):\n{context}\n\n"
                            f"PREGUNTA DEL CLIENTE: {question}\n\n"
                            f"IMPORTANTE: Si la pregunta es sobre un producto que NO aparece en el CONTEXTO, "
                            f"NO respondas nada más que la frase de escalación. "
                            f"Los únicos precios válidos son: {', '.join(sorted(context_prices)) if context_prices else 'ninguno listado'}."
                        )
                    }
                ],
                temperature=0.0,
                max_tokens=600
            )

            answer = response.choices[0].message.content.strip()

            # Validación: si la respuesta menciona precios que NO están en el contexto, forzar escalación
            answer_prices = self._extract_prices(answer)
            invalid_prices = answer_prices - context_prices
            if invalid_prices and context_prices:
                print(f"⚠️  Precios inventados detectados: {invalid_prices}")
                answer = "Necesito consultar esto con el equipo, un momento por favor 🙏"

        except Exception as e:
            answer = f"❌ Error al consultar la IA: {str(e)}"
            sources = []

        return {"answer": answer, "sources": sources, "confidence": confidence}

    def add_product(self, filename: str, content: str):
        if not filename.endswith('.txt'):
            filename += '.txt'

        filepath = os.path.join(self.products_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        self._load_knowledge()
        return filename