# RAG - Generator Module (Gemini v1.5 Flash compatible + smart fallback)
# Uses the Google Gemini `google.generativeai` client when a GEMINI_API_KEY is configured.

import os
import re
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    _gemini_available = True
except ImportError:
    _gemini_available = False


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generator
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates structured answers using Google Gemini API with RAG context.
    Falls back to an intelligent rule-based answer builder if no API key.
    """

    SYSTEM_PROMPT = (
        "You are an expert AI Teaching Assistant specialising in Computer Science. "
        "Your job is to give clear, accurate, student-friendly explanations "
        "based ONLY on the provided course material context. "
        "Always structure your answer with: "
        "1) a clear definition, "
        "2) a concise explanation, "
        "3) a real-world or academic example where relevant, "
        "4) key points to remember. "
        "Be precise and never hallucinate facts outside the context."
    )

    def __init__(self,
                 api_key: str = None,
                 model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model   = model   or os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")
        self._client = None

        if self.api_key and _gemini_available:
            try:
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
                print("  [Generator] Gemini API initialised")
            except Exception as e:
                print(f"  [Generator] Gemini init failed: {e}")
                self._client = None

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a clean context block for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            content  = chunk.get('content', '').strip()
            meta     = chunk.get('metadata', {})
            score    = chunk.get('relevance_score', 0.0)
            subject  = meta.get('subject', 'N/A')
            topic    = meta.get('topic', 'N/A')
            source   = meta.get('source_file', meta.get('source', 'course material'))
            parts.append(
                f"[Source {i}] {subject} → {topic} | relevance={score:.2f} | file={source}\n"
                f"{content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Gemini answer ─────────────────────────────────────────────────────────

    def _gemini_answer(self, query: str, context: str) -> str:
        """Call Gemini with structured RAG prompt."""
        full_prompt = f"""{self.SYSTEM_PROMPT}

Use ONLY the course material below to answer the question.
If the context does not cover the topic, say so clearly.

══ COURSE MATERIAL CONTEXT ══
{context}

══ STUDENT QUESTION ══
{query}

══ YOUR ANSWER ══
Structure your response:
**Definition:** (1-2 sentences)
**Explanation:** (clear, detailed)
**Example:** (concrete example from the material if available)
**Key Points:**
• point 1
• point 2
• point 3
"""
        response = self._client.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1200,
                top_p=0.9,
            )
        )
        return response.text.strip()

    # ── Intelligent fallback (no API) ─────────────────────────────────────────

    def _smart_fallback(self, query: str, chunks: List[Dict]) -> str:
        """
        When there is no Gemini API key or the Gemini call fails: build a proper
        structured answer by intelligently extracting and organising content
        from the provided chunks.
        """
        if not chunks:
            return (
                f"I could not find specific information about **'{query}'** "
                "in the current course materials. "
                "Please make sure the relevant PDFs have been processed and trained."
            )

        query_lower  = query.lower()
        query_words  = set(re.sub(r'[^\w\s]', '', query_lower).split())
        stop_words   = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in',
                        'to', 'how', 'does', 'do', 'explain', 'define',
                        'describe', 'give', 'me', 'tell', 'about', 'with'}
        key_words    = query_words - stop_words

        # ── Score every sentence across all chunks ────────────────────────────
        all_sentences = []
        for chunk in chunks:
            content = chunk.get('content', '')
            meta    = chunk.get('metadata', {})
            for sent in re.split(r'(?<=[.!?])\s+', content):
                sent = sent.strip()
                if len(sent.split()) < 5:
                    continue
                sent_lower = sent.lower()
                score = sum(1 for kw in key_words if kw in sent_lower)
                all_sentences.append((score, sent, meta))

        all_sentences.sort(key=lambda x: x[0], reverse=True)

        # ── Grab best sentences for each role ─────────────────────────────────
        def _pick(n=3, min_score=0) -> List[str]:
            seen = set()
            result = []
            for sc, sent, _ in all_sentences:
                if sc < min_score:
                    continue
                norm = sent[:60]
                if norm in seen:
                    continue
                seen.add(norm)
                result.append(sent)
                if len(result) >= n:
                    break
            return result

        top_sentences = _pick(6, min_score=1) or _pick(6, min_score=0)

        # Subject / topic from best chunk
        best_meta    = chunks[0].get('metadata', {})
        subject      = best_meta.get('subject', '')
        topic        = best_meta.get('topic', '').replace('_', ' ').title()
        source_files = list({c.get('metadata', {}).get('source_file', 'course material')
                             for c in chunks})

        # ── Build structured answer ────────────────────────────────────────────
        answer_parts = []

        # Header
        clean_query = query.strip().rstrip('?')
        answer_parts.append(f"### {clean_query}\n")

        # Definition (best-matching sentence)
        if top_sentences:
            answer_parts.append(f"**Definition / Overview:**\n{top_sentences[0]}\n")
        
        # Explanation (next 2-3 sentences)
        if len(top_sentences) > 1:
            explanation = ' '.join(top_sentences[1:4])
            answer_parts.append(f"**Explanation:**\n{explanation}\n")

        # Key Points (remaining sentences as bullets)
        if len(top_sentences) > 4:
            bullet_sents = top_sentences[4:]
            bullets = '\n'.join(f"• {s}" for s in bullet_sents)
            answer_parts.append(f"**Key Points:**\n{bullets}\n")

        # Example detection (look for "for example", "e.g.", "consider", "such as")
        example_sents = [s for _, s, _ in all_sentences
                         if any(kw in s.lower()
                                for kw in ['for example', 'e.g.', 'consider', 'such as', 'instance'])]
        if example_sents:
            answer_parts.append(f"**Example:**\n{example_sents[0]}\n")

        # Source note
        subject_str = f" ({subject})" if subject else ""
        topic_str   = f" — {topic}" if topic else ""
        answer_parts.append(
            f"\n*📚 Sources: {', '.join(source_files[:3])}{subject_str}{topic_str}*"
        )

        return '\n\n'.join(answer_parts)

    # ── Main public method ────────────────────────────────────────────────────

    def generate_answer(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate a structured answer using Gemini if available, else smart fallback.
        """
        answer = ""

        if self._client:
            try:
                context = self._build_context(chunks)
                answer  = self._gemini_answer(query, context)
                model_used = self.model
            except Exception as e:
                print(f"  [Generator] Gemini call failed: {e} — using fallback")
                answer     = self._smart_fallback(query, chunks)
                model_used = "smart_fallback"
        else:
            answer     = self._smart_fallback(query, chunks)
            model_used = "smart_fallback"

        return {
            'answer':       answer,
            'model':        model_used,
            'context_used': len(chunks),
            'timestamp':    datetime.now().isoformat(),
            'sources':      [c.get('metadata', {}).get('source_file',
                               c.get('metadata', {}).get('source', 'unknown'))
                             for c in chunks],
        }

    # ── Quality evaluator ─────────────────────────────────────────────────────

    def evaluate_answer_quality(self, answer: str,
                                query: str,
                                chunks: List[Dict]) -> Dict[str, float]:
        """Heuristic quality scoring for a generated answer."""
        words        = answer.split()
        word_count   = len(words)
        has_def      = '**Definition' in answer or 'definition' in answer.lower()
        has_example  = any(kw in answer.lower()
                           for kw in ['example', 'e.g.', 'for instance', 'consider'])
        has_bullets  = '•' in answer or '- ' in answer or any(
                           f'{i}.' in answer for i in range(1, 6))
        has_sources  = '📚 Sources' in answer or 'Source' in answer
        context_rel  = len(chunks) > 0

        # Keyword coverage
        query_words  = set(w.lower() for w in query.split() if len(w) > 3)
        answer_lower = answer.lower()
        coverage     = sum(1 for w in query_words if w in answer_lower)
        kw_score     = coverage / len(query_words) if query_words else 0.0

        quality = (
            0.25 * float(context_rel) +
            0.20 * float(has_def)     +
            0.15 * kw_score           +
            0.15 * float(has_example) +
            0.10 * float(has_bullets) +
            0.10 * min(1.0, word_count / 150) +
            0.05 * float(has_sources)
        )

        return {
            'quality_score':     round(quality, 4),
            'answer_length':     word_count,
            'context_relevance': context_rel,
            'has_definition':    has_def,
            'has_examples':      has_example,
            'has_structure':     has_bullets,
            'keyword_coverage':  round(kw_score, 4),
        }


if __name__ == "__main__":
    gen = AnswerGenerator()
    status = "Gemini API" if gen._client else "Smart Fallback"
    print(f"Answer generator ready — mode: {status}")