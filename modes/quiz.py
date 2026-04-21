# Quiz Generation Module (v2 — GPT-powered real MCQs with smart fallback)
# Generates meaningful, subject-specific questions from course material.

from typing import List, Dict, Any, Optional
import re
import random
from datetime import datetime


class QuizGenerator:
    """
    Generates quiz questions using GPT (when available) with an intelligent
    keyword/sentence-based fallback for offline use.
    """

    def __init__(self, generator=None):
        # Accept an AnswerGenerator instance (for GPT access)
        self._generator = generator
        self._try_load_generator()

    def _try_load_generator(self):
        if self._generator is None:
            try:
                from rag.generator import AnswerGenerator
                self._generator = AnswerGenerator()
            except Exception:
                self._generator = None

    # ── GPT Quiz caller ───────────────────────────────────────────────────────

    def _gpt_generate_questions(self, topic: str, context: str,
                                 num: int, q_type: str) -> List[Dict]:
        """Ask GPT to generate well-formed questions from context."""
        if not (self._generator and getattr(self._generator, '_client', None)):
            return []

        if q_type == 'mcq':
            fmt = (
                f"Generate {num} multiple-choice questions about '{topic}' "
                f"from the text below. Each question must have:\n"
                f"- Question text\n"
                f"- 4 options labeled A) B) C) D)\n"
                f"- Correct answer label\n"
                f"Format each as:\n"
                f"Q: <question>\nA) <opt>\nB) <opt>\nC) <opt>\nD) <opt>\nAnswer: <letter>\n\n"
                f"Text:\n{context}"
            )
        else:
            fmt = (
                f"Generate {num} short-answer questions about '{topic}' "
                f"from the text below. Each question should test understanding, "
                f"not memory. Format: Q: <question>\nAnswer: <brief answer>\n\n"
                f"Text:\n{context}"
            )

        try:
            resp = self._generator._client.chat.completions.create(
                model=self._generator.model,
                messages=[
                    {"role": "system", "content": "You are a CS exam question creator for university students."},
                    {"role": "user", "content": fmt},
                ],
                temperature=0.5,
                max_tokens=1200,
            )
            return self._parse_gpt_questions(resp.choices[0].message.content, q_type)
        except Exception as e:
            print(f"  [Quiz] GPT call failed: {e}")
            return []

    def _parse_gpt_questions(self, text: str, q_type: str) -> List[Dict]:
        """Parse GPT output into question dicts."""
        questions = []
        blocks = re.split(r'\n(?=Q:)', text.strip())

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block.startswith('Q:'):
                continue

            lines = block.split('\n')
            q_text = lines[0].replace('Q:', '').strip()
            if not q_text:
                continue

            if q_type == 'mcq':
                options = {}
                correct = None
                for line in lines[1:]:
                    m = re.match(r'^([A-D])\)\s*(.*)', line.strip())
                    if m:
                        options[m.group(1)] = m.group(2).strip()
                    ans = re.match(r'^Answer:\s*([A-D])', line.strip(), re.I)
                    if ans:
                        correct = ans.group(1).upper()

                if len(options) >= 2:
                    opt_list = [f"{k}) {v}" for k, v in sorted(options.items())]
                    correct_text = options.get(correct, list(options.values())[0])
                    questions.append({
                        'question_number': i + 1,
                        'type': 'mcq',
                        'question': q_text,
                        'options': opt_list,
                        'correct_option': correct_text,
                        'correct_letter': correct,
                        'difficulty': 'medium',
                        'topic': '',
                        'source': 'GPT',
                        'marks': 1,
                    })
            else:
                answer_hint = ''
                for line in lines[1:]:
                    ans = re.match(r'^Answer:\s*(.*)', line.strip(), re.I)
                    if ans:
                        answer_hint = ans.group(1).strip()
                        break
                questions.append({
                    'question_number': i + 1,
                    'type': 'short_answer',
                    'question': q_text,
                    'difficulty': 'medium',
                    'topic': '',
                    'source': 'GPT',
                    'marks': 2,
                    'hint': answer_hint,
                    'expected_length': '2-3 sentences',
                })

        return questions

    # ── Fallback question builders ────────────────────────────────────────────

    def _fallback_mcq(self, chunk: Dict, question_num: int) -> Optional[Dict]:
        """Build a meaningful fill-in MCQ from a complete sentence."""
        content = chunk.get('content', '')
        meta = chunk.get('metadata', {})

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content)
                     if len(s.split()) >= 8 and len(s.split()) <= 40]
        if not sentences:
            return None

        sentence = random.choice(sentences[:5])

        # Pick a meaningful noun/term to blank out (>= 5 chars, alpha-only)
        words = [w for w in sentence.split() if w.isalpha() and len(w) >= 5]
        if not words:
            return None

        concept = random.choice(words)
        question_text = sentence.replace(concept, '_____', 1)

        # Build distractors from other sentences in the same chunk
        all_words = [w for w in content.split() if w.isalpha() and len(w) >= 5 and w != concept]
        distractors = list({w for w in random.sample(all_words, min(5, len(all_words)))})[:3]

        options = list(set([concept] + distractors))[:4]
        while len(options) < 4:
            options.append(random.choice(['Process', 'Algorithm', 'Structure', 'Method', 'System']) )
        random.shuffle(options)

        return {
            'question_number': question_num,
            'type': 'mcq',
            'question': f"Fill in the blank: {question_text}",
            'options': options,
            'correct_option': concept,
            'difficulty': meta.get('difficulty', 'medium'),
            'topic': meta.get('topic', 'general').replace('_', ' ').title(),
            'source': meta.get('source_file', 'course material'),
            'marks': 1,
        }

    def _fallback_short_answer(self, chunk: Dict, question_num: int) -> Optional[Dict]:
        """Build a meaningful short-answer question."""
        content = chunk.get('content', '')
        meta = chunk.get('metadata', {})
        topic = meta.get('topic', 'general').replace('_', ' ').title()

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content)
                     if len(s.split()) >= 8]
        if not sentences:
            return None

        base = sentences[0]

        templates = [
            f"Explain: {base[:80]}...",
            f"Based on the material, explain the significance of **{topic}**.",
            f"In 2-3 sentences, describe: {base[:60]}...",
            f"What does the following statement mean? '{base[:70]}...'",
        ]
        question_text = random.choice(templates)

        return {
            'question_number': question_num,
            'type': 'short_answer',
            'question': question_text,
            'difficulty': meta.get('difficulty', 'medium'),
            'topic': topic,
            'source': meta.get('source_file', 'course material'),
            'marks': 2,
            'hint': content[:120] + '...',
            'expected_length': '2-3 sentences',
        }

    # ── Context builder for GPT ───────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict], max_chars: int = 2000) -> str:
        parts = []
        total = 0
        for c in chunks:
            text = c.get('content', '')
            if total + len(text) > max_chars:
                text = text[:max_chars - total]
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return '\n\n'.join(parts)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_mcq_quiz(self, chunks: List[Dict], num_questions: int = 5) -> Dict[str, Any]:
        topic = self._get_topic(chunks)
        context = self._build_context(chunks)

        questions = self._gpt_generate_questions(topic, context, num_questions, 'mcq')

        if len(questions) < num_questions:
            # Fill gaps with fallback
            for chunk in chunks:
                if len(questions) >= num_questions:
                    break
                q = self._fallback_mcq(chunk, len(questions) + 1)
                if q:
                    questions.append(q)

        # Re-number
        for i, q in enumerate(questions):
            q['question_number'] = i + 1
            if not q.get('topic'):
                q['topic'] = chunks[i % len(chunks)].get('metadata', {}).get('topic', 'general').replace('_', ' ').title() if chunks else 'General'

        return {
            'quiz_type': 'mcq',
            'total_questions': len(questions),
            'questions': questions[:num_questions],
            'duration_minutes': len(questions) * 2,
            'generated_at': datetime.now().isoformat(),
            'powered_by': 'GPT' if (self._generator and self._generator._client) else 'Fallback',
        }

    def generate_short_answer_quiz(self, chunks: List[Dict], num_questions: int = 5) -> Dict[str, Any]:
        topic = self._get_topic(chunks)
        context = self._build_context(chunks)

        questions = self._gpt_generate_questions(topic, context, num_questions, 'short_answer')

        if len(questions) < num_questions:
            for chunk in chunks:
                if len(questions) >= num_questions:
                    break
                q = self._fallback_short_answer(chunk, len(questions) + 1)
                if q:
                    questions.append(q)

        for i, q in enumerate(questions):
            q['question_number'] = i + 1

        return {
            'quiz_type': 'short_answer',
            'total_questions': len(questions),
            'questions': questions[:num_questions],
            'duration_minutes': len(questions) * 3,
            'generated_at': datetime.now().isoformat(),
            'powered_by': 'GPT' if (self._generator and self._generator._client) else 'Fallback',
        }

    def generate_mixed_quiz(self, chunks: List[Dict], num_questions: int = 10) -> Dict[str, Any]:
        half = max(1, num_questions // 2)
        mcq_result = self.generate_mcq_quiz(chunks, half)
        sa_result  = self.generate_short_answer_quiz(chunks, num_questions - half)

        all_qs = mcq_result['questions'] + sa_result['questions']
        for i, q in enumerate(all_qs):
            q['question_number'] = i + 1

        return {
            'quiz_type': 'mixed',
            'total_questions': len(all_qs),
            'questions': all_qs,
            'duration_minutes': len(all_qs) * 2.5,
            'difficulty_distribution': self._get_difficulty_distribution(chunks),
            'generated_at': datetime.now().isoformat(),
            'powered_by': 'GPT' if (self._generator and self._generator._client) else 'Fallback',
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_quiz_response(self, quiz: Dict, responses: List[str]) -> Dict[str, Any]:
        score = 0
        total_marks = 0
        results = []

        for i, question in enumerate(quiz.get('questions', [])):
            total_marks += question.get('marks', 1)
            user_ans = responses[i] if i < len(responses) else ''
            correct  = self._check_answer(question, user_ans)
            earned   = question.get('marks', 1) if correct else 0
            score   += earned
            results.append({
                'question_number': question['question_number'],
                'question_type':   question['type'],
                'question':        question['question'],
                'your_answer':     user_ans,
                'correct_answer':  question.get('correct_option', question.get('hint', 'See course material')),
                'is_correct':      correct,
                'marks_obtained':  earned,
                'marks_total':     question.get('marks', 1),
                'topic':           question.get('topic', ''),
            })

        pct = (score / total_marks * 100) if total_marks > 0 else 0
        return {
            'quiz_type':       quiz.get('quiz_type', 'mixed'),
            'total_questions': len(quiz.get('questions', [])),
            'attempted':       len([r for r in results if r['your_answer']]),
            'score':           score,
            'total_marks':     total_marks,
            'percentage':      f"{pct:.1f}%",
            'results':         results,
            'feedback':        self._generate_feedback(pct),
            'evaluation_timestamp': datetime.now().isoformat(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_topic(self, chunks: List[Dict]) -> str:
        if not chunks:
            return 'General'
        topics = [c.get('metadata', {}).get('topic', '') for c in chunks]
        topics = [t for t in topics if t]
        if topics:
            from collections import Counter
            return Counter(topics).most_common(1)[0][0].replace('_', ' ').title()
        subject = chunks[0].get('metadata', {}).get('subject', 'General')
        return subject

    def _get_difficulty_distribution(self, chunks: List[Dict]) -> Dict[str, int]:
        dist = {'easy': 0, 'medium': 0, 'hard': 0}
        for c in chunks:
            d = c.get('metadata', {}).get('difficulty', 'medium')
            if d in dist:
                dist[d] += 1
        return dist

    def _check_answer(self, question: Dict, response: str) -> bool:
        if not response or not response.strip():
            return False
        if question.get('type') == 'mcq':
            correct = question.get('correct_option', '').lower()
            user    = response.lower()
            # Match on letter or full text
            letter  = question.get('correct_letter', '')
            return (correct in user) or (letter and user.startswith(letter.lower())) or (correct == user)
        # Short answer: just check it's a real attempt
        return len(response.strip().split()) >= 5

    def _generate_feedback(self, percentage: float) -> str:
        if percentage >= 85:
            return "🏆 Excellent! You've mastered this topic."
        elif percentage >= 70:
            return "✅ Good job! Review the questions you missed to improve further."
        elif percentage >= 50:
            return "📚 Fair performance. Revisit the weak areas and try again."
        elif percentage >= 30:
            return "⚠️ Needs improvement. Re-read the material and practice more."
        else:
            return "❌ Keep studying! Go through the course material and try again."

    # ── Topic-specific quiz ───────────────────────────────────────────────────

    def generate_topic_specific_quiz(self, chunks: List[Dict], topic: str, num_questions: int = 5) -> Dict[str, Any]:
        topic_chunks = [c for c in chunks
                        if topic.lower() in c.get('metadata', {}).get('topic', '').lower()
                        or topic.lower() in c.get('content', '').lower()[:200]]
        if not topic_chunks:
            return {'error': f'No content found for topic: {topic}'}
        return self.generate_mixed_quiz(topic_chunks, num_questions)


if __name__ == "__main__":
    qg = QuizGenerator()
    mode = "GPT" if (qg._generator and qg._generator._client) else "Fallback"
    print(f"Quiz generator v2 ready — mode: {mode}")