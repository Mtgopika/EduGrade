
import numpy as np
import torch
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from rapidfuzz import fuzz


def preprocess_text(text):
    return text.lower().strip()

def tfidf_similarity(student_ans, answer_key):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_ans] + [answer_key])
    similarity_score = np.dot(vectors[0], vectors[1:].T).toarray().flatten()[0]
    return similarity_score * 100

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze()

def bert_similarity(student_ans, answer_key):
    student_vector = bert_embedding(student_ans)
    key_vector = bert_embedding(answer_key)
    similarity = torch.nn.functional.cosine_similarity(student_vector, key_vector, dim=0)
    return similarity.item() * 100

def levenshtein_similarity(student_ans, answer_key):
    return fuzz.ratio(student_ans, answer_key)


# ----------- QUESTION EXTRACTION -----------

import re
import json

def extract_main_question(line):
    pattern = re.compile(r"^\(?([IVXLCDM]+)[\.\)]\)?\s*(.*?)(?:\s*\(Total:\s*(\d+(\.\d+)?)\s*marks\))?$", re.DOTALL)
    match = re.match(pattern, line)
    if match:
        print(f"Detected Main Question: {match.groups()}")
        return match.group(1).strip(), match.group(2).strip() if match.group(2) else "", float(match.group(3)) if match.group(3) else None
    return None, None, None

def extract_sub_question(line):
    pattern = re.compile(r"^\(?([a-z])[)\.]\)?\s*(.*?)\s*\(Total:\s*(\d+(\.\d+)?)\s*marks\)", re.DOTALL)
    match = re.match(pattern, line)
    if match:
        print(f"Detected Sub Question: {match.groups()}")
        return match.group(1).strip(), match.group(2).strip(), float(match.group(3))
    return None, None, None

def extract_sub_point(line):
    # This version matches content from "-" up to the (Total: x marks), even if it spans lines
    pattern = re.compile(r"^[\-\•]\s*((?:.|\n)*?)\s*\(Total\s*:\s*(\d+(?:\.\d+)?)\s*marks\)", re.DOTALL)
    match = re.match(pattern, line)
    if match:
        print(f"Detected Sub Point: {match.groups()}")
        return match.group(1).strip(), float(match.group(2))  # Return text and marks
    return None, None


def extract_answers_from_answer_key(answer_key_text):
    answers = {}
    last_main_question, last_sub_question = None, None

    buffer = ""
    inside_sub_point = False

    for line in answer_key_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        print(f"Processing Line: {line}")

        # If inside a multi-line sub-point, keep buffering
        if inside_sub_point:
            buffer += " " + line
            if re.search(r"\(Total\s*:\s*\d+(\.\d+)?\s*marks\)", buffer):
                sub_point_text, sub_point_marks = extract_sub_point(buffer.strip())
                if sub_point_text:
                    if last_sub_question:
                        answers[last_main_question]["sub_questions"][last_sub_question]["sub_points"][sub_point_text] = {
                            "answer": sub_point_text,
                            "marks": sub_point_marks
                        }
                    else:
                        answers[last_main_question]["sub_points"][sub_point_text] = {
                            "answer": sub_point_text,
                            "marks": sub_point_marks
                        }
                buffer = ""
                inside_sub_point = False
            continue

        # Start of a sub-point
        if re.match(r"^[\-\•]\s*", line):
            buffer = line
            inside_sub_point = True
            if re.search(r"\(Total\s*:\s*\d+(\.\d+)?\s*marks\)", line):
                sub_point_text, sub_point_marks = extract_sub_point(line)
                if sub_point_text:
                    if last_sub_question:
                        answers[last_main_question]["sub_questions"][last_sub_question]["sub_points"][sub_point_text] = {
                            "answer": sub_point_text,
                            "marks": sub_point_marks
                        }
                    else:
                        answers[last_main_question]["sub_points"][sub_point_text] = {
                            "answer": sub_point_text,
                            "marks": sub_point_marks
                        }
                buffer = ""
                inside_sub_point = False
            continue

        # Main question detection
        question_num, answer_text, marks = extract_main_question(line)
        if question_num:
            answers[question_num] = {
                "answer": answer_text,
                "marks": marks,
                "sub_questions": {},
                "sub_points": {}
            }
            last_main_question = question_num
            last_sub_question = None
            continue

        # Sub-question detection
        sub_question_num, sub_answer_text, sub_marks = extract_sub_question(line)
        if sub_question_num and last_main_question:
            answers[last_main_question]["sub_questions"][sub_question_num] = {
                "answer": sub_answer_text,
                "marks": sub_marks,
                "sub_points": {}
            }
            last_sub_question = sub_question_num
            continue

        # Add extra content to main or sub-question if no other match
        if last_main_question:
            if last_sub_question:
                answers[last_main_question]["sub_questions"][last_sub_question]["answer"] += " " + line
            else:
                answers[last_main_question]["answer"] += " " + line

    # Final cleanup
    for q in answers.values():
        q["answer"] = q["answer"].strip()
        for sub_q in q["sub_questions"].values():
            sub_q["answer"] = sub_q["answer"].strip()
            for sub_point in sub_q["sub_points"].values():
                sub_point["answer"] = sub_point["answer"].strip()

    return answers


def extract_answers_from_sheet(answer_sheet_text):

    student_answers = {}
    main_question_pattern = re.compile(r"^\(?([IVXLCDM]+)[\.\)]\)?")
    sub_question_pattern = re.compile(r"^\(?([a-z])[)\.]", re.IGNORECASE)



    last_main_question = None
    last_sub_question = None

    lines = answer_sheet_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if re.match(r"^Part [A-B]$", line, re.IGNORECASE):
            i += 1
            continue

        main_match = re.match(main_question_pattern, line)
        if main_match:
            last_main_question = main_match.group(1).strip()
            if last_main_question not in student_answers:
                student_answers[last_main_question] = {
                    "sub_questions": {},
                    "answer": ""
                }
            last_sub_question = None
            question_text = line[len(main_match.group(0)):].strip()
            if question_text:
                student_answers[last_main_question]["answer"] += question_text + " "
            i += 1
            continue

        sub_match = re.match(sub_question_pattern, line)
        if sub_match and last_main_question:
            last_sub_question = sub_match.group(1).strip()
            if last_sub_question not in student_answers[last_main_question]["sub_questions"]:
                student_answers[last_main_question]["sub_questions"][last_sub_question] = {
                    "answer": ""
                }
            sub_question_text = line[len(sub_match.group(0)):].strip()
            if sub_question_text:
                student_answers[last_main_question]["sub_questions"][last_sub_question]["answer"] += sub_question_text + " "
            i += 1
            continue

        if last_main_question:
            if last_sub_question:
                student_answers[last_main_question]["sub_questions"][last_sub_question]["answer"] += line + " "
            else:
                student_answers[last_main_question]["answer"] += line + " "

        i += 1

    for q in student_answers.values():
        q["answer"] = q["answer"].strip()
        for sub_q in q["sub_questions"].values():
            sub_q["answer"] = sub_q["answer"].strip()

    return student_answers




# ----------- MATCHING & SCORING -----------

def match_questions(answer_key_text, student_answers_text):


    # Step 1: Extract structured data from text
    answer_key = extract_answers_from_answer_key(answer_key_text)
    student_answers = extract_answers_from_sheet(student_answers_text)

    matched_answers = {}

    # Step 2: Match answers based on question numbers
    for q_num, key_data in answer_key.items():
        if q_num in student_answers:

            combined_answer_key = key_data["answer"]

            for sub_point, sub_data in key_data.get("sub_points", {}).items():
                combined_answer_key += f"\n{sub_point}: {sub_data['answer']}"

            matched_answers[q_num] = {
                "answer_key": combined_answer_key,
                "student_answer": student_answers.get(q_num, {}).get("answer", ""),
                "marks": key_data["marks"],
                "sub_points": key_data.get("sub_points", {})
            }

            # Match sub-questions (a, b, c, d)
            for sub_q_num, sub_q_data in key_data.get("sub_questions", {}).items():
                sub_combined_answer_key = sub_q_data["answer"]

                for sub_point, sub_data in sub_q_data.get("sub_points", {}).items():
                    sub_combined_answer_key += f"\n{sub_point}: {sub_data['answer']}"

                matched_answers[q_num]["sub_questions"] = matched_answers[q_num].get("sub_questions", {})
                matched_answers[q_num]["sub_questions"][sub_q_num] = {
                    "answer_key": sub_combined_answer_key,
                    "student_answer": student_answers.get(q_num, {}).get("sub_questions", {}).get(sub_q_num, {}).get("answer", ""),
                    "marks": sub_q_data["marks"],
                    "sub_points": sub_q_data.get("sub_points", {})
                }

    return matched_answers


from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from rake_nltk import Rake
import networkx as nx
import nltk

def normalize_dict(d):
    max_val = max(d.values()) if d else 1
    return {k: v / max_val for k, v in d.items()}
import re

def get_tfidf_scores(text):
    text = text.strip()
    if not text or len(text.split()) < 2:
        return {}

    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b\w\w+\b")
    try:
        tfidf.fit([text])
        scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        return normalize_dict(scores)
    except ValueError:
        return {}  # Handles case when vocabulary is empty


def get_rake_scores(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked = rake.get_ranked_phrases_with_scores()
    return normalize_dict({phrase: score for score, phrase in ranked})

def get_yake_scores(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return normalize_dict({kw: 1/score for kw, score in keywords})  # Inverse to normalize

def get_pos_scores(text):
    words = nltk.word_tokenize(text.lower())
    tags = nltk.pos_tag(words)
    keywords = [w for w, t in tags if t in ('NN', 'NNS', 'JJ')]
    return {kw: 1.0 for kw in keywords}

def ensemble_keyword_weights(text):
    tfidf_scores = get_tfidf_scores(text)
    rake_scores = get_rake_scores(text)
    yake_scores = get_yake_scores(text)
    pos_scores = get_pos_scores(text)

    # Union of all keywords
    all_keywords = set(tfidf_scores) | set(rake_scores) | set(yake_scores) | set(pos_scores)

    keyword_weights = {}
    for kw in all_keywords:
        weights = [
            tfidf_scores.get(kw, 0),
            rake_scores.get(kw, 0),
            yake_scores.get(kw, 0),
            pos_scores.get(kw, 0)
        ]
        keyword_weights[kw] = round(sum(weights) / len(weights), 3)  # simple average

    return keyword_weights

import numpy as np
import re
import json
import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rapidfuzz import fuzz

# Load NLP models
nlp = spacy.load("en_core_web_sm")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

# Keyword extraction functions
def extract_keywords_tfidf(text):
    # Added a check for empty text
    if not text or text.isspace():
        return []  # Return empty list if text is empty or contains only whitespace

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_tfidf_scores if score > 0]
    return keywords

def extract_keywords_ner(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents]
    keywords += [chunk.text for chunk in doc.noun_chunks]
    return keywords

def extract_all_keywords_with_weights(text):
    tfidf_scores = get_tfidf_scores(text)
    rake_scores = get_rake_scores(text)
    yake_scores = get_yake_scores(text)
    pos_scores = get_pos_scores(text)

    all_keywords = set(tfidf_scores) | set(rake_scores) | set(yake_scores) | set(pos_scores)

    keyword_weights = {}
    for kw in all_keywords:
        weights = [
            tfidf_scores.get(kw, 0),
            rake_scores.get(kw, 0),
            yake_scores.get(kw, 0),
            pos_scores.get(kw, 0)
        ]
        keyword_weights[kw] = round(sum(weights) / len(weights), 3)

    total = sum(keyword_weights.values()) or 1
    keyword_weights = {k: v / total for k, v in keyword_weights.items()}  # Normalize to sum to 1
    return keyword_weights




def get_semantic_similarity(text1, text2):
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2)[0][0].item()

def get_nli_relationship(premise, hypothesis):
    result = nli(f"{premise} </s> {hypothesis}")[0]
    return result['label'].lower(), result['score']

# Core grading logic





from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')


def split_into_sentences(text):
    return sent_tokenize(text.strip())

def grade_answer(answer_key_text, student_answer_text, max_marks, is_sub_point=False):
    # Extract keyword weights
    key_weights = extract_all_keywords_with_weights(answer_key_text)
    student_keywords = extract_all_keywords_with_weights(student_answer_text)

    matched_keywords = set(key_weights).intersection(set(student_keywords))
    missing_keywords = set(key_weights) - matched_keywords

    # Rank keywords by weight
    sorted_keywords = sorted(key_weights.items(), key=lambda x: x[1], reverse=True)
    top_keywords = {k for k, _ in sorted_keywords[:max(1, len(sorted_keywords) // 3)]}

    # Compute penalty
    penalty = 0
    for kw in missing_keywords:
        if kw in top_keywords:
            penalty += key_weights[kw] * 0.3  # Reduced penalty for out-of-order keywords
        else:
            penalty += key_weights[kw] * 0.1
  # Light penalty for minor ones

    matched_weight = sum(key_weights.get(k, 0) for k in matched_keywords)
    keyword_score = max(matched_weight - penalty, 0)  # Ensure score is not negative

    # Semantic similarity
    semantic_score = get_semantic_similarity(answer_key_text, student_answer_text)

    # Global contradiction check
    global_nli_result, global_nli_conf = get_nli_relationship(answer_key_text, student_answer_text)

    # Sentence-level contradiction check
    key_sentences = split_into_sentences(answer_key_text)
    student_sentences = split_into_sentences(student_answer_text)
    contradiction_detected = False
    contradiction_details = []

    for key_sentence in key_sentences:
        best_match = None
        best_score = 0.0

        for student_sentence in student_sentences:
            sim_score = get_semantic_similarity(key_sentence, student_sentence)
            if sim_score > best_score:
                best_score = sim_score
                best_match = student_sentence

        if best_score > 0.4 and best_match:
            nli_result, nli_confidence = get_nli_relationship(key_sentence, best_match)
            if nli_result == "contradiction" and nli_confidence > 0.7:
                contradiction_detected = True
                contradiction_details.append((key_sentence, best_match, nli_confidence))
                break

    # Levenshtein similarity score
    lev_score = fuzz.ratio(answer_key_text.lower(), student_answer_text.lower())

    # Final score computation
    if contradiction_detected:
        if is_sub_point:
            total_score = 0
            reason = f"Contradiction in sub-point. Confidence: {contradiction_details[0][2]:.2f}"
        else:
            total_score = max(0, round(semantic_score * max_marks * 0.5, 2))
            reason = f"Contradiction in sentence detected. Confidence: {contradiction_details[0][2]:.2f}"
    else:
        # Base score: average of semantic and keyword
        keyword_marks = keyword_score * max_marks
        total_score = round((0.4 * keyword_marks + 0.6 * semantic_score * max_marks), 2)
        reason = "Scored based on keyword match and semantic similarity"

        # ✅ Override for near-exact match (Levenshtein)
        if lev_score > 90:
            total_score = max_marks
            reason = f"Full marks: near-exact match (Levenshtein {lev_score}%)"

    return total_score, semantic_score, global_nli_result, global_nli_conf, list(matched_keywords), reason

def calculate_scores(matched_answers):
    total_obtained_marks = 0
    total_marks = 0

    score_details = []
    or_groups = [("II", "III"), ("IV", "V"), ("VI", "VII"), ("VIII", "IX")]
    attempted_or_questions = {}

    for question_num, data in matched_answers.items():
        student_ans = data.get("student_answer", "").strip()
        correct_ans = data.get("answer_key", "").strip()
        max_marks = data.get("marks", 0) or 0

        for group in or_groups:
            if question_num in group:
                if group not in attempted_or_questions and student_ans:
                    attempted_or_questions[group] = question_num

        if any(question_num in group and attempted_or_questions[group] != question_num for group in or_groups):
            continue

        obtained_marks = 0
        sub_point_total = sum(sp["marks"] for sp in data.get("sub_points", {}).values())
        adjusted_main_marks = max_marks - sub_point_total
        total_marks += max_marks

        question_result = {
            "Question Number": question_num,
            "Max Marks": max_marks,
            "Marks Obtained": 0,
            "Semantic Similarity": 0,
            "NLI": "",
            "NLI Confidence": 0,
            "Matched Keywords": [],
            "Reason": "",
            "sub_questions": {},
            "sub_points": {}
        }

        # ✅ Handle direct sub-points
        if data.get("sub_points"):
            for sub_point_text, sp_data in data["sub_points"].items():
                sp_correct_ans = sp_data["answer"]
                sp_marks = sp_data["marks"]

                sentences = sent_tokenize(student_ans)
                best_score = 0
                best_sentence = student_ans

                for sent in sentences:
                    sim_score = get_semantic_similarity(sp_correct_ans, sent)
                    if sim_score > best_score:
                        best_score = sim_score
                        best_sentence = sent

                sp_student_ans = best_sentence

                # Grade
                sp_score, sp_sem_score, sp_nli, sp_nli_conf, sp_matched, sp_reason = grade_answer(
                    sp_correct_ans, sp_student_ans, sp_marks, is_sub_point=True
                )

                # ✅ Override if near-exact match
                if sp_correct_ans.lower() in sp_student_ans.lower():
                    sp_score = sp_marks

                total_obtained_marks += sp_score
                obtained_marks += sp_score

                question_result["sub_points"][sub_point_text] = {
                    "Marks Obtained": sp_score,
                    "Max Marks": sp_marks,
                    "Semantic Similarity": round(sp_sem_score, 2),
                    "NLI": sp_nli,
                    "NLI Confidence": round(sp_nli_conf, 2),
                    "Matched Keywords": sp_matched,
                    "Reason": sp_reason
                }

        else:
            obtained_marks, semantic_score, nli_result, nli_conf, matched_keywords, reason = grade_answer(
                correct_ans, student_ans, max_marks
            )
            total_obtained_marks += obtained_marks
            question_result.update({
                "Marks Obtained": obtained_marks,
                "Semantic Similarity": round(semantic_score, 2),
                "NLI": nli_result,
                "NLI Confidence": round(nli_conf, 2),
                "Matched Keywords": matched_keywords,
                "Reason": reason
            })

        # ✅ Handle sub-questions
        for sub_q_num, sub_data in data.get("sub_questions", {}).items():
            sub_student_ans = sub_data.get("student_answer", "").strip()
            sub_correct_ans = sub_data.get("answer_key", "").strip()
            sub_marks = sub_data.get("marks", 0) or 0

            sub_score = 0
            sub_result = {
                "Marks Obtained": 0,
                "Max Marks": sub_marks,
                "Semantic Similarity": 0,
                "NLI": "",
                "NLI Confidence": 0,
                "Matched Keywords": [],
                "Reason": "",
                "sub_points": {}
            }

            # ✅ Handle sub-points inside sub-questions
            if sub_data.get("sub_points"):
                for sub_point_text, sp_data in sub_data["sub_points"].items():
                    sp_correct_ans = sp_data["answer"]
                    sp_marks = sp_data["marks"]

                    sentences = sent_tokenize(sub_student_ans)
                    best_score = 0
                    best_sentence = sub_student_ans

                    for sent in sentences:
                        sim_score = get_semantic_similarity(sp_correct_ans, sent)
                        if sim_score > best_score:
                            best_score = sim_score
                            best_sentence = sent

                    sp_student_ans = best_sentence

                    # Grade
                    # Grade
                    sp_score, sp_sem_score, sp_nli, sp_nli_conf, sp_matched, sp_reason = grade_answer(
                        sp_correct_ans, sp_student_ans, sp_marks, is_sub_point=True
                    )

                    # ✅ Exact match override only if there is no contradiction
                    if sp_correct_ans.lower() in sp_student_ans.lower() and sp_nli != "contradiction":
                        sp_score = sp_marks


                    total_obtained_marks += sp_score
                    sub_score += sp_score

                    sub_result["sub_points"][sub_point_text] = {
                        "Marks Obtained": sp_score,
                        "Max Marks": sp_marks,
                        "Semantic Similarity": round(sp_sem_score, 2),
                        "NLI": sp_nli,
                        "NLI Confidence": round(sp_nli_conf, 2),
                        "Matched Keywords": sp_matched,
                        "Reason": sp_reason
                    }

                sub_result["Marks Obtained"] = sub_score
                sub_result["Reason"] = "Scored based on sub-points only"
            else:
                sub_score, sub_sem_score, sub_nli, sub_nli_conf, sub_matched, sub_reason = grade_answer(
                    sub_correct_ans, sub_student_ans, sub_marks
                )
                total_obtained_marks += sub_score
                sub_result.update({
                    "Marks Obtained": sub_score,
                    "Semantic Similarity": round(sub_sem_score, 2),
                    "NLI": sub_nli,
                    "NLI Confidence": round(sub_nli_conf, 2),
                    "Matched Keywords": sub_matched,
                    "Reason": sub_reason
                })

            obtained_marks += sub_score
            question_result["sub_questions"][sub_q_num] = sub_result

        question_result["Marks Obtained"] = obtained_marks
        score_details.append(question_result)

    final_percentage = (total_obtained_marks / total_marks) * 100 if total_marks > 0 else 0
    return score_details, final_percentage
