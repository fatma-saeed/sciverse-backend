"""Render scientific claims from checked tool evidence, never provider prose."""
import re

def finalize(result, question):
    arabic = bool(re.search(r'[\u0600-\u06ff]', question))
    picks = result.get('picks', [])
    if result.get('status') != 'complete':
        result['answer'] = 'لم يكتمل البحث؛ لا توجد توصية نهائية.' if arabic else 'Search incomplete; no final recommendation.'
    elif not picks:
        result['answer'] = 'لم يجد البحث مرشحين ضمن القيود المطبقة.' if arabic else 'No candidates found under the applied search constraints.'
    else:
        lines = []
        for rank, p in enumerate(picks, 1):
            f, c = p['facts'], p['check']
            p['sourceRank'] = p['rank']
            p['rank'] = rank
            p['scoreInterval'] = None
            p['confidence'] = None
            p['scoreDefinitions'] = {
                'score_pct': 'target-and-purpose dataset percentile',
                'check.similarity_pct': 'body-level similarity dataset percentile',
                'check.classifier_pct': 'body-level classifier dataset percentile',
                'check.final_pct': 'body-level combined dataset percentile'}
            if arabic:
                lines += [f"{rank}. {p['name']}: مرشح للدراسة مقارنةً بـ{f['target']['name']} لغرض {f['purpose']}.",
                    f"الترتيب المئيني للهدف والغرض داخل مجموعة البيانات: {p['score_pct']}؛ المقارنة العامة مع الكوكب: {c['similarity_pct']}؛ المصنّف: {c['classifier_pct']}. هذه مقاييس مختلفة وليست نسب تشابه مادي أو احتمالات نجاح."]
            else:
                lines += [f"{rank}. {p['name']}: candidate for investigation alongside {f['target']['name']} for {f['purpose']}.",
                    f"Target-and-purpose dataset percentile: {p['score_pct']}. Body-level similarity percentile: {c['similarity_pct']}; classifier percentile: {c['classifier_pct']}. These are distinct rankings, not physical similarity percentages or success probabilities."]
            for label, key in [('Smaller measured feature gaps', 'closest_features'), ('Larger measured feature gaps', 'widest_gaps')]:
                lines.append(label + ': ' + '; '.join(f"{v['label']}: Earth {v['earth']} vs target {v['target']} {v['unit']}" for v in f[key]))
            lines.append('Audit: ' + '; '.join(p['audit']['flags'] or ['No flags recorded; this does not establish site verification.']))
            lines.append('Physical limits: ' + '; '.join(f['cannot_be_reproduced'].values()))
        lines.append('Exploratory candidates only. Coarse spatial sampling, unverified coordinates and missing measurements can affect rankings. Field verification is required; ice resources, biology, subsurface conditions and mission suitability are not established.')
        result['answer'] = '\n\n'.join(lines)
    result['answerMethod'] = 'deterministic rendering of checked tool evidence; LLM selects tools'
    result['limitations'] = 'Dataset percentiles only; not mission approval or confirmed novelty. No web search. User intent interpretation by the LLM still requires review.'
    return result
