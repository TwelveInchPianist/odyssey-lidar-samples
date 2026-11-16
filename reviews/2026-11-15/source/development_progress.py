"""Read-only, reviewed development checklist; never an FCW readiness input."""
from datetime import date
import json
from pathlib import Path
import re

PRESENTATION_REVISION = 'night-dogfood-1'
STATUSES = {'complete', 'blocked', 'remaining'}


def load_development_progress(path):
    """Compute earned checklist points once. Missing/invalid evidence shows no score."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 256_000:
            raise ValueError('Checklist exceeds its size limit')
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
            raise ValueError('Unsupported checklist schema')
        date.fromisoformat(data['reviewed_on'])
        if not isinstance(data.get('milestones'), list) or not data['milestones']:
            raise ValueError('Milestones are missing')
        evidence = data['evidence']
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError('Evidence index is missing')
        for item in evidence.values():
            if not isinstance(item, dict) or not isinstance(item.get('label'), str):
                raise ValueError('Invalid evidence label')
            if not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:txt|json)', item.get('file', '')):
                raise ValueError('Invalid evidence filename')
        ids = set()
        milestone_ids = set()
        earned = total = completed = count = 0
        blockers = []
        for milestone in data['milestones']:
            if not isinstance(milestone, dict) or not isinstance(milestone.get('title'), str):
                raise ValueError('Invalid milestone')
            group_id=milestone['id']
            if not isinstance(group_id,str) or not re.fullmatch(r'[a-z0-9_-]+',group_id) or group_id in milestone_ids:
                raise ValueError('Invalid or duplicate milestone ID')
            milestone_ids.add(group_id)
            criteria = milestone['criteria']
            if not isinstance(criteria, list) or not criteria:
                raise ValueError('Milestone has no criteria')
            group_earned = group_total = group_complete = 0
            for criterion in criteria:
                key = criterion['id']
                if not isinstance(key, str) or not re.fullmatch(r'[a-z0-9_-]+', key) or key in ids:
                    raise ValueError('Invalid or duplicate criterion ID')
                ids.add(key)
                weight = criterion['weight']
                if type(weight) is not int or not 1 <= weight <= 100:
                    raise ValueError('Invalid criterion weight')
                if criterion['status'] not in STATUSES:
                    raise ValueError('Unknown criterion status')
                for field in ('title', 'completion_rule'):
                    if not isinstance(criterion.get(field), str) or not criterion[field].strip():
                        raise ValueError('Criterion explanation is missing')
                refs = criterion.get('evidence_ids', [])
                if not isinstance(refs, list) or any(ref not in evidence for ref in refs):
                    raise ValueError('Unknown evidence reference')
                if criterion['status'] == 'complete':
                    if not refs:
                        raise ValueError('Completed criterion needs reviewed evidence')
                    group_earned += weight
                    group_complete += 1
                else:
                    if not isinstance(criterion.get('blocker'), str) or not criterion['blocker'].strip():
                        raise ValueError('Remaining criterion needs a blocker or next step')
                    blockers.append({'id':key, 'title':criterion.get('blocker_title',criterion['title']), 'text':criterion['blocker'],
                                     'priority':criterion.get('priority', 10), 'milestone':milestone['title']})
                group_total += weight
                count += 1
            milestone.update(earned_points=group_earned, total_points=group_total,
                             completed_criteria=group_complete, criterion_count=len(criteria),
                             status='complete' if group_complete == len(criteria) else 'in_progress')
            earned += group_earned
            total += group_total
            completed += group_complete
        if total != 100:
            raise ValueError('Reviewed milestone weights must total 100')
        blockers.sort(key=lambda item: (item['priority'], item['id']))
        data.update(available=True, percent=earned, earned_points=earned, total_points=total,
                    completed_criteria=completed, criterion_count=count, blockers=blockers,
                    completed_milestones=sum(m['status']=='complete' for m in data['milestones']),
                    presentation_revision=PRESENTATION_REVISION,
                    meaning='Weighted development checklist completion; not warning readiness, detection probability, or reliability.')
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {'available':False, 'percent':None, 'evidence':{},
                'presentation_revision':PRESENTATION_REVISION,
                'message':'Reviewed development checklist unavailable. Live warning checks are unaffected.'}
