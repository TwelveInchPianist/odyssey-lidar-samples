"""Immutable candidate artifacts and atomic, idle-only model activation.

The trainer may save candidates/reports but MUST NOT activate them. Application
must hold its capture/start lock across activate/rollback and supply can_activate
checking both capture and legacy review idle. Callback checks alone are not a
cross-thread lock. A Session pins load_active() once; there is no hot-swap API.
This registry never changes FCW policy, calibration, confidence or timing gates.
"""
import json
import os
from pathlib import Path
import re
import runtime_clock as time
import uuid
from .radar_forecast import VERSION, OBJECTIVE, validate_model, validate_report, digest

MAX_ARTIFACT_BYTES = 8_000_000
_UNSET = object()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Invalid immutable artifact identity')
    return value


def _read(path):
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError('Registry artifact is too large')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict): raise ValueError('Malformed registry artifact')
    return data


def _atomic(path, value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_ARTIFACT_BYTES: raise ValueError('Registry artifact is too large')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _immutable(path, value):
    if path.exists():
        if _read(path) != value: raise ValueError('Immutable artifact was replaced or corrupted')
        return
    _atomic(path, value)


def _reference(pointer):
    if pointer is None: return None
    return {k: pointer.get(k) for k in ('model_id', 'report_id', 'engine_version',
                                       'runtime_signature', 'evaluation_data_sha256', 'activation_id')}


def _runtime(model, engine_version, signature):
    _id(signature)
    provenance = model.get('provenance', {})
    if provenance.get('engine_version') != engine_version or provenance.get('runtime_signature') != signature:
        raise ValueError('Model engine/runtime code provenance mismatch')


class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root)

    def save_candidate(self, model, report=None):
        """May be called by offline worker. Does not create/change active.json."""
        validate_model(model)
        if report is not None: validate_report(report, model)
        _immutable(self.root/'models'/(_id(model['model_id'])+'.json'), model)
        if report is not None:
            _immutable(self.root/'reports'/(_id(report['report_id'])+'.json'), report)
        return {'model_id': model['model_id'], 'report_id': report['report_id'] if report else None,
                'qualified': bool(report and report['qualified']), 'activated': False}

    def _pointer(self):
        path = self.root/'active.json'
        if not path.exists(): return None
        p = _read(path)
        if p.get('pointer_digest') != digest({k: v for k, v in p.items() if k != 'pointer_digest'}):
            raise ValueError('Active pointer integrity failed')
        if p.get('schema_version') != 1 or p.get('feature_version') != VERSION or p.get('objective') != OBJECTIVE:
            raise ValueError('Incompatible active model pointer')
        return p

    @property
    def active_id(self):
        p = self._pointer()
        return p.get('model_id') if p else None

    def load_candidate(self, model_id, report_id):
        model = _read(self.root/'models'/(_id(model_id)+'.json'))
        report = _read(self.root/'reports'/(_id(report_id)+'.json'))
        comparator_id = report.get('comparison_active_model_id')
        comparator = _read(self.root/'models'/(_id(comparator_id)+'.json')) if comparator_id is not None else None
        validate_report(report, model, comparator)
        if model['model_id'] != model_id or report['report_id'] != report_id:
            raise ValueError('Registry filename/content mismatch')
        return model, report

    def load_active(self, engine_version, *, expected_runtime_signature):
        """Return a verified frozen model, or None for baseline. Errors fail closed.

        Caller catches ValueError/OSError and falls back to the unchanged baseline,
        reporting the error; it must never clear warning safety gates on failure.
        """
        p = self._pointer()
        if p is None or p.get('model_id') is None: return None
        if p.get('engine_version') != engine_version: raise ValueError('Active model engine mismatch')
        if p.get('runtime_signature') != expected_runtime_signature: raise ValueError('Active model runtime code mismatch')
        model, report = self.load_candidate(p['model_id'], p['report_id'])
        _runtime(model, engine_version, expected_runtime_signature)
        if not report['qualified']: raise ValueError('Active model no longer passes declared qualification')
        if p.get('evaluation_data_sha256') != report['evaluation_data_sha256']:
            raise ValueError('Active evaluation identity mismatch')
        return model

    def activate(self, model_id, report_id, *, can_activate, expected_sources,
                 expected_eval_digest, engine_version, expected_runtime_signature, expected_active_id=_UNSET):
        """Automatically activate qualified model; caller MUST hold app lock.

        expected_sources must be {'training': current training SHA256 manifest,
        'evaluation': current evaluation SHA256 manifest}; callers recompute/verify
        these against their immutable catalog, not copy an untrusted report.
        expected_eval_digest identifies the exact evaluated example corpus.
        """
        if not callable(can_activate) or not can_activate(): raise ValueError('Capture/review must be idle')
        if not isinstance(engine_version, str) or not engine_version: raise ValueError('Engine identity required')
        if expected_active_id is _UNSET: raise ValueError('Expected baseline model identity required')
        current = self._pointer(); active_id = current.get('model_id') if current else None
        if active_id != expected_active_id: raise ValueError('Active baseline changed; reevaluate activation')
        model, report = self.load_candidate(model_id, report_id)
        if not report['qualified']: raise ValueError('Candidate unqualified: '+', '.join(report['reasons']))
        if (not isinstance(expected_sources, dict) or set(expected_sources) != {'training', 'evaluation'} or
            expected_sources['training'] != model['training_source_hashes'] or
            expected_sources['evaluation'] != report['evaluation_source_hashes'] or
            expected_eval_digest != report['evaluation_data_sha256']):
            raise ValueError('Source dataset or evaluation identity changed')
        _runtime(model, engine_version, expected_runtime_signature)
        # Do not retain an unverified/tampered pointer as a rollback target.
        if current and active_id is not None:
            self.load_active(engine_version, expected_runtime_signature=expected_runtime_signature)
        if not can_activate(): raise ValueError('Capture/review became active')
        if self.active_id != expected_active_id: raise ValueError('Concurrent active pointer change')
        if active_id == model_id and current.get('report_id') == report_id:
            return {'activated': False, 'already_active': True, 'model_id': model_id}
        if report.get('comparison_active_model_id') != active_id:
            raise ValueError('Candidate was not evaluated against the current active model')
        pointer = {'schema_version': 1, 'feature_version': VERSION, 'objective': OBJECTIVE,
                   'model_id': model_id, 'report_id': report_id, 'engine_version': engine_version,
                   'runtime_signature': expected_runtime_signature,
                   'evaluation_data_sha256': report['evaluation_data_sha256'],
                   'previous': _reference(current), 'activation_id': uuid.uuid4().hex,
                   'activated_at_unix_s': time.time(), 'action': 'automatic_qualified_activation'}
        pointer['pointer_digest'] = digest(pointer)
        # Journal first; a crash before pointer replacement leaves an unused event,
        # not a partially activated model. Original model/report versions persist.
        _immutable(self.root/'activations'/(pointer['activation_id']+'.json'), pointer)
        if not can_activate(): raise ValueError('Capture/review became active before commit')
        _atomic(self.root/'active.json', pointer)
        return {'activated': True, 'model_id': model_id, 'report_id': report_id,
                'activation_id': pointer['activation_id'], 'previous_model_id': active_id}

    def rollback(self, *, can_activate, engine_version, expected_runtime_signature, expected_active_id=_UNSET):
        """Automatic known-good rollback, serialized by the same application lock."""
        if not callable(can_activate) or not can_activate(): raise ValueError('Capture/review must be idle')
        _id(expected_runtime_signature)
        if expected_active_id is _UNSET or self.active_id != expected_active_id:
            raise ValueError('Expected active model changed')
        current = self._pointer()
        if current is None: return {'rolled_back': False, 'model_id': None}
        previous = current.get('previous')
        prior_event = None
        if previous:
            event_id = previous.get('activation_id')
            if not isinstance(event_id, str) or not re.fullmatch(r'[0-9a-f]{32}', event_id):
                raise ValueError('Invalid rollback journal identity')
            prior_event = _read(self.root/'activations'/(event_id+'.json'))
            if (prior_event.get('pointer_digest') != digest({k: v for k, v in prior_event.items() if k != 'pointer_digest'}) or
                _reference(prior_event) != previous): raise ValueError('Rollback journal integrity failed')
        if previous and previous.get('model_id') is not None:
            if previous.get('engine_version') != engine_version: raise ValueError('Rollback engine mismatch')
            model, report = self.load_candidate(previous['model_id'], previous['report_id'])
            _runtime(model, engine_version, expected_runtime_signature)
            if not report['qualified']: raise ValueError('Rollback target is not qualified')
        pointer = {'schema_version': 1, 'feature_version': VERSION, 'objective': OBJECTIVE,
                   'model_id': previous.get('model_id') if previous else None,
                   'report_id': previous.get('report_id') if previous else None,
                   'engine_version': engine_version,
                   'runtime_signature': expected_runtime_signature,
                   'evaluation_data_sha256': previous.get('evaluation_data_sha256') if previous else None,
                   'previous': prior_event.get('previous') if prior_event else None,
                   'activation_id': uuid.uuid4().hex, 'activated_at_unix_s': time.time(),
                   'action': 'automatic_rollback', 'reverted_model_id': current.get('model_id')}
        pointer['pointer_digest'] = digest(pointer)
        _immutable(self.root/'activations'/(pointer['activation_id']+'.json'), pointer)
        if not can_activate() or self.active_id != expected_active_id:
            raise ValueError('Capture/review or baseline changed before rollback')
        _atomic(self.root/'active.json', pointer)
        return {'rolled_back': True, 'model_id': pointer['model_id'], 'activation_id': pointer['activation_id']}
