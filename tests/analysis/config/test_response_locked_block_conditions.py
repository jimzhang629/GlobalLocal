"""The response-locked block-balanced sets must mirror the stimulus-locked ones.

They exist to rerun the four block-balanced decodes time-locked to the response,
to check whether the stimulus-locked block effects are reaction-time differences.
That comparison only means something if the two versions select the same
trials and split them the same way, differing in nothing but the event the
epochs are locked to.
"""

import pandas as pd
import pytest

from src.analysis.config import experiment_conditions as ec
from src.analysis.config.condition_registry import CONDITION_REGISTRY, get_conditions_obj
from src.analysis.utils.epoch_metadata_utils import parse_event_name


ANALYSES = ['lwpc', 'lwps', 'congruency_by_switch_prop', 'switch_type_by_inc_prop']
PAIRS = [pytest.param(f'stimulus_{a}_block_balanced_conditions',
                      f'response_{a}_block_balanced_conditions', id=a)
         for a in ANALYSES]
CELL_SETS = ['congruency_by_block_conditions', 'switch_type_by_block_conditions']


def _to_response(name):
    return name.replace('Stimulus_', 'Response_', 1)


def _renamed(groups):
    return [[_to_response(n) for n in group] for group in groups]


def _named_conditions(entry):
    names = [n for groups in entry['comparisons'].values() for g in groups for n in g]
    names += [n for s in entry['pooled_shuffle'] for g in s['strings_to_find'] for n in g]
    return names


@pytest.mark.parametrize('stim_label,resp_label', PAIRS)
def test_response_entry_mirrors_its_stimulus_entry(stim_label, resp_label):
    stim, resp = CONDITION_REGISTRY[stim_label], CONDITION_REGISTRY[resp_label]

    assert resp['comparisons'] == {k: _renamed(g) for k, g in stim['comparisons'].items()}
    assert ([s['strings_to_find'] for s in resp['pooled_shuffle']]
            == [_renamed(s['strings_to_find']) for s in stim['pooled_shuffle']])
    for key in ('balance_strata', 'anova_factors', 'anova_interactions'):
        assert resp[key] == stim[key], key

    stim_ctx, resp_ctx = stim['context_comparison'], resp['context_comparison']
    renamed = {'condition_name', 'display_name'}
    assert {k: v for k, v in resp_ctx.items() if k not in renamed} == \
        {k: v for k, v in stim_ctx.items() if k not in renamed}
    assert resp_ctx['condition_name'] != stim_ctx['condition_name']
    assert resp_ctx['display_name'] != stim_ctx['display_name']


@pytest.mark.parametrize('stim_label,resp_label', PAIRS)
def test_every_condition_a_response_entry_names_exists(stim_label, resp_label):
    conditions = get_conditions_obj(resp_label)
    missing = [n for n in _named_conditions(CONDITION_REGISTRY[resp_label])
               if n not in conditions]
    assert not missing, f'{resp_label} names conditions it does not define: {missing}'


@pytest.mark.parametrize('stim_label,resp_label', PAIRS)
def test_building_the_response_entries_leaves_the_stimulus_ones_alone(stim_label, resp_label):
    names = _named_conditions(CONDITION_REGISTRY[stim_label])
    assert names and all(n.startswith('Stimulus_') for n in names), names
    assert all(n.startswith('Stimulus_') for n in get_conditions_obj(stim_label))


@pytest.mark.parametrize('cell_set', CELL_SETS)
def test_response_cells_copy_the_stimulus_cells(cell_set):
    stim = getattr(ec, f'stimulus_{cell_set}')
    resp = getattr(ec, f'response_{cell_set}')
    assert set(resp) == {_to_response(n) for n in stim}
    for name, spec in stim.items():
        r = resp[_to_response(name)]
        assert r['BIDS_events'] == [e.replace('Stimulus/', 'Response/', 1)
                                    for e in spec['BIDS_events']]
        assert {k: v for k, v in r.items() if k != 'BIDS_events'} == \
            {k: v for k, v in spec.items() if k != 'BIDS_events'}


def _response_event_names():
    """One accurate response event per congruency x switch type x block, in the
    tag layout the response-locked epochs use (as `response_experiment_conditions`
    already relies on)."""
    return [f'Response/{c}{inc}.0/{t}{sw}.0/BigLetters/SmallLetterh/Taskg/'
            f'TargetLetters/Responded1.0/TrialCount{i}.0/ReactionTime900.0/Accuracy1.0/D57'
            for i, (c, inc, t, sw) in enumerate(
                (c, inc, t, sw) for c in 'ci' for inc in (25, 75)
                for t in 'rs' for sw in (25, 75))]


@pytest.mark.parametrize('cell_set', CELL_SETS)
def test_metadata_queries_select_the_right_response_events(cell_set):
    """The loader applies each cell's metadata_query to metadata parsed from the
    epochs' event names; on Response/ events it must pick exactly the events its
    BIDS_events name."""
    events = _response_event_names()
    metadata = pd.DataFrame([parse_event_name(e) for e in events])
    for name, spec in getattr(ec, f'response_{cell_set}').items():
        selected = set(metadata.loc[metadata.eval(spec['metadata_query']), 'full_event_name'])
        expected = {e for e in events
                    if any(e.startswith(f'{b}/') for b in spec['BIDS_events'])}
        assert len(expected) == 2, name
        assert selected == expected, name
