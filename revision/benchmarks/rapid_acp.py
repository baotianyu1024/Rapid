"""Validate the offline per-table ACP decision before constructing HugeCTR."""
import json


def read_acp_plan(path, ranks, global_batch, hot):
    plan = json.loads(path.read_text())
    if (plan.get('schema') != 'rapid-hugectr-acp-v1' or plan['ranks'] != ranks or
            plan['local_batch'] * ranks != global_batch or plan['table_lengths'] != hot):
        raise ValueError('ACP plan schema, layout, ranks, or batch size does not match training')
    tables = plan['tables']
    if len(tables) != len(hot) or sorted(x['table'] for x in tables) != list(range(len(hot))):
        raise ValueError('ACP plan must specify each table exactly once')
    if any(x['strategy'] not in ('Unique', 'Reduction') for x in tables):
        raise ValueError('ACP plan contains an invalid strategy')
    if any(x['pooling'] != hot[x['table']] for x in tables):
        raise ValueError('ACP plan table pooling does not match training')
    return [x['table'] for x in tables if x['strategy'] == 'Unique']
