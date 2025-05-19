from io import StringIO

import pandas as pd
import requests

STRING = 'https://string-db.org/api'
SPECIES = 9606
CALLER_IDENTITY = 'www.idssp.ee.ntu.edu.tw'


def get_ppi_encoder(chosen_genes: list[str], score: str = 'score', threshold: float = 0.0, logger=None):
    """Get PPI for chosen genes

    Args:
        chosen_genes (list[str]): list of chosen genes

    Returns:
        ppi (pd.DataFrame): PPI network
    """
    gene_encoder = dict(zip(chosen_genes, range(len(chosen_genes))))

    if logger:
        logger.info(f'Link to STRING API: {_get_graph_link(chosen_genes)}')

    res_text = _get_ppi_network_from_string(chosen_genes)

    ppi = pd.read_csv(StringIO(res_text), sep='\t')
    ppi = ppi[['preferredName_A', 'preferredName_B', score]]
    ppi.drop_duplicates(inplace=True)
    ppi = ppi[ppi[score] >= threshold]
    #ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].map(lambda x: gene_encoder[x])
    ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].applymap(lambda x: gene_encoder[x]) #different versions of pandas
    return ppi


def _get_ppi_network_from_string(gene_list: list[str]):
    """Get PPI network from STRING API.

    Args:
        gene_list (list[str]): list of genes

    Returns:
        res_text (str): response from STRING API in text format
    """
    request_url = '/'.join([STRING, 'tsv', 'network'])

    # Speed up the process by getting the identifier first.
    if len(gene_list) > 100:
        gene_list = _get_identifier(gene_list).values()

    params = {
        'identifiers': '%0d'.join(gene_list),           # Genes
        'species': SPECIES,                             # Species NCBI identifier for homo sapiens
        'caller_identity': CALLER_IDENTITY              # App name
    }

    response = requests.post(request_url, data=params)
    if response.status_code != 200:
        raise ConnectionError(f'Responce from STRING: {response.status_code}')
    return response.text


def _get_identifier(gene_list: list[str]):
    """Each gene has corresponding identifiers in STRING. Get the best one out of them.

    Args:
        gene_list (list[str]): list of genes

    Returns:
        identifiers (dict[str, str]): corresponding identifiers
    """
    params = {
        'identifiers': '\r'.join(gene_list),            # Genes
        'species': SPECIES,                             # Species NCBI identifier Homo Sapiens
        'limit': 1,                                     # Return only one (best) identifier per protein
        'caller_identity': CALLER_IDENTITY              # App name
    }
    request_url = '/'.join([STRING, 'tsv-no-header', 'get_string_ids'])
    response = requests.post(request_url, data=params)

    if response.status_code != 200:
        raise ConnectionError(f'Responce from STRING: {response.status_code}')

    identifiers: dict[str, str] = {}
    for line in response.text.strip().split('\n'):
        counter, identifier, species_id, species, gene, meaning = line.split('\t')
        identifiers[gene] = identifier
    if set(identifiers.keys()) != set(gene_list):
        raise ValueError('Cannot get correct identifiers.')
    return identifiers


def _get_graph_link(gene_list: list[str]):
    params = {'identifiers': '\r'.join(gene_list), 'species': SPECIES, 'caller_identity': CALLER_IDENTITY}
    request_url = '/'.join([STRING, 'tsv-no-header', 'get_link'])
    res = requests.post(request_url, data=params)
    if res.status_code == 200:
        return res.text.strip()
    raise ConnectionError(f'Responce from STRING: {res.status_code}')


if __name__ == '__main__':
    BRCA = ['ESR1', 'EFTUD2', 'HSPA8', 'STAU1', 'SHMT2', 'ACTB', 'GSK3B',
            'YWHAB', 'UBXN6', 'PRKRA', 'BTRC', 'DDX23', 'SSR1', 'TUBA1C',
            'SNIP1', 'SRSF5', 'ERBB2', 'MKI67', 'PGR', 'PLAU']
    LUAD = ['HNRNPU', 'STAU1', 'KDM1A', 'SERBP1', 'DHX9', 'EMC1', 'SSR1',
            'PUM1', 'CLTC', 'PRKRA', 'KRR1', 'OCIAD1', 'CDC73', 'SLC2A1',
            'HIF1A', 'PKM', 'CADM1', 'EPCAM', 'ALCAM', 'PTK7']
    COAD = ['HNRNPL', 'HNRNPU', 'HNRNPA1', 'ZBTB2', 'SERBP1', 'RPL4', 'HNRNPK',
            'HNRNPR', 'TFCP2', 'DHX9', 'RNF4', 'PUM1', 'ABCC1', 'CD44',
            'ALCAM', 'ABCG2', 'ALDH1A1', 'ABCB1', 'EPCAM', 'PROM1']

    # Get PPI network for all genes.
    all_genes = list(set(BRCA + LUAD + COAD))
    ppi = get_ppi_encoder(all_genes)

    print(_get_identifier(all_genes))

    # Add reverse edges. (A, B) and (B, A) are the same.
    ppi = pd.concat([ppi, ppi.rename(columns={
        'preferredName_A': 'preferredName_B', 'preferredName_B': 'preferredName_A'
    })])

    # Get number of edges for each cancer type.
    print(ppi['preferredName_A'].isin(BRCA).sum())
    print(ppi['preferredName_A'].isin(LUAD).sum())
    print(ppi['preferredName_A'].isin(COAD).sum())
