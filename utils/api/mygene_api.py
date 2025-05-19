import requests
from requests.adapters import HTTPAdapter, Retry

MYGENE_API_VERSION = 3
MYGENE_MAX_REQUEST = 500
MYGENE_MAX_THREADS = 64
MYGENE_API_BASEURL = f'https://mygene.info/v{MYGENE_API_VERSION}'
MYGENE_GENE_ENDPOINT = '/'.join([MYGENE_API_BASEURL, 'gene'])

#TODO for all
# Write warning handling
# Retry when failed

def _get_gene_annotation_from_mygene_api(ensembl_id, **kwargs):
    '''
    Get the a gene annotation using GET method with ensembl_ids from the MYGENE API.

    :param ensembl_id: An ensembl_id to transform. Please see the MYGENE API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the MYGENE API documentations for the valid parameters.
    '''
    params = kwargs

    for param in params:
        if type(params[param]) == list:
            params[param] = ','.join(params[param])

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.get('/'.join([MYGENE_GENE_ENDPOINT, ensembl_id]), params=params)
    response.raise_for_status()

    return response.json()

def _post_gene_annotations_from_mygene_api(ensembl_ids, **kwargs):
    '''
    Get the gene annotations using POST method with ensembl_ids from the MYGENE API.

    :param ensembl_ids: A list of ensembl_ids to transform. Please see the MYGENE API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the MYGENE API documentations for the valid parameters.
    '''
    data = f'ids={",".join(ensembl_ids)}'

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    params = kwargs

    for param in params:
        if type(params[param]) == list:
            params[param] = ','.join(params[param])

        #TODO Handle int situation
        data = '&'.join([data, '='.join([param, params[param]])])

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504], allowed_methods=frozenset(['GET', 'POST']))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.post(
        MYGENE_GENE_ENDPOINT,
        data=data,
        headers=headers,
    )
    response.raise_for_status()

    return response.json()

def get_gene_annotation(ensembl_id, method='GET', **kwargs):
    '''
    Get one gene annotation.

    :param ensembl_id: An ensembl_id to transform. Please see the MYGENE API documentations.
    :param method: The method you want to used for request.
    :param **kwargs: The parameters that you want to add in the request. Please see the MYGENE API documentations for the valid parameters.
    '''
    if method == 'GET':
        return _get_gene_annotation_from_mygene_api(ensembl_id=ensembl_id, **kwargs)
    elif method == 'POST':
        return _post_gene_annotations_from_mygene_api(ensembl_ids=[ensembl_id], **kwargs)
    else:
        raise KeyError('Wrong methods. Only accept GET and POST')

def get_gene_annotations(ensembl_ids, **kwargs):
    '''
    Get the gene annotations using POST method with ensembl_ids.

    :param ensembl_ids: A list of ensembl_ids to transform. Please see the MYGENE API documentations.
    :param **kwargs: The parameters that you want to add in the download. Please see the MYGENE API documentations for the valid parameters.
    '''
    return _post_gene_annotations_from_mygene_api(ensembl_ids=ensembl_ids, **kwargs)
